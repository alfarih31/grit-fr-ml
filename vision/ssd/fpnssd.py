import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from ..utils import box_utils

class FPNSSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, extra_layers: nn.ModuleList, loops: List,
                 classification_headers: nn.ModuleList, regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(FPNSSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.extra_layers = extra_layers
        self.is_test = is_test
        self.config = config
        self.regression_headers = regression_headers
        self.classification_headers = classification_headers
        self.loops = loops
        self.upsampler = ([
            nn.Upsample(size=(300, 300), mode='bilinear', align_corners=True),
            nn.Upsample(size=(150, 150), mode='bilinear', align_corners=True),
            nn.Upsample(size=(75, 75), mode='bilinear', align_corners=True),
            nn.Upsample(size=(38, 38), mode='bilinear', align_corners=True),
            nn.Upsample(size=(19, 19), mode='bilinear', align_corners=True),
            nn.Upsample(size=(10, 10), mode='bilinear', align_corners=True),
            nn.Upsample(size=(5, 5), mode='bilinear', align_corners=True),
            nn.Upsample(size=(3, 3), mode='bilinear', align_corners=True),
        ])

        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start = 0
        end = 0
        header_index = 0
        y = x
        for i, loop in enumerate(self.loops):
            n, _, f = loop
            end = start + n + 1
#            z = x
            for j, layer in enumerate(self.base_net[start:end]):
                if j == 0:
                    y = layer(x)
                else:
                    x = x + layer(x)
            x = self.base_net[end](x)
            x = self.upsampler[i](x)
            x = x + y
            start = end+1
            end = start+f
            for k, layer in enumerate(self.base_net[start:end]):
                if k == f-1:
                    path = getattr(layer, 'conv')
                    for _lay in path[:-2]:
                        x = _lay(x)
                    y = x
                    for _lay in path[-2:]:
                        x = _lay(x)
                    if i == 3:
                        confidence, location = self.compute_headers(header_index, x)
                        confidences.append(confidence)
                        locations.append(location)
                        header_index += 1
                else:
                    x = layer(x)
            if i >= 2:
                confidence, location = self.compute_headers(header_index, y)
                confidences.append(confidence)
                locations.append(location)
                header_index += 1
            start = end

        for layer in self.extra_layers:
            x = layer(x)
            confidence, location = self.compute_headers(header_index, x)
            confidences.append(confidence)
            locations.append(location)
            header_index += 1
        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations

    def compute_headers(self, i, x):
        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()
        location = location.view(location.size(0), -1, 4)

        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if not (k.startswith("classification_headers")
                                                              or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.regression_headers.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
#        self.extra_layers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.extra_layers.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance, self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
