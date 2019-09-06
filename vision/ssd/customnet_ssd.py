import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.mobilenet_v2 import InvertedResidual
from ..nn.custom import CustomNet

from .customssd import CUSTOMSSD
from .predictor import Predictor
from .config import customnet_ssd_config as config

def activate_sum(onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.LeakyReLU
    return ReLU()


def SeperableConv2d(in_channels, out_channels, kernel_size=1, stride=1,
        padding=0, use_batch_norm=True,onnx_compatible=False):
    """Replace Conv2d with a depthwise Conv2d and Pointwise Conv2d.
    """
    ReLU = nn.ReLU if onnx_compatible else nn.LeakyReLU
    if use_batch_norm:
        return Sequential(
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                groups=in_channels, stride=stride, padding=padding),
            BatchNorm2d(in_channels),
            ReLU(),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )
    else:
        return Sequential(
            Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                groups=in_channels, stride=stride, padding=padding),
            ReLU(),
            Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
        )

def Separabledeconv(inp, oup, kernel, padding=0, onnx_compatible=False):
    ReLU = nn.ReLU if onnx_compatible else nn.LeakyReLU
    return nn.Sequential(
        # dw
        nn.ConvTranspose2d(inp, inp, kernel, 2, padding, groups=inp, bias=False),
        BatchNorm2d(inp),
        ReLU(),

        # pw
        Conv2d(inp, oup, 1, bias=False),
        BatchNorm2d(oup),
    )

def create_customnet_ssd(num_classes, is_test=False):
    base_net = CustomNet()

    extra_layers = ModuleList([
        InvertedResidual(1260, 1024, 2, 0.25), # 38 to 19
        InvertedResidual(1024, 512, 2, 0.25), # 19 to 10
        InvertedResidual(512, 256, 2, 0.4), # 10 to 5
        InvertedResidual(256, 256, 2, 0.5), # 5 to 3
        Conv2d(256, 128, 3, 2),
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=1260, out_channels=4 * 4, kernel_size=3, padding=1,), #38 576
        Conv2d(in_channels=1024, out_channels=6 * 4, kernel_size=3, padding=1,), #19 1024
        Conv2d(in_channels=512, out_channels=6 * 4, kernel_size=3, padding=1,), #10 512
        Conv2d(in_channels=256, out_channels=6 * 4, kernel_size=3, padding=1,), #5 256
        Conv2d(in_channels=256, out_channels=4 * 4, kernel_size=3, padding=1,), #3 256
        Conv2d(in_channels=128, out_channels=4 * 4, kernel_size=3, padding=1,), #1 256
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=1260, out_channels=4 * num_classes, kernel_size=3, padding=1,), #38 576
        Conv2d(in_channels=1024, out_channels=6 * num_classes, kernel_size=3, padding=1,), #19 1024
        Conv2d(in_channels=512, out_channels=6 * num_classes, kernel_size=3, padding=1,), #10 512
        Conv2d(in_channels=256, out_channels=6 * num_classes, kernel_size=3, padding=1,), #5 256
        Conv2d(in_channels=256, out_channels=4 * num_classes, kernel_size=3, padding=1,), #3 256
        Conv2d(in_channels=128, out_channels=4 * num_classes, kernel_size=3, padding=1,), #1 256
    ])

    return CUSTOMSSD(num_classes, base_net, extra_layers, classification_headers, regression_headers,
                    config=config)


def create_customnet_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
