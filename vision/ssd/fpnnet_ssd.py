import torch
from torch.nn import Conv2d, Sequential, ModuleList, BatchNorm2d
from torch import nn
from ..nn.mobilenet_v2 import InvertedResidual
from ..nn.fpn import FPN

from .fpnssd import FPNSSD
from .predictor import Predictor
from .config import fpnnet_ssd_config as config

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

def upsample(inp, oup):
    return Sequential(
        nn.ReLU(),
        nn.Conv2d(inp, oup, 1),
        nn.BatchNorm2d(oup),
    )

def create_fpnnet_ssd(num_classes, is_test=False):
    base_net = FPN(3)

    extra_layers = ModuleList([
        InvertedResidual(320, 512, 2, 0.15, onnx_compatible=True)
    ])

    regression_headers = ModuleList([
        Conv2d(in_channels=288, out_channels=4 * 4, kernel_size=3, padding=1), #38 576
        Conv2d(in_channels=64, out_channels=6 * 4, kernel_size=3, padding=1,), #19 1024
        Conv2d(in_channels=384, out_channels=6 * 4, kernel_size=3, padding=1,), #19 1024
        Conv2d(in_channels=576, out_channels=6 * 4, kernel_size=3, padding=1,), #10 512
        Conv2d(in_channels=960, out_channels=6 * 4, kernel_size=3, padding=1,), #5 256
        Conv2d(in_channels=320, out_channels=6 * 4, kernel_size=3, padding=1,), #3 256
        Conv2d(512, 6 * 4, 1),
    ])

    classification_headers = ModuleList([
        Conv2d(in_channels=288, out_channels=4 * num_classes, kernel_size=3, padding=1), #38 576
        Conv2d(in_channels=64, out_channels=6 * num_classes, kernel_size=3, padding=1,), #19 1024
        Conv2d(in_channels=384, out_channels=6 * num_classes, kernel_size=3, padding=1,), #19 1024
        Conv2d(in_channels=576, out_channels=6 * num_classes, kernel_size=3, padding=1,), #10 512
        Conv2d(in_channels=960, out_channels=6 * num_classes, kernel_size=3, padding=1,), #5 256
        Conv2d(in_channels=320, out_channels=6 * num_classes, kernel_size=3, padding=1,), #3 256
        Conv2d(512, 6 * num_classes, 1),
    ])

    return FPNSSD(num_classes, base_net.features, extra_layers, base_net.loops, classification_headers, regression_headers,
                    config=config, is_test=is_test)


def create_fpn_ssd_predictor(net, candidate_size=200, nms_method=None, sigma=0.5, device=torch.device('cpu')):
    predictor = Predictor(net, config.image_size, config.image_mean,
                          config.image_std,
                          nms_method=nms_method,
                          iou_threshold=config.iou_threshold,
                          candidate_size=candidate_size,
                          sigma=sigma,
                          device=device)
    return predictor
