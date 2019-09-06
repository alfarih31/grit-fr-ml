import torch.nn as nn
#from torchsummary import summary

def seperableconv(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, 2, padding=1, groups=inp),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),
        nn.Conv2d(inp, oup, 1),
        nn.BatchNorm2d(oup),
    )

def block(inp, oup, stride=2):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, padding=1, groups=inp),
        nn.BatchNorm2d(oup)
    )

def increase_ch(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, use_batch_norm=True, onnx_compatible=False):
        super(InvertedResidual, self).__init__()
        ReLU = nn.ReLU if onnx_compatible else nn.ReLU6

        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expand_ratio == 1:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
        else:
            if use_batch_norm:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                )
            else:
                self.conv = nn.Sequential(
                    # pw
                    nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                    ReLU(inplace=True),
                    # dw
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                    ReLU(inplace=True),
                    # pw-linear
                    nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class FPN(nn.Module):
    def __init__(self, input_channel):
        super(FPN, self).__init__()
        # n, c, f
        self.loops = [[2, 16, 1], # 150
                      [3, 24, 2], # 75
                      [6, 48, 4], # 38
                      [6, 64, 5], # 19
                      [6, 96, 5], # 10
                      [3, 160, 2], # 5
                      [2, 320, 1]] # 3
        self.features = []
        inp = input_channel
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
        for n, c, f in self.loops:
            oup = c
            self.features.append(increase_ch(inp, oup))
            for _ in range(n):
                self.features.append(block(inp, inp, stride=1))
            self.features.append(seperableconv(inp, oup))
            for i in range(f):
                if i == 0:
                    self.features.append(InvertedResidual(oup, oup, 2, 1, onnx_compatible=True))
                else:
                    self.features.append(InvertedResidual(oup, oup, 1, 6, onnx_compatible=True))
            inp = c
        self.features = nn.Sequential(*self.features)

    def forward(self, x):
        start = 0
        end = 0
        for i, loop in enumerate(self.loops):
            n, _, f = loop
            end = start + n + 1
            for j, layer in enumerate(self.features[start:end]):
                if j == 0:
                    y = layer(x)
                else:
                    x = x + layer(x)
            x = self.features[end](x)
            x = self.upsampler[i](x)
            x += y
            start = end+1
            end = start+f
            for layer in self.features[start:end]:
                x = layer(x)
            start = end
        return x

#net = FPN(3)
#summary(net, (3, 300, 300), 1)
