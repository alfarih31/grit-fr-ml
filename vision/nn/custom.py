import torch
import torch.nn as nn

def block(inp):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, stride=1, padding=1, groups=inp),
        nn.BatchNorm2d(inp),
    )

def downsize(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, inp, kernel_size=3, padding=1, stride=2, groups=inp), # DW
        nn.BatchNorm2d(inp),
    )

def create_block(inp, oup):
    layers = []
    for _ in range(int(oup/inp)):
        layers.append(block(inp))
    return nn.ModuleList(layers)

class CustomNet(nn.Module):
    def __init__(self, input_size=300, input_channel=3):
        super(CustomNet, self).__init__()
        self.features = nn.ModuleList()
        self.ReLU = nn.ReLU(inplace=True)
        self.features.append(create_block(input_channel, input_channel*4)),

        self.features.append(downsize(input_channel*4, input_channel*4)) # To 150 0

        self.features.append(create_block(12, 60))

        self.features.append(downsize(60, 60)) # To 75 2

        self.features.append(create_block(60, 420))

        self.features.append(downsize(420, 420)) # To 38 3

        self.features.append(create_block(420, 1260))

    def forward(self, x):
        for layers in self.features:
            if isinstance(layers, nn.Sequential):
                x = layers(x)
            elif isinstance(layers, nn.ModuleList):
                for i, layer in enumerate(layers):
                    if i == 0:
                        y = layer(x)
                    else:
                        y = torch.cat((y, layer(x)), dim=1)
                x = self.ReLU(y)

        return x
