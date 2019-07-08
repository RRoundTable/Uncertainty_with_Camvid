import torch
import torch.nn as nn

class DenseLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, dropout=0.5):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(in_channels))
        self.add_module('relu', nn.ReLU(True))
        self.add_module('conv', nn.Conv2d(in_channels, growth_rate, kernel_size=3,
                                          stride=1, padding=1, bias=True))
        self.add_module('drop', nn.Dropout2d(dropout))

    def forward(self, input):
        return super().forward(input)

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers, dropout, upsample=False):
        super().__init__()
        self.upsample = upsample
        self.layers = nn.ModuleList([DenseLayer(in_channels + i * growth_rate,
                                                growth_rate, dropout)
                                     for i in range(n_layers)])

    def forward(self, x):
        if self.upsample:
            new_feature = []
            # we pass all previous activations into each dense layer normally
            # but we only store each dense layer's output in the new_feature array
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], dim=1)
                new_feature.append(out)
            return torch.cat(new_feature, dim=1)
        else:
            for layer in self.layers:
                out = layer(x)
                x = torch.cat([x, out], dim=1)
            return x


class TransitionDown(nn.Sequential):
    def __init__(self, in_channels, dropout=0.5):
        super().__init__()
        self.add_module('norm', nn.BatchNorm2d(num_features=in_channels))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(in_channels, in_channels,
                                          kernel_size=1, stride=1,
                                          padding=0, bias=True))
        self.add_module('drop', nn.Dropout2d(dropout))
        self.add_module('maxpool', nn.MaxPool2d(2))

    def forward(self, x):
        return super().forward(x)

class TransitionUp(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.convTrans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=3,
            stride=2, padding=0, bias=True
        )

    def forward(self, x, skip):
        out = self.convTrans(x)
        out = center_crop(out, skip.size(2), skip.size(3))
        out = torch.cat([out, skip], 1)
        return out

class Bottlenect(nn.Sequential):
    def __init__(self, in_channels, growth_rate, n_layers, dropout):
        super().__init__()
        self.add_module('bottleneck',
                        DenseBlock(in_channels, growth_rate,
                                   n_layers, dropout, upsample=True))

    def forward(self, x):
        return super().forward(x)

def center_crop(layer, max_height, max_width):
    _, _, h, w = layer.size()
    xy1 = (w - max_width) // 2
    xy2 = (h - max_height) // 2
    return layer[:, :, xy2:(xy2 + max_height), xy1:(xy1 + max_width)]

