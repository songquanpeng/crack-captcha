import torch
from torch import nn


class VGG(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        init_dim = 64
        max_dim = 256
        layers = [
            VGGConvBlock(2, 3, init_dim),  # 64
            VGGConvBlock(2, init_dim, init_dim * 2),  # 128
            VGGConvBlock(3, init_dim * 2, max_dim),  # 256
            VGGConvBlock(3, max_dim, max_dim),  # 256
        ]
        self.conv = nn.Sequential(*layers)
        dim_conv = 768
        dim_fc = 4 * args.num_classes
        self.fc = nn.ModuleList()
        for i in range(args.num_chars):
            layers = [
                nn.Linear(dim_conv, dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(dim_fc, dim_fc),
                nn.ReLU(inplace=True),
                nn.Linear(dim_fc, args.num_classes)
            ]
            self.fc.append(nn.Sequential(*layers))

    def forward(self, x):
        h = self.conv(x)
        h = h.view(x.shape[0], -1)
        out = []
        for head in self.fc:
            out.append(head(h))
        out = torch.cat(out)
        return out


class VGGConvBlock(nn.Module):
    def __init__(self, num_conv, in_dim, out_dim, kernel_size=3, stride=1):
        super().__init__()
        layers = []
        for i in range(num_conv):
            layers.extend([
                nn.Conv2d(in_dim, out_dim, kernel_size, stride, padding=1),
                nn.ReLU()
            ])
            in_dim = out_dim
        layers.append(nn.MaxPool2d(2, 2))
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)
