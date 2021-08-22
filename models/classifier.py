import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert args.img_width == 51 and args.img_height == 23
        layers = [
            nn.Conv2d(3, 16, 3, 1, 1),  # 16x51x23
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, 1, 1),  # 32x51x23
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1),  # 64x26x12
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),  # 128x13x6
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        ]
        self.conv = nn.Sequential(*layers)

        dim_fc = 4 * args.num_classes
        self.fc = nn.ModuleList()
        for i in range(args.num_chars):
            layers = [
                nn.Linear(128 * 13 * 6, dim_fc),
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
