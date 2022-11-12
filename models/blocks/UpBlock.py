# https://arxiv.org/pdf/2109.12098.pdf
import torch
import torch.nn as nn

class UpBlock(nn.Module):
    def __init__(self, channels_in, channels_out, bilinear=True, batchnorm=False):
        super().__init__()
        self.batchnorm = batchnorm

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(channels_in, channels_in//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_in//2) if self.batchnorm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_in//2, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out) if self.batchnorm else nn.Identity(),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        out = torch.cat([x1, x2], dim=1)
        out = self.conv(out)
        return out