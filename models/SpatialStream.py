import torch
import torch.nn as nn
import torch.nn.functional as F

from models.blocks.ConvBlock import ConvBlock

class SpatialStream(nn.Module):
    def __init__(self, channels_in, channels_out, batchnorm = False):
      super().__init__()
      self.channels_in = channels_in
      self.channels_out = channels_out
      self.batchnorm = batchnorm
      self._make_layers()
    
    def _make_layers(self):
      # 1x conv
      self.conv = nn.Sequential(
          nn.Conv2d(self.channels_in, 64, kernel_size=3, stride=1, padding=1),
          nn.BatchNorm2d(64) if self.batchnorm else nn.Identity(),
          nn.ReLU(True)
      )
      # Encoder: 6x conv & identity (downsample)
      self.encoder1 = nn.Sequential(
          ConvBlock(64, 64, 1, self.batchnorm),
          ConvBlock(64, 64, 1, self.batchnorm, False, True)
      )
      self.encoder2 = nn.Sequential(
          ConvBlock(64, 128, 2, self.batchnorm),
          ConvBlock(128, 128, 1, self.batchnorm, False, True)
      )
      self.encoder3 = nn.Sequential(
          ConvBlock(128, 256, 2, self.batchnorm),
          ConvBlock(256, 256, 1, self.batchnorm, False, True)
      )
      self.encoder4 = nn.Sequential(
          ConvBlock(256, 256, 2, self.batchnorm),
          ConvBlock(256, 256, 1, self.batchnorm, False, True)
      )
      self.encoder5 = nn.Sequential(
          ConvBlock(256, 512, 2, self.batchnorm),
          ConvBlock(512, 512, 1, self.batchnorm, False, True)
      )
      self.encoder6 = nn.Sequential(
          ConvBlock(512, 1024, 2, self.batchnorm),
          ConvBlock(1024, 1024, 1, self.batchnorm, False, True)
      )
      # Decoder: 6x conv & identity & upsample
      self.decoder1 = nn.Sequential(
          ConvBlock(1024, 1024, 1, self.batchnorm),
          ConvBlock(1024, 1024, 1, self.batchnorm, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder2 = nn.Sequential(
          ConvBlock(1024, 512, 1, self.batchnorm),
          ConvBlock(512, 512, 1, self.batchnorm, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder3 = nn.Sequential(
          ConvBlock(512, 256, 1, self.batchnorm),
          ConvBlock(256, 256, 1, self.batchnorm, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder4 = nn.Sequential(
          ConvBlock(256, 128, 1, self.batchnorm),
          ConvBlock(128, 128, 1, self.batchnorm, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder5 = nn.Sequential(
          ConvBlock(128, 64, 1, self.batchnorm),
          ConvBlock(64, 64, 1, self.batchnorm, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      self.decoder6 = nn.Sequential(
          ConvBlock(64, 32, 1, self.batchnorm),
          ConvBlock(32, 32, 1, self.batchnorm, False, True),
          nn.UpsamplingBilinear2d(scale_factor=2)
      )
      # 1x conv & identity & downsample
      self.down = nn.Sequential(
          ConvBlock(32, 16, 1, self.batchnorm, self.channels_out),
          ConvBlock(self.channels_out, 16, 1, self.batchnorm, self.channels_out, True),
      )

    def forward(self, rgb_ddd_img):
      # Conv
      out = self.conv(rgb_ddd_img)
      #Encoder
      for encode in [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5, self.encoder6]:
        out = encode(out)
      #Decoder
      lateral_out = []
      for decode in [self.decoder1, self.decoder2, self.decoder3, self.decoder4, self.decoder5, self.decoder6]:
        out = decode(out)
        lateral_out.append(out)
      # Downsample
      out = self.down(out)
      out = F.interpolate(out, size=(rgb_ddd_img.shape[-2], rgb_ddd_img.shape[-1]), mode='bilinear')
      # Return output and lateral output
      return out, lateral_out
