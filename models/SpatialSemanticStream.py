import torch
import torch.nn as nn

from models.SemanticStream import SemanticStream
from models.SpatialStream import SpatialStream

class SpatialSemanticStream(nn.Module):
  def __init__(self, channels_in, pick, batchnorm = False):
    super().__init__()
    self.channels_in = channels_in
    self.batchnorm = batchnorm
    self.pick = pick
    if self.pick: 
        self.channels_out = 1
    else: 
        self.channels_out = 3
    self._make_layers()

  def _make_layers(self):
    # spatial
    self.spatial = SpatialStream(self.channels_in, self.channels_out, self.batchnorm)
    # semantic
    self.semantic = SemanticStream(self.channels_out, self.batchnorm)
    # Merging
    if self.pick:
        self.merge = torch.add
    else:
        self.merge = nn.Conv2d(2*self.channels_out, self.channels_out, kernel_size=1, stride=1)

  def forward(self, rgb_ddd_img, language_command):
    out_spatial, lateral_outs = self.spatial(rgb_ddd_img)
    rgb_image = rgb_ddd_img[:,:3,:,:]
    out_semantics = self.semantic(rgb_image, language_command, lateral_outs)
    if self.pick:
        out_semantics = self.merge(out_spatial, out_semantics)
    else: 
        out_semantics = self.merge(torch.cat((out_spatial, out_semantics), axis=1))

    return out_semantics
