import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from models.SpatialSemanticStream import SpatialSemanticStream

class PickModel(nn.Module):
  def __init__(self, num_rotations, batchnorm = False):
    super().__init__()
    self.num_rotations = num_rotations
    self.model = SpatialSemanticStream(channels_in=6, pick=True, batchnorm=batchnorm)

  def forward(self, rgb_ddd_img, language_command):
    # Rotate input multiple times and run forward for each one
    out_all = []
    for i in range(self.num_rotations):
      angle = 360/self.num_rotations * i
      rotated_img = TF.rotate(rgb_ddd_img, angle, torchvision.transforms.InterpolationMode.BILINEAR)
      # Un-rotate output
      out = self.model.forward(rotated_img, language_command)
      out = TF.rotate(out, -angle, torchvision.transforms.InterpolationMode.BILINEAR)
      out_all.append(out)
    out_all = torch.cat(out_all, dim=1)
    return out_all
