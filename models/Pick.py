import torch
import torchvision
import torchvision.transforms.functional as TF

class Pick(SpatialSemanticStream):
  def __init__(self, num_rotations, batchnorm = False):
    super().__init__(6, True, batchnorm)
    self.num_rotations = num_rotations

  def forward(self, rgb_ddd_img, language_command):
    # Rotate input multiple times and run forward for each one
    for i in range(self.num_rotations):
      angle = 360/self.num_rotations * i
      rotated_img = TF.rotate(rgb_ddd_img, angle, torchvision.transforms.InterpolationMode.BILINEAR)
      # Un-rotate output
      out_spatial, out_semantics = super().forward(rotated_img, language_command)
      out_spatial = TF.rotate(out_spatial, -angle, torchvision.transforms.InterpolationMode.BILINEAR)
      out_semantics = TF.rotate(out_semantics, -angle, torchvision.transforms.InterpolationMode.BILINEAR)
      if(i == 0):
        out_spatial_all = out_spatial
        out_semantics_all = out_semantics
      else:
        out_spatial_all = torch.cat((out_spatial_all, out_spatial), dim=1)
        out_semantics_all = torch.cat((out_semantics_all, out_semantics), dim=1)
    return out_spatial_all, out_semantics_all
