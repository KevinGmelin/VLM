import torch
import torchvision
import torchvision.transforms.functional as TF

from SpatialSemanticStream import SpatialSemanticStream

class PlaceQuerry(SpatialSemanticStream):
  def __init__(self, num_rotations, crop_size, batchnorm = False):
    # crop_size: tuple for 2D dimensions for cropping size
    super().__init__(6, False, batchnorm)
    self.num_rotations = num_rotations
    self.crop_size = crop_size

  def forward(self, rgb_ddd_img, language_command, pick_coord):
    # pick_coord: tuple for 2D pick coordinates
    # Do forward on input once, then crop-rotate-crop output multiple times
    out_spatial, out_semantics = super().forward(rgb_ddd_img, language_command)
    # Crop
    large_crop_dim = int((self.crop_size[0]**2 + self.crop_size[1]**2)**0.5) # Hypotenuse length
    large_crop_spatial = TF.crop(out_spatial,
                                 pick_coord[0]-large_crop_dim//2, pick_coord[1]-large_crop_dim//2,
                                 large_crop_dim, large_crop_dim)
    large_crop_semantics = TF.crop(out_semantics,
                                   pick_coord[0]-large_crop_dim//2, pick_coord[1]-large_crop_dim//2,
                                   large_crop_dim, large_crop_dim)
    for i in range(self.num_rotations):
      # Rotate
      angle = 360/self.num_rotations * i
      rotated_spatial = TF.rotate(large_crop_spatial, angle, torchvision.transforms.InterpolationMode.BILINEAR)
      rotated_semantics = TF.rotate(large_crop_semantics, angle, torchvision.transforms.InterpolationMode.BILINEAR)
      # Crop
      rotated_center = large_crop_dim//2
      small_crop_spatial = TF.crop(rotated_spatial,
                                   rotated_center-self.crop_size[0]//2, rotated_center-self.crop_size[1]//2,
                                   self.crop_size[0], self.crop_size[1])
      small_crop_semantics = TF.crop(rotated_semantics,
                                     rotated_center-self.crop_size[0]//2, rotated_center-self.crop_size[1]//2,
                                     self.crop_size[0], self.crop_size[1])
      # Concat
      if(i == 0):
        out_spatial_all = small_crop_spatial
        out_semantics_all = small_crop_semantics
      else:
        out_spatial_all = torch.cat((out_spatial_all, small_crop_spatial), dim=1)
        out_semantics_all = torch.cat((out_semantics_all, small_crop_semantics), dim=1)
    return out_spatial_all, out_semantics_all

class PlaceValue(SpatialSemanticStream):
  def __init__(self, batchnorm = False):
    super().__init__(6, False, batchnorm)

  def forward(self, rgb_ddd_img, language_command):
    out_spatial, out_semantics = super().forward(rgb_ddd_img, language_command)
    return out_spatial, out_semantics
