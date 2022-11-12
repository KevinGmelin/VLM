import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from models.SpatialSemanticStream import SpatialSemanticStream


class PlaceModel(nn.Module):
    def __init__(self, num_rotations, crop_size, batchnorm=False):
        super().__init__()
        self.num_rotations = num_rotations
        self.crop_size = crop_size
        self.batchnorm = batchnorm
        self.query_net = SpatialSemanticStream(
            channels_in=6, pick=False, batchnorm=batchnorm
        )
        self.key_net = SpatialSemanticStream(
            channels_in=6, pick=False, batchnorm=batchnorm
        )

    def forward(self, rgb_ddd_img, language_command, pick_location):
        # TODO: Support batch size greater than 1 in place model
        assert (
            rgb_ddd_img.shape[0] == 1
        ), "Place model currently only supports a batch size of 1"

        query_tensor = self.query_net(rgb_ddd_img, language_command)
        
        large_crop_dim = int(2**0.5 * self.crop_size)  # Hypotenuse length
        large_crop = TF.crop(
            query_tensor,
            pick_location[0] - large_crop_dim // 2,
            pick_location[1] - large_crop_dim // 2,
            large_crop_dim,
            large_crop_dim,
        )
        rotated_query_list = []
        for i in range(self.num_rotations):
            # Rotate
            angle = 360 / self.num_rotations * i
            rotated_img = TF.rotate(
                large_crop, angle, torchvision.transforms.InterpolationMode.BILINEAR
            )
            center_coord = large_crop_dim // 2
            small_crop_img = TF.crop(
                rotated_img,
                center_coord - self.crop_size // 2,
                center_coord - self.crop_size // 2,
                self.crop_size,
                self.crop_size,
            )
            rotated_query_list.append(small_crop_img)
        all_crops = torch.stack(rotated_query_list)
        assert all_crops.shape[1] == 1
        all_crops = all_crops[:, 0]

        # Key net forward
        key_tensor = self.key_net(rgb_ddd_img, language_command)
        # Cross correlation
        out = nn.functional.conv2d(key_tensor, all_crops, padding='same')
        return out
