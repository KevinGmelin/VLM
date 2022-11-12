import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF

from SpatialSemanticStream import SpatialSemanticStream


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

        # Query net forward
        large_crop_dim = int(2**0.5 * self.crop_size)  # Hypotenuse length
        large_crop_img = TF.crop(
            rgb_ddd_img,
            pick_location[0] - large_crop_dim // 2,
            pick_location[1] - large_crop_dim // 2,
            large_crop_dim,
            large_crop_dim,
        )
        query_list = []
        for i in range(self.num_rotations):
            # Rotate
            angle = 360 / self.num_rotations * i
            rotated_img = TF.rotate(
                large_crop_img, angle, torchvision.transforms.InterpolationMode.BILINEAR
            )
            center_coord = large_crop_dim // 2
            small_crop_img = TF.crop(
                rotated_img,
                center_coord - self.crop_size[0] // 2,
                center_coord - self.crop_size[1] // 2,
                self.crop_size[0],
                self.crop_size[1],
            )
            query_list.append(self.query_net(small_crop_img, language_command))
        query_tensor = torch.Tensor(query_list)
        assert query_tensor.shape[1] is 1
        query_tensor = query_tensor[:, 0]

        # Key net forward
        key_tensor = self.key_net(rgb_ddd_img, language_command)
        # Cross correlation
        out = nn.functional.conv2d(key_tensor, query_tensor, padding='same')
        return out
