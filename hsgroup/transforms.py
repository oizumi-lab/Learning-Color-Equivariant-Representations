from functools import partial

import numpy as np
from PIL import Image
import torch
import torchvision

import utils.color_utils as utils


# Define a custom transform to scale all channels by a random factor
class RandomScaling:
    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, img: Image.Image):
        rgb = np.array(img, dtype=np.float32)
        random_scale = np.random.rand(1)
        if self.dataset == "r":
            # set g, b channels to 0 in final image. Preserve r channel
            rgb[:, :, 1:] = 0
        elif self.dataset == "rg":
            rgb[:, :, 2] = 0
            rgb[:, :, np.random.randint(0, 2, (1,))] *= random_scale
        elif self.dataset == "blues":
            deleted_channel = np.random.randint(0, 2, (1,))
            rgb[:, :, deleted_channel] = 0
            modified_channel = np.random.choice([1 - deleted_channel[0], 2])
            rgb[:, :, modified_channel] *= random_scale
        elif self.dataset == "rgb":
            permutation = np.random.permutation(3)
            rgb[:, :, permutation[0]] *= random_scale
            rgb[:, :, permutation[1]] = 0
        else:
            raise ValueError("Invalid dataset. Select from r, rg, blues, rgb")
        return Image.fromarray(rgb.astype(np.uint8))


class RandomColor:
    def __init__(self):
        pass

    def __call__(self, img: Image.Image):
        # Randomly rotate hue
        hue_rotation = np.random.randint(0, 256)
        return utils.rotate_hue(img, hue_rotation, rgb_out=True)


class HueSeparation:
    def __init__(self, n_groups, rgb=True):
        self.n_groups = n_groups
        self.rgb = rgb

    def __call__(self, img):
        img_hsv = img.convert("HSV")
        rotate = partial(
            utils.rotate_hue,
            img_hsv,
            rgb_out=self.rgb,
            rgb_in=False
        )
        x_transformed = []
        for i in range(self.n_groups):
            # hue rotation; hue range: [0, 255]
            hue_idx = i // self.n_groups_saturation
            rot_amount = hue_idx * 256 // self.n_groups
            x = rotate(rot_amount)
            
            # append tensor
            x = torch.tensor(np.array(x), dtype=torch.float32).permute(2, 0, 1) / 255
            x_transformed.append(x)
            
        return torch.stack(x_transformed, dim=0) # (n_groups, 3, H, W)


class HueLuminanceSeparation:
    def __init__(self, n_groups, n_groups_saturation, rgb=True, frac_space=0.2):
        self.n_groups = n_groups
        self.n_groups_saturation = n_groups_saturation
        self.rgb = rgb
        self.frac_space = frac_space

    def __call__(self, img: Image.Image) -> torch.Tensor:
        img_hsv = img.convert("HSV")
        rotate = partial(
            utils.rotate_hue,
            img_hsv,
            rgb_out=False,
            rgb_in=False
        )
        shift = partial(
            utils.scale_saturation,
            rgb_out=self.rgb,
            rgb_in=False
        )
        x_transformed = []
        for i in range(self.n_groups * self.n_groups_saturation):
            # hue rotation; hue range: [0, 255]
            hue_idx = i // self.n_groups_saturation
            rot_amount = hue_idx * 256 // self.n_groups
            x = rotate(rot_amount)
        
            # stulation shift; satulation range: [-255, 255]
            sat_idx = i % self.n_groups_saturation
            mid_sat_group = self.n_groups_saturation // 2
            scale_factor = 256 / 2 * self.frac_space
            shift_amount = int(
                (sat_idx - mid_sat_group) / mid_sat_group * scale_factor
            )
            x = shift(x, shift_amount)

            # append tensor
            x = torch.tensor(np.array(x), dtype=torch.float32).permute(2, 0, 1) / 255
            x_transformed.append(x)
            
        return torch.stack(x_transformed, dim=0) # (n_groups * n_groups_saturation, 3, H, W)


class TensorReshape:
    def __init__(self):
        pass

    def __call__(self, img):
        # Recall this will be applied to one image, so the output should be three (rather than four) dimensional
        return img.view(-1, img.shape[-2], img.shape[-1])
