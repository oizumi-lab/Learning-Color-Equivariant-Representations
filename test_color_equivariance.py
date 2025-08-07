import argparse
import json

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import lightning_model
from utils.color_utils import rotate_hue
from hsgroup.transforms import HueSeparation, HueLuminanceSeparation, TensorReshape
from datasets.dataloaders import MEAN, STD, get_cifar


def _unnormalize_cifar(tensor: torch.Tensor) -> torch.Tensor:
    """
    unnormalize image tensors loaded from CIFAR datasets.
    """
    if tensor.dim() != 3 or tensor.shape[0] != 3:
        raise ValueError(f"Expected tensor of shape (3,H,W), got {tensor.shape}")
    device = tensor.device
    mean = torch.tensor(MEAN["cifar"]).to(device).view(3, 1, 1)
    std  = torch.tensor(STD["cifar"]).to(device).view(3, 1, 1)
    img = tensor * std + mean
    return img.clamp(0.0, 1.0)


def to_pil_from_normalized(normalized: torch.Tensor) -> Image.Image:
    """
    Convert normalized tensor to a PIL Image.
    The tensor is expected to be in the shape of (3, H, W) and normalized
    using CIFAR dataset loader
    """
    img = _unnormalize_cifar(normalized)               # (3,H,W), in [0,1]
    img = img.permute(1, 2, 0).cpu().numpy()             # (H,W,3)
    arr = (img * 255.0).round().astype("uint8")
    return Image.fromarray(arr)


def create_lifting_layer(params: dict) -> transforms.Compose:
    """
    Creates the lifting layer based on the number of hue and saturation groups.
    """
    n_h = params["n_groups_hue"]
    n_s = params["n_groups_saturation"]
    assert n_h > 0, f"got invalid n_groups_hue, {n_h}"
    assert n_s > 0, f"got invalid n_groups_saturation, {n_s}"

    if n_s == 1:
        transform_list = [
            HueSeparation(n_h),
            transforms.Normalize(MEAN["cifar"], STD["cifar"]),
            TensorReshape(),
        ]
    else:
        transform_list = [
            HueLuminanceSeparation(n_h, n_s),
            transforms.Normalize(MEAN["cifar"], STD["cifar"]),
            TensorReshape(),
        ]
    return transforms.Compose(transform_list)


class PilImageLoader:
    """
    A wrapper for a dataloader that yields PIL images.
    It unnnormalizes and converts image tensors from the original dataloader.
    """
    def __init__(self, dataloader: torch.utils.data.DataLoader):
        self.dataloader = dataloader

    def __iter__(self):
        for data in self.dataloader:
            images, _ = data  # Assuming (image_tensor, label) format
            for image_tensor in images:
                yield to_pil_from_normalized(image_tensor)

    def __len__(self):
        return len(self.dataloader.dataset)


class Subnetwork(nn.Module):
    def __init__(self, model, top_layer):
        super().__init__()
        self.original_model = model
        self.top_layer = top_layer

        assert self.top_layer in dict(model.named_modules()), \
            f"the module named '{self.top_layer}' does not exist. \
            You can check the module names by calling model.named_modules()."

    def forward(self, x):
        # get the output of the top layer with register_forward_hook.
        # the hook is removed after the forward pass
        # to keep self.original_model original state.

        outputs = {}
        module = dict(self.original_model.named_modules())[self.top_layer]

        def hook(module, input, output):
            outputs[self.top_layer] = output

        handle = module.register_forward_hook(hook)
        
        _ = self.original_model(x)
        handle.remove()

        return outputs[self.top_layer]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test color equivariance of a model.")
    parser.add_argument(
        "--model_path", "-m", type=str, default="./model_manifest/cifar_h4s3_1999.json",
        help="Path to the model manifest file."
    )
    parser.add_argument(
        "--layer_name", "-l", type=str, default="layers.2.6.bn2",
        help="Name of the layer to extract the subnetwork from."
    )
    parser.add_argument(
        "--num_images", "-n", type=int, default=10,
        help="Number of images for testing."
    )
    args = parser.parse_args()

    # unpack arguments
    FILE_PATH = args.model_path
    LAYER_NAME = args.layer_name
    NUM_IMAGES = args.num_images

    # load model
    with open(FILE_PATH, "r") as f:
        params = json.load(f)
    model = lightning_model.LitHSGCNN.load_from_checkpoint(params["resume"], params=params)
    model = model.net
    model.to('cuda')
    model.eval()
    subnet = Subnetwork(model, LAYER_NAME)

    # load images
    trainloader, valloader, testloader = get_cifar()
    testloader = PilImageLoader(testloader)

    # equivariance test
    ## feed rotated images
    Y = []
    lifting_layer = create_lifting_layer(params)
    rotation = np.arange(params["n_groups_hue"]) * (360 / params["n_groups_hue"])
    hue_angle = lambda x: x*256/360
    for angle in tqdm(rotation, desc="Rotating hues"):
        for i, x_pil in enumerate(testloader):
            if i > NUM_IMAGES - 1:  # limit to first NUM_IMAGES images
                break
            x = lifting_layer(
                rotate_hue(x_pil, angle=hue_angle(angle), rgb_out=True)
            ).unsqueeze(0).cuda()
            y = subnet(x)
            B, C, H, W = y.shape
            y = y.reshape(
                B, params["n_groups_hue"], params["n_groups_saturation"], -1, H, W
            )
            Y.append(y)
    Y = torch.stack(Y, dim=0).squeeze() # (N_h * NUM_IMAGES, N_h, N_s, C, H, W)
    _, N_h, N_s, C, H, W = Y.shape
    Y = Y.reshape(N_h, NUM_IMAGES, N_h, N_s, C, H, W)
    Y = Y.permute(1, 0, 2, 3, 4, 5, 6)  # (NUM_IMAGES, N_h, N_h, N_s, C, H, W)

    ## calculate equivariance
    equivariance_error = []
    for i in range(NUM_IMAGES):
        for j in range(len(rotation)):
            f_g_x = Y[i, j, ...]
            g_f_x = Y[i, 0, ...].roll(j, dims=0)
            equivariance_error.append(
                torch.norm(f_g_x - g_f_x)
            )
    equivariance_error = torch.stack(equivariance_error, dim=0)
    equivariance_error = equivariance_error.reshape(NUM_IMAGES, len(rotation), -1)
