import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms

import lightning_model
from utils.color_utils import rotate_hue
from hsgroup.transforms import HueLuminanceSeparation, TensorReshape
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


# load model
FILE_PATH = "./model_manifest/cifar_h4s3_1999.json"
LAYER_NAME = "layers.2.6.bn2"
with open(FILE_PATH, "r") as f:
    params = json.load(f)
model = lightning_model.LitHSGCNN.load_from_checkpoint(params["resume"], params=params)
model = model.net
model.to('cuda')
model.eval()
subnet = Subnetwork(model, LAYER_NAME)

# load images
trainloader, valloader, testloader = get_cifar()
for data in testloader:
    pass
img, t = data
for x in img:
    x_pil = to_pil_from_normalized(x)
    #plt.figure()
    #plt.imshow(z)
    #plt.axis("off")
    #plt.show()

# equivariance test
## feed rotated images
lifting_layer = transforms.Compose([
    HueLuminanceSeparation(
       params["n_groups_hue"], params["n_groups_saturation"]
    ),
    transforms.Normalize(MEAN["cifar"], STD["cifar"]),
    TensorReshape(),
])
Y = []
rotation = np.arange(params["n_groups_hue"]) * (360 / params["n_groups_hue"])
hue_angle = lambda x: x*256/360
for angle in rotation:
    x = rotate_hue(x_pil, angle=hue_angle(angle), rgb_out=True)
    print(f"=========== {angle} ===========")
    plt.figure()
    plt.imshow(x)
    plt.axis("off")
    plt.show()
    
    x = lifting_layer(x).unsqueeze(0).cuda()
    y = subnet(x)
    B, C, H, W = y.shape
    y = y.reshape(
        1, params["n_groups_hue"], params["n_groups_saturation"], -1, H, W
    )
    print(y.shape)
    Y.append(y)
Y = torch.stack(Y, dim=0) # (N_d, 1, N_h, N_s, C, H, W)
Y = Y.squeeze() # (N_d, N_h, N_s, C, H, W)

## calculate equivariance
equivariance_error = []
for i in range(len(rotation)):
    f_g_x = Y[i]
    g_f_x = Y[0].roll(i, dims=0)
    equivariance_error.append(
        torch.norm(f_g_x - g_f_x)
    )
equivariance_error = torch.stack(
    equivariance_error, dim=0
)
