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


# ------------------------
import torch
import torch.nn as nn
from torchinfo import summary
import timm

import sys
sys.path.append('/home/chanseok-lim/Projects/lim_equivariance/lie-deriv')
from lee.layerwise_lee import selective_apply
from lee.transforms import img_like


def print_module_tree(module: nn.Module, prefix: str = '', last: bool = True):
    """ Recursively prints the module's children in a tree-like format."""
    connector = '└── ' if last else '├── '
    print(f'{prefix}{connector}{module._get_name()}')
    prefix += '    ' if last else '│   '

    children = list(module.named_children())
    for i, (name, child) in enumerate(children):
        last_child = (i == len(children) - 1)
        print(f'{prefix}{"└── " if last_child else "├── "}{name}: {child._get_name()}')
        print_module_tree(child, prefix, last_child)
        

def model_information(model, inp=None):
    # print the model info
    print("\nModel Summary:")
    if inp is not None:
        model_summary = summary(model,input_data=inp)
    else:
        model_summary = summary(model,input_size=(1, 3, 224, 224))
    print(model_summary)
    
    # print the module tree
    print("\nModule Tree:")
    print_module_tree(model)


def collect_module_outputs(model, input_data):
    output_dict = {}

    def hook(module, input, output):
        module_name = module_name_mapping[id(module)]
        output_dict[module_name] = {
            'shape': output.shape,
            'image-like': img_like(output.shape),
        }

    # register hooks
    module_name_mapping = {id(module): name for name, module in model.named_modules()}
    register_hook = lambda m: m.register_forward_hook(hook)
    handles = selective_apply(model, register_hook)

    # forward pass to call resistered hooks
    model(input_data)

    # remove hooks
    for h in handles:
        h.remove()

    return output_dict


def find_latest_imagelike_layer(output_dict):
    former_module_name = None
    former_output = {}
    
    for module_name, output in output_dict.items():
        image_like = output['image-like']
        former_image_like = former_output.get('image-like', False)
    
        if not image_like and former_image_like:
            print('\nThe latest image-like module was found:')
            print(f'former: {former_module_name}, shape: {former_output["shape"]}')
            print(f'current: {module_name}, shape: {output["shape"]}')
            break
    
        former_module_name = module_name
        former_output = output


def show_latest_imagelike_layer(args):
    if 'model_name' in args:
        MODEL_NAME = args['model_name']
        model = timm.create_model(MODEL_NAME, pretrained=True)
    elif 'model' in args:
        model = args['model']
    else:
        raise KeyError(f'args must have a key either \'model_name\' or \'model\'')
    model.to('cuda')
    model.eval()

    if 'input_data' in args:
        input_data = args['input_data']
    else:
        input_data = torch.randn((1, 3, 224, 224)).to('cuda')
    output_dict = collect_module_outputs(model, input_data)
    find_latest_imagelike_layer(output_dict)

    model_information(model, input_data)
# ------------------------


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
for i, data in enumerate(testloader):
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
# ------------------------
show_model_info = True
if show_model_info:
    input_data = Image.fromarray(np.random.randn(32,32,3).astype(np.uint8))
    show_latest_imagelike_layer(
        {'model': model,
        'input_data': lifting_layer(input_data).unsqueeze(0).cuda()
        }
    )
# ------------------------
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
