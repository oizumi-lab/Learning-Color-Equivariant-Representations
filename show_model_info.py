import argparse
import json

import numpy as np
from PIL import Image
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


def load_lightning_model(file_path):
    from lightning_model import LitHSGCNN
    import torchvision.transforms as transforms
    from datasets.dataloaders import MEAN, STD, get_cifar
    from hsgroup.transforms import HueLuminanceSeparation, TensorReshape
    # load model
    with open(file_path, "r") as f:
        params = json.load(f)
    model = LitHSGCNN.load_from_checkpoint(params["resume"], params=params)
    model = model.net
    model.to('cuda')
    model.eval()
    # load preprocessing layer for models
    lifting_layer = transforms.Compose([
        HueLuminanceSeparation(
           params["n_groups_hue"], params["n_groups_saturation"]
        ),
        transforms.Normalize(MEAN["cifar"], STD["cifar"]),
        TensorReshape(),
    ])
    return model, lifting_layer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Show model information and latest image-like layer.'
    )
    parser.add_argument(
        'model_path', type=str, 
        help='Path to the model manifest file.'
    )
    args = parser.parse_args()
    model, lifting_layer = load_lightning_model(args.model_path)
    input_data = Image.fromarray(np.random.randn(32,32,3).astype(np.uint8))
    show_latest_imagelike_layer(
        {'model': model, 
         'input_data': lifting_layer(input_data).unsqueeze(0).cuda()
        }
    )
    