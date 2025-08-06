import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import math
import sys
import torchvision
import torchvision.transforms as transforms

from hsgroup.transforms import HueSeparation, TensorReshape, HueLuminanceSeparation, RandomScaling
from datasets.custom_datasets import DSpritesDataset, HDF5Dataset, SmallNorbDataset, CustomMNIST, HSVMNIST

from collections import namedtuple
import numpy as np
import h5py
import parquet
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
import pyarrow.parquet as pq
import io
from PIL import Image, ImageStat

dataloaders = namedtuple("dataloaders", ["train", "val", "test"])

DATA_PATH_DSPRITES = "./data/dsprites.npz"
DATA_PATH_SHAPES3D = "./data/3dshapes.h5"
DATA_PATH_SMALLNORB = "./data/smallnorb/"
DATA_NAME_SMALLNORB_TEST = "test-00000-of-00001-b4af1727fb5b132e.parquet"
DATA_NAME_SMALLNORB_TRAIN = "train-00000-of-00001-ba54590c34eb8af1.parquet"
MEAN = {
    "camelyon17": [0.5, 0.5, 0.5],
    "shapes3d": [0.5, 0.5, 0.5],
    "cifar": [0.4914, 0.4822, 0.4465],
    "smallnorb": [0.5, 0.5, 0.5],
    "imagenet": [0.485, 0.456, 0.406],
    "caltech101": [0.4914, 0.4822, 0.4465],
    "flowers": [0.4914, 0.4822, 0.4465],
    "pets": [0.4914, 0.4822, 0.4465],
    "cars": [0.4914, 0.4822, 0.4465],
    "STL10": [0.4914, 0.4822, 0.4465]
}
STD = {
    "camelyon17": [0.5, 0.5, 0.5],
    "shapes3d": [0.5, 0.5, 0.5],
    "cifar": [0.2023, 0.1994, 0.2010],
    "smallnorb": [0.5, 0.5, 0.5],
    "imagenet": [0.229, 0.224, 0.225],
    "caltech101": [0.2023, 0.1994, 0.2010],
    "flowers": [0.2023, 0.1994, 0.2010],
    "pets": [0.2023, 0.1994, 0.2010],
    "cars": [0.2023, 0.1994, 0.2010],
    "STL10": [0.2023, 0.1994, 0.2010]
}


def get_camelyon17(n_groups_hue = 1, n_groups_luminance = 1, batch_size=64, ours=True):
    transform_train = transforms.Compose(
        [
            # transforms.Resize(224),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )
    transform_test = transforms.Compose(
        [
            # transforms.Resize(224),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    dataset = get_dataset("camelyon17", download=False)
    train_set = dataset.get_subset("train", transform=transform_train)
    val_set = dataset.get_subset("val", transform=transform_test)
    eval_set = dataset.get_subset("test", transform=transform_test)
    train_loader = get_train_loader("standard", train_set, batch_size=batch_size, shuffle=True)
    val_loader = get_eval_loader("standard", val_set, batch_size=batch_size)
    eval_loader = get_eval_loader("standard", eval_set,batch_size=batch_size)
    return dataloaders(train=train_loader, val=val_loader, test=eval_loader)


def get_shapes3d(n_groups_hue = 1, n_groups_saturation=1, batch_size=128, ours=True):
    train_test_split = 0.8
    # Define data transformations
    if ours:
        transform_train = transforms.Compose(
            [
                # transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                HueSeparation(n_groups_hue),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                TensorReshape(),
            ]
        )

        transform_test = transforms.Compose(
            [
                # transforms.Resize(224),
                HueSeparation(n_groups_hue),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                TensorReshape(),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    # Load 3D Objects data
    file_train = h5py.File(DATA_PATH_SHAPES3D, "r")
    file_test = h5py.File(DATA_PATH_SHAPES3D, "r")
    labels = file_train["labels"][()]
    # hue_floor, hue_wall, hue_obj, scale, shape, orientation, doesn't matter
    labels = labels.reshape((10, 10, 10, 8, 4, 15, 6))
    labels_c1 = labels[:5, :5, :5].reshape(5 * 5 * 5 * 8 * 4 * 15, 6)
    labels_c2 = labels[5:, 5:, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 6)
    labels_c3 = labels[:5, :5, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 6) 
    images = file_train["images"][()]
    images = images.reshape((10, 10, 10, 8, 4, 15, 64, 64, 3))
    images_c1 = images[:5, :5, :5].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)
    images_c2 = images[5:, 5:, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)
    images_c3 = images[:5, :5, 5:].reshape(5 * 5 * 5 * 8 * 4 * 15, 64, 64, 3)
    # trim dataset to 60000
    random_choice = np.random.choice(len(images_c1), len(images_c1), replace=False)     # for train test split randomization
    # create train and test sets
    array_train = np.sort(random_choice[: int(len(images_c1) * train_test_split)])
    imgs_train_c1 = images_c1[array_train]
    labels_train_c1 = labels_c1[array_train]
    # test
    array_test = np.sort(random_choice[int(len(images_c1) * train_test_split) :])
    imgs_test_c1 = images_c1[array_test]
    labels_test_c1 = labels_c1[array_test]
    # test c2
    imgs_test_c2 = images_c2[array_test]
    labels_test_c2 = labels_c2[array_test]
    # test c3
    imgs_test_c3 = images_c3[array_test]
    labels_test_c3 = labels_c3[array_test]

    trainset = HDF5Dataset(imgs_train_c1, labels_train_c1, transform=transform_train)
    testset = HDF5Dataset(imgs_test_c1, labels_test_c1, transform=transform_test)
    testset_2 = HDF5Dataset(imgs_test_c2, labels_test_c2, transform=transform_test)
    testset_3 = HDF5Dataset(imgs_test_c3, labels_test_c3, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)
    testloader_2 = torch.utils.data.DataLoader(testset_2, batch_size=batch_size, shuffle=False, num_workers=24)
    testloader_3 = torch.utils.data.DataLoader(testset_3, batch_size=batch_size, shuffle=False, num_workers=24)
    return dataloaders(train=trainloader, val=testloader, test=[testloader, testloader_2, testloader_3])


def get_cifar(n_groups_hue=1, n_groups_luminance=1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.25),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape(),
        ])

        transform_test = transforms.Compose([
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_cifar100(n_groups_hue=1, n_groups_luminance=1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.25),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape(),
        ])

        transform_test = transforms.Compose([
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])


    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)
    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_tinyimagenet(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            TensorReshape(),
        ])

        transform_test = transforms.Compose([
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            TensorReshape(),    
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    
    trainset = torchvision.datasets.ImageFolder("/n/fs/huesatcnn/invariant-classification/data/tiny-imagenet-200/train", transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)

    testset = torchvision.datasets.ImageFolder("/n/fs/huesatcnn/invariant-classification/data/tiny-imagenet-200/val", transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)

    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_smallnorb(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True, frac_space=1.0):
    train_split = True
    data_train = []
    labels_train = []
    table = pq.read_table(DATA_PATH_SMALLNORB + DATA_NAME_SMALLNORB_TRAIN)
    for i in range(len(table)):
        image = Image.open(io.BytesIO(table['image_lt'][i][0].as_py()))
        label = table['category'][i].as_py()
        lighting = table['lighting'][i].as_py()
        if not train_split or (lighting > 1 and lighting < 4):
            data_train.append(image)
            labels_train.append(label)
            
    data_test = []
    labels_test = []
    data_test_lowlight = []
    labels_test_lowlight = []
    data_test_mid = []
    labels_test_mid = []
    data_test_highlight = []
    labels_test_highlight = []

    table = pq.read_table(DATA_PATH_SMALLNORB + DATA_NAME_SMALLNORB_TEST)
    for i in range(len(table)):
        image = Image.open(io.BytesIO(table['image_lt'][i][0].as_py()))
        label = table['category'][i].as_py()
        lighting = table['lighting'][i].as_py()

        data_test.append(image)
        labels_test.append(label)

        if lighting > 3:
            data_test_highlight.append(image)
            labels_test_highlight.append(label)
        elif lighting < 2:
            data_test_lowlight.append(image)
            labels_test_lowlight.append(label)
        else:
            data_test_mid.append(image)
            labels_test_mid.append(label)

    
    transform_train = transforms.Compose(
        [
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance, frac_space),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )
    transform_test = transforms.Compose(
        [
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance, frac_space),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            TensorReshape(),
        ]
    )

    dataset_train = SmallNorbDataset(data_train, labels_train, transform=transform_train)
    dataset_test = SmallNorbDataset(data_test, labels_test, transform=transform_test)
    dataset_test_lowlight = SmallNorbDataset(data_test_lowlight, labels_test_lowlight, transform=transform_test)
    dataset_test_highlight = SmallNorbDataset(data_test_highlight, labels_test_highlight, transform=transform_test)
    dataset_test_mid = SmallNorbDataset(data_test_mid, labels_test_mid, transform=transform_test)

    trainloader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=24)
    testloader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=24)
    testloader_lowlight = torch.utils.data.DataLoader(dataset_test_lowlight, batch_size=batch_size, shuffle=False, num_workers=24)
    testloader_highlight = torch.utils.data.DataLoader(dataset_test_highlight, batch_size=batch_size, shuffle=False, num_workers=24)
    testloader_mid = torch.utils.data.DataLoader(dataset_test_mid, batch_size=batch_size, shuffle=False, num_workers=24)

    return dataloaders(train=trainloader, val=testloader, test=[testloader, testloader_lowlight, testloader_mid, testloader_highlight])


def get_tinyimagenet(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            TensorReshape(),
        ])

        transform_test = transforms.Compose([
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            TensorReshape(),    
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor()
        ])
    
    trainset = torchvision.datasets.ImageFolder("/n/fs/huesatcnn/invariant-classification/data/tiny-imagenet-200/train", transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)

    testset = torchvision.datasets.ImageFolder("/n/fs/huesatcnn/invariant-classification/data/tiny-imagenet-200/val", transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)

    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_tinyimagenet224(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            TensorReshape(),
        ])

        transform_test = transforms.Compose([
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            TensorReshape(),    
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.ToTensor()
        ])
    
    trainset = torchvision.datasets.ImageFolder("/n/fs/huesatcnn/invariant-classification/data/tiny-224/train", transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)

    testset = torchvision.datasets.ImageFolder("/n/fs/huesatcnn/invariant-classification/data/tiny-224/val", transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)

    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_caltech101(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.25),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            HueSeparation(1),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])

    set = torchvision.datasets.Caltech101(root='./data', download=False, transform=transform_train)
    trainset, testset = torch.utils.data.random_split(set, [math.floor(0.67 * len(set)), math.ceil(0.33 * len(set))])
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)
    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_flowers(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    trainset = torchvision.datasets.Flowers102(root='./data', split="train", transform=transform_train, download=False)
    valset = torchvision.datasets.Flowers102(root='./data', split="val", transform=transform_train, download=False)
    trainset = torch.utils.data.ConcatDataset([trainset, valset])
    testset = torchvision.datasets.Flowers102(root='./data', split="test", transform=transform_test, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)
    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_pets(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.25),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    trainset = torchvision.datasets.OxfordIIITPet(root='./data', split="trainval", transform=transform_train, download=False)
    test_set = torchvision.datasets.OxfordIIITPet(root='./data', split="test", transform=transform_test, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
    testloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=24)
    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_cars(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.25),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    trainset = torchvision.datasets.StanfordCars(root='./data', split="train", transform=transform_train, download=False)
    testset = torchvision.datasets.StanfordCars(root='./data', split="test", transform=transform_test, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)
    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_STL10(n_groups_hue = 1, n_groups_luminance = 1, batch_size=128, ours=True):
    if ours:
        transform_train = transforms.Compose([
            transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0.25),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            TensorReshape()
        ])
    else:
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

    trainset = torchvision.datasets.STL10(root='./data', split="train", transform=transform_train, download=False)
    testset = torchvision.datasets.STL10(root='./data', split="test", transform=transform_test, download=False)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=24)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=24)
    return dataloaders(train=trainloader, val=testloader, test=testloader)


def get_camelyon17(n_groups_hue = 1, n_groups_luminance = 1, batch_size=64, ours=True, frac_space=1.0):
    if ours:
        transform_train = transforms.Compose(
            [
                HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance, frac_space),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                TensorReshape(),
            ]
        )
        transform_test = transforms.Compose(
            [
                HueSeparation(n_groups_hue) if n_groups_luminance == 1 else HueLuminanceSeparation(n_groups_hue, n_groups_luminance, frac_space),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                TensorReshape(),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    dataset = get_dataset("camelyon17", download=False)
    train_set = dataset.get_subset(
            "train",
            transform=transform_train,
        )
    val_set = dataset.get_subset(
            "val",
            transform=transform_test
        )
    eval_set = dataset.get_subset(
            "test",
            transform=transform_test
        )
    train_loader = get_train_loader(
        "standard",
        train_set,
        batch_size=batch_size)
    val_loader = get_eval_loader(
        "standard",
        val_set,
        batch_size=batch_size)
    eval_loader = get_eval_loader(
        "standard",
        eval_set,
        batch_size=batch_size)

    return dataloaders(train=train_loader, val=val_loader, test=eval_loader)


def parse_dataloader(name):
    if name == "shapes3d":
        return get_shapes3d
    elif name == 'tinyimagenet':
        return get_tinyimagenet
    elif name == 'tinyimagenet224':
        return get_tinyimagenet224
    elif name == "cifar":
        return get_cifar
    elif name == "cifar100":
        return get_cifar100
    elif name == "caltech101":
        return get_caltech101
    elif name == "flowers":
        return get_flowers
    elif name == "camelyon17":
        return get_camelyon17
    elif name == "oxford_pets":
        return get_pets
    elif name == "stanford_cars":
        return get_cars
    elif name == "STL10":
        return get_STL10
    else:
        raise ValueError("Invalid dataset name")
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    get_shapes3d(batch_size=4)