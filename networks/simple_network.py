import sys
sys.path.append("/Users/felixomahony/Documents/git/invariant-classification")

import torch
import torch.nn as nn
import numpy as np

from hsgroup import hsgconv
import ceconv
from ceconv.pooling import GroupCosetMaxPool, GroupMaxPool2d
from ceconv.ceconv2d import CEConv2d


class SimpleNetwork(nn.Module):
    def __init__(self, num_classes, n_groups_hue, n_groups_saturation, ours=True):
        super(SimpleNetwork, self).__init__()

        if ours:
            conv_1 = hsgconv.GroupConvHS(3, 32, 3, padding=1, n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation)
            pool2d_1 = nn.MaxPool2d((2, 2))
            conv_2 = hsgconv.GroupConvHS(32, 64, 3, padding=1, n_groups_hue=n_groups_hue, n_groups_saturation=n_groups_saturation)
            pool2d_2 = nn.MaxPool2d((2, 2))
            group_pool = hsgconv.GroupPool(n_groups_hue * n_groups_saturation)
        else:
            assert n_groups_saturation==1
            conv_1 = ceconv.CEConv2d(1, n_groups_hue, 3, 32, 3, padding=1)
            pool2d_1 = GroupMaxPool2d((2,2))
            conv_2 = ceconv.CEConv2d(n_groups_hue, n_groups_hue, 32, 64, 3, padding=1)
            pool2d_2 = GroupMaxPool2d((2,2))
            group_pool = GroupCosetMaxPool()
        flatten = nn.Flatten()
        fc1 = nn.Linear(64*7*7, 128)
        fc2 = nn.Linear(128, num_classes)
        actv = nn.ReLU()
        finactv = nn.Softmax()
        self.network = nn.Sequential(conv_1, actv, pool2d_1, conv_2, actv, pool2d_2, group_pool, flatten, fc1, actv, fc2, finactv)

    def forward(self, x):
        return self.network(x)
    
class Z2CNN(nn.Module):
    """
    Derived from description given in https://arxiv.org/pdf/1602.07576.pdf
    """

    def __init__(self, num_classes, n_groups_hue, n_groups_saturation, ours=True):
        super(Z2CNN, self).__init__()

        if not ours:
            assert n_groups_saturation == 1
        self.n_classes = num_classes
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        self.ours = ours
        n_channels = int(20 / np.sqrt(n_groups_hue * n_groups_saturation))
        layers = [
            self.conv(3, n_channels, 3, first_layer=True),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.max_pool((2,2)),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, self.n_classes, 4),
            self.batch_norm(self.n_classes),
            self.group_pool(),
            nn.Flatten(),
            nn.Softmax(),
        ]
        self.network = nn.Sequential(*layers)
    
    def conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, first_layer=False):
        if self.ours:
            return hsgconv.GroupConvHS(
                in_channels, 
                out_channels, 
                kernel_size, 
                padding=padding,
                stride=stride, 
                n_groups_hue=self.n_groups_hue, 
                n_groups_saturation=self.n_groups_saturation
                )
        else:
            return CEConv2d(
                in_rotations = 1 if first_layer else self.n_groups_hue,
                out_rotations = self.n_groups_hue,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False
                )
    
    def batch_norm(self, channels):
        if self.ours:
            return hsgconv.GroupBatchNorm2d(
                channels, 
                momentum=0.1, 
                n_groups_hue=self.n_groups_hue, 
                n_groups_saturation=self.n_groups_saturation)
        else:
            return nn.BatchNorm3d(channels, momentum=0.1)

    def max_pool(self, filter_size):
        if self.ours:
            return nn.MaxPool2d(filter_size)
        else:
            return GroupMaxPool2d(filter_size)
        
    def group_pool(self):
        if self.ours:
            return hsgconv.GroupPool(self.n_groups_hue * self.n_groups_saturation)
        else:
            return GroupCosetMaxPool()

    
    def forward(self, x):
        return self.network(x)
    

class HybridZ2CNN(nn.Module):
    """
    Derived from description given in https://arxiv.org/pdf/1602.07576.pdf
    """

    def __init__(self, num_classes, n_groups_hue, n_groups_saturation, ours=True, group_pool=False):
        super(HybridZ2CNN, self).__init__()

        if not ours:
            assert n_groups_saturation == 1
        self.n_classes = num_classes
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        self.ours = ours
        n_channels = int(20 / np.sqrt(n_groups_hue * n_groups_saturation))
        layers = [
            self.conv(3, n_channels, 3, first_layer=True),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.conv(n_channels, n_channels, 3),
            self.batch_norm(n_channels),
            nn.ReLU(),
            self.group_pool() if group_pool else ExplodeGroupDim(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(
                n_channels if group_pool else n_channels * self.n_groups_hue * n_groups_saturation, 
                n_channels, 
                3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, 3),
            nn.BatchNorm2d(n_channels),
            nn.ReLU(),
            nn.Conv2d(n_channels, self.n_classes, 4),
            nn.BatchNorm2d(self.n_classes),
            nn.Flatten(),
            nn.Softmax(),
        ]
        self.network = nn.Sequential(*layers)
    
    def conv(self, in_channels, out_channels, kernel_size, stride=1, padding=0, first_layer=False):
        if self.ours:
            return hsgconv.GroupConvHS(
                in_channels, 
                out_channels, 
                kernel_size, 
                padding=padding,
                stride=stride, 
                n_groups_hue=self.n_groups_hue, 
                n_groups_saturation=self.n_groups_saturation
                )
        else:
            return ceconv.CEConv2d(
                in_rotations = 1 if first_layer else self.n_groups_hue,
                out_rotations = self.n_groups_hue,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                bias=False
                )
    
    def batch_norm(self, channels):
        if self.ours:
            return hsgconv.GroupBatchNorm2d(
                channels, 
                momentum=0.1, 
                n_groups_hue=self.n_groups_hue, 
                n_groups_saturation=self.n_groups_saturation)
        else:
            return nn.BatchNorm3d(channels, momentum=0.1)

    def max_pool(self, filter_size):
        if self.ours:
            return nn.MaxPool2d(filter_size)
        else:
            return GroupMaxPool2d(filter_size)
        
    def group_pool(self):
        if self.ours:
            return hsgconv.GroupPool(self.n_groups_hue * self.n_groups_saturation)
        else:
            return GroupCosetMaxPool()

    
    def forward(self, x):
        return self.network(x)

class ExplodeGroupDim(nn.Module):
    def __init__(self):
        super(ExplodeGroupDim, self).__init__()

    def forward(self, x):
        s = x.shape
        return x.view(s[0], -1, s[-2], s[-1])

if __name__ == "__main__":
    h = 4
    s = 3
    c = 10

    # create networks
    simple_network_ours = SimpleNetwork(c, h, s, ours=True)
    simple_network_ceconv = SimpleNetwork(c, h, 1, ours=False)   # nb ceconv only has hue

    z2cnn_ours = Z2CNN(c, h, s, ours=True)
    z2cnn_ceconv = Z2CNN(c, h, 1, ours=False)

    hybrid_ours = HybridZ2CNN(c, h, s, ours=True)
    hybrid_ceconv = HybridZ2CNN(c, h, 1, ours=False)

    # create inputs
    input_ours = torch.rand((64, 3 * h * s, 28, 28))
    input_ceconv = torch.rand((64, 3, 28, 28))

    # generate outputs
    output_simple_ours = simple_network_ours(input_ours)
    output_z2cnn_ours = z2cnn_ours(input_ours)
    output_hybrid_ours = hybrid_ours(input_ours)

    output_simple_ceconv = simple_network_ceconv(input_ceconv)
    output_z2cnn_ceconv = z2cnn_ceconv(input_ceconv)
    output_hybrid_ceconv = hybrid_ceconv(input_ceconv)

    print(output_simple_ours.shape)
    print(output_z2cnn_ours.shape)
    print(output_hybrid_ours.shape)
    print(output_simple_ceconv.shape)
    print(output_z2cnn_ceconv.shape)
    print(output_hybrid_ceconv.shape)
