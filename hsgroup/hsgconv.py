"""
© Felix O'Mahony
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class GroupConvHS(nn.Module):
    """
    Group Convolution Layer with hue and luminance group equivariance
    -----------------------
    This is a layer to be used within the global hue and luminance equivariance network. It is relatively simple, since no geometric transformation of the input tensor must take place. Rather, the input tensor has its group channels permuted so that each possible permutation of the color space (in hue, which we think of as rotation) and luminance (which we think of as scaling/permuting around radii of groups) occurs.

    This is described fully in the published paper.
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=0,
            n_groups_hue=1,
            n_groups_saturation=1,
            bias = False,
            rescale_luminance = True,
        ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        if type(self.kernel_size) != int:
            self.kernel_size = self.kernel_size[0]  # we only permit square kernels
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation
        self.bias = bias
        self.rescale_luminance = rescale_luminance
        self.conv_weight = nn.Parameter(torch.Tensor(
            self.out_channels,
            self.n_groups_hue * self.n_groups_saturation * self.in_channels,
            self.kernel_size,
            self.kernel_size
        ))
        # Initialize the weights
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        # NOTICE:
        # self.group_conv_weight does not work because it does not follow
        # the update of conv_weight. However, we can not remove it, and also
        # mask and sfs, to use previous model weights including them in its
        # state_dict.
        self.construct_masks()
        self.register_buffer("group_conv_weight", self.construct_group_conv_weight())


    def construct_masks(self) -> None:
        N_hue = self.n_groups_hue
        N_saturation = self.n_groups_saturation
        mask = torch.zeros(
            N_hue, N_saturation, N_hue, N_saturation, 2, dtype=torch.int8
        )
        sfs = torch.ones(
            N_hue, N_saturation, dtype=torch.float32
        )
        hue_grid, saturation_grid = torch.meshgrid(
            torch.arange(N_hue),
            torch.arange(N_saturation)
        )
        group_grid = torch.stack(
            (hue_grid, saturation_grid), 
            dim=-1
        )
        for i in range(N_hue):
            for j in range(N_saturation):
                shift = i - N_hue // 2
                mask[i, j, :, :] = group_grid.roll(i, dims=0).roll(shift, dims=1)
                if shift > 0:
                    mask[i, j, :, :shift, :] = -1
                elif shift < 0:
                    mask[i, j, :, shift:, :] = -1
                # scale factors for invalid saturation elements
                sfs[i, j] = (N_saturation - abs(shift)) / N_saturation
        self.mask = mask
        self.sfs = sfs


    def construct_group_conv_weight(self):
        N_hue = self.n_groups_hue
        N_saturation = self.n_groups_saturation
        C_in = self.in_channels
        C_out = self.out_channels
        H_kernel, W_kernel = self.kernel_size, self.kernel_size
        conv_weight = torch.zeros(
            N_hue, 
            N_saturation, 
            C_out, 
            N_hue, 
            N_saturation, 
            C_in, 
            H_kernel, 
            W_kernel,
            dtype=self.conv_weight.dtype
        ).to(self.conv_weight.device)
        cw = self.conv_weight.view(
            C_out, 
            N_hue, 
            N_saturation, 
            C_in, 
            H_kernel, 
            W_kernel
        )
        # assign the values according to the mask so that 
        # conv_weight[i, j, :, k, l, :, :, :] corresponds to 
        # cw[:, self.mask[i,j,k,l,0], self.mask[i,j,k,l,1], :, :, :]
        # except where the mask is -1, in which case we set the value to 0.
        for i in range(N_hue):
            for j in range(N_saturation):
                for k in range(N_hue):
                    for l in range(N_saturation):
                        if self.mask[i,j,k,l,0] != -1:
                            conv_weight[i, j, :, k, l, :, :, :] = cw[:, self.mask[i,j,k,l,0], self.mask[i,j,k,l,1], :, :, :]
                            conv_weight[i, j, :, k, l, :, :, :] /= self.sfs[i, j]
        return conv_weight
    

    def forward_2(self, x) -> torch.Tensor:
        """
        incoming tensor shape:
            (batch_size, n_groups_hue * n_groups_saturation * in_channels, height, width)
        outgoing tensor shape:
            (batch_size, n_groups_hue * n_groups_saturation * out_channels, height, width)
        """
        N_hue = self.n_groups_hue
        N_saturation = self.n_groups_saturation
        C_in = self.in_channels
        C_out = self.out_channels
        H_kernel, W_kernel = self.kernel_size, self.kernel_size
        conv_weight = torch.zeros(
            N_hue, 
            N_saturation, 
            C_out, 
            N_hue, 
            N_saturation, 
            C_in, 
            H_kernel, 
            W_kernel,
            dtype=self.conv_weight.dtype
        ).to(x.device)
        cw = self.conv_weight.view(
            C_out, 
            N_hue, 
            N_saturation, 
            C_in, 
            H_kernel, 
            W_kernel
        )
        for i in range(N_hue):
            for j in range(N_saturation):
                shift = j - N_saturation // 2
                conv_weight[i, j, :, :, :, :, :, :] = cw.roll(i, dims=1).roll(shift, dims=2)
                if shift > 0:
                    conv_weight[i, j, :, :, :shift, :, :, :] *= 0
                elif shift < 0:
                    conv_weight[i, j, :, :, shift:, :, :, :] *= 0
                # rescale owing to saturation zero padding
                scale_factor = (N_saturation - abs(shift)) / N_saturation
                conv_weight[i, j, :, :, :, :, :, :] /= scale_factor
                # the mask has four dimensions and is boolean
                    # we want to assign the values of conv_weight[a, b, :, c, d, :, :, :] to cw[:, i, j, :,:,:]
                    # where mask[a, b, c, d] is True
                # we can do this by reshaping conv_weight to have shape (n_groups * n_groups_saturation, out_channels, n_groups * n_groups_luminance, in_channels, kernel_size, kernel_size)
                # and then reshaping cw to have shape (out_channels, n_groups * n_groups_saturation, in_channels, kernel_size, kernel_size)
        conv_weight = conv_weight.view(
            N_hue * N_saturation * C_out, 
            N_hue * N_saturation * C_in,
            H_kernel, 
            W_kernel
        )
        out_tensors = F.conv2d(x, conv_weight, stride=self.stride, padding=self.padding)
        return out_tensors

    
    def forward(self, x):
        out_tensors = self.forward_2(x)
        return out_tensors


class GroupPool(nn.Module):
    def __init__(
        self, 
        n_groups_total, 
        pool_operation=lambda x: torch.max(x, dim=1)[0], 
        verbose=False, 
        name=None
    ) -> None:
        super().__init__()
        self.verbose = verbose
        self.n_groups = n_groups_total
        if verbose:
            print("verbose is sent to True in Pooling Layer")
        self.name = name if name is not None else "GroupPool"
        self.pool_operation = pool_operation

    def forward(self, x):
        N, _, H, W = x.shape
        x = x.view(N, self.n_groups, -1, H, W)
        # incoming tensor is of shape (batch_size, n_groups * channels, height, width)
        # outgoing tensor should be of shape (batch_size, channels, height, width)
        y = self.pool_operation(x)
        return y


class GroupBatchNorm2d(nn.Module):
    def __init__(
        self, 
        num_features, 
        n_groups_hue=4, 
        n_groups_saturation=1, 
        momentum=0.1
    ):
        super().__init__()
        self.batch_norm = nn.BatchNorm3d(num_features, momentum=momentum)
        self.num_features = num_features
        self.n_groups_hue = n_groups_hue
        self.n_groups_saturation = n_groups_saturation

    def forward(self, x):
        """
        incoming tensor is of shape (batch_size, n_groups * channels, height, width)"""
        if x.shape[1] != self.n_groups_hue * self.n_groups_saturation * self.num_features:
            raise ValueError(
                f"Expected {self.n_groups_hue * self.n_groups_saturation * self.num_features} channels in tensor, but got {x.shape[1]} channels"
            )
        N, _, H, W = x.shape
        x = x.view(N, -1, self.num_features, H, W)
        x = x.permute(0, 2, 1, 3, 4) # (N, num_features, n_groups, H, W)
        y = self.batch_norm(x)
        y = y.permute(0, 2, 1, 3, 4)
        y = y.reshape(N, -1, H, W)
        return y


if __name__=="__main__":
    # NB input tensor has shape (batch, groups_hue * groups_saturation * channels, w, h)
    test_input = torch.randn(1, 4 * 3 * 3, 32, 32)
    group_conv = GroupConvHS(3, 3, 3, n_groups_hue=4, n_groups_saturation=3)
    group_conv(test_input)