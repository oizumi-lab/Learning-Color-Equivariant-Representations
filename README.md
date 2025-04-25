# Learning-Color-Equivariant-Representations
Official repository for Learning Color Equivariant Representations ([arXiv](https://arxiv.org/abs/2406.09588)).

## Abstract

In this paper, we introduce group convolutional neural networks (GCNNs) equivariant to color variation. GCNNs have been designed for a variety of geometric transformations from 2D and 3D rotation groups, to semi-groups such as scale. Despite the improved interpretability, accuracy and generalizability of these architectures, GCNNs have seen limited application in the context of perceptual quantities. Notably, the recent CEConv network uses a GCNN to achieve equivariance to hue transformations by convolving input images with a hue rotated RGB filter. However, this approach leads to invalid RGB values which break equivariance and degrade performance. We resolve these issues with a lifting layer that transforms the input image directly, thereby circumventing the issue of invalid RGB values and improving equivariance error by over three orders of magnitude. Moreover, we extend the notion of color equivariance to include equivariance to saturation and luminance shift. Our hue-, saturation-, luminance- and color-equivariant networks achieve strong generalization to out-of-distribution perceptual variations and improved sample efficiency over conventional architectures. We demonstrate the utility of our approach on synthetic and real world datasets where we consistently outperform competitive baselines.

## Experiments

In order to run experiments, ensure the correct dataloader is present in `/datasets/dataloaders.py` (we have included dataloaders for the experiments presented in the paper). Experiments can be run on single GPU (`train_sGPU.py`) or multiple GPUs (`train_mGPU.py`) using

```
python train_sGPU.py --exp_class=dataset_name --exp_name=hyperparameter_name
```

where `dataset_name` should match `/experiments/dataset_name.json` and `hyperparameter_name` should match the specific set of hyperparameters defined in the .json file. For example, running Oxford Pets with 3 hue group with seed 1999 would entail

```
python train_sGPU.py --exp_class=oxford_pets.json --exp_name=h3s1_1999
```
