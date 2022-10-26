#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief  Module containing functions that are useful for training and testing ViT on CIFAR-10.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   25 Oct 2022.
"""

import torch
import torch.utils.tensorboard
import torchvision
import timm

# My imports
import vitcifar10


def build_model(nclasses: int = 10):
    """
    @brief Create Vision Transformer (ViT) model pre-trained on ImageNet-21k 
           (14 million images, 21,843 classes) at resolution 224x224 
           fine-tuned on ImageNet 2012 (1 million images, 1,000 classes) 
           at resolution 384x384.

    @param[in]  nclasses  Number of classes, CIFAR-10 has obviously 10 clsses.
    """
    net = timm.create_model('vit_base_patch16_384', pretrained=True)
    net.head = torch.nn.Linear(net.head.in_features, nclasses)
    net.cuda()

    return net


def build_preprocessing_transforms(size: int = 384, randaug_n: int = 2, 
                                   randaug_m: int = 14):
    """
    @brief Preprocessing and data augmentation.

    @param[in]  size  Target size of the images to be resized prior 
                      processing by the network.

    @returns a tuple of two transforms, one for training and another one for testing.
    """
    # Preprocessing for training
    train_preproc_tf = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.Resize(size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Data augmentation
    train_preproc_tf.transforms.insert(0, vitcifar10.RandAugment(randaug_n, randaug_m))
    
    # Preprocessing for testing
    valid_preproc_tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return train_preproc_tf, valid_preproc_tf


def setup_tensorboard(log_dir) -> torch.utils.tensorboard.SummaryWriter:
    """
    @param[in]  log_dir  Path to the Tensorboard log directory.
    @returns the Tensorboard writer.
    """
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=log_dir)
    return writer


if __name__ == '__main__':
    raise RuntimeError('[ERROR] The module vitcifar10 is not supposed to be run as an executable.')
