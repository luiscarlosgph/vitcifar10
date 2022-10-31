#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief  Script to download the CIFAR-10. 
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   31 Oct 2022.
"""

import argparse
import numpy as np
import torch
import torch.utils.data.sampler
import torch.utils.tensorboard
import torchvision
import timm
import tqdm
import os
import time
import random

# My imports
import vitcifar10


def help(option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '--data':    'Path to the CIFAR-10 data directory (required: True)',
    }
    return help_msg[option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch ViT for training/validation on CIFAR-10.')
    args.add_argument('--data', required=True, type=str,
                      help=help('--data'))
    
    return  args.parse_args()


def main():
    # Parse command line parameters
    args = parse_cmdline_params()

    # Prepare preprocessing layers
    train_preproc_tf, test_preproc_tf = vitcifar10.build_preprocessing_transforms()

    print('[INFO] CIFAR-10 download started.')

    # Download training and validation sets (if they are not already downloaded)
    torchvision.datasets.CIFAR10(root=args.data, train=True,
                                 download=True, 
                                 transform=train_preproc_tf)
    torchvision.datasets.CIFAR10(root=args.data, train=False,
                                 download=True,
                                 transform=test_preproc_tf)

    print('[INFO] CIFAR-10 download finished.')
    

if __name__ == '__main__':
    main()
