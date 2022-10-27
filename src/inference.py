#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief  Test a model trained on CIFAR-10 on a single image.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   27 Oct 2022.
"""

import argparse
import numpy as np
import torch
import torchvision
import PIL

# My imports
import vitcifar10


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '--image': 'Path to the image (required: True)',
        '--model': 'Path to the .pt model file (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch ViT for training/validation on CIFAR-10.')
    args.add_argument('--image', required=True, type=str, help=help('--image'))
    args.add_argument('--model', required=True, type=str, help=help('--model'))
    
    return  args.parse_args()


def main():
    # Parse command line parameters
    args = parse_cmdline_params()

    # Prepare preprocessing layers
    _, test_preproc_tf = vitcifar10.build_preprocessing_transforms(size=(384,384))

    # Load the image
    im_rgb = PIL.Image.open(args.image)
    if len(im_rgb.getbands()) != 3:
        raise ValueError('[ERROR] The input image must be a colour image.')

    # Build model
    net = vitcifar10.build_model()

    # Load weights from file
    state = torch.load(args.model) 
    net.load_state_dict(state['net'])

    # Preprocess image
    im_tensor = test_preproc_tf(im_rgb).to('cuda')

    # Run inference 
    output = torch.argmax(net(torch.unsqueeze(im_tensor, 0))).item()
    
    # Print result
    cifar10_classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')
    print("It is a {}!".format(cifar10_classes[output]))


if __name__ == '__main__':
    main()
