#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief  Main script to run the testing on the CIFAR-10 dataset.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   26 Oct 2022.
"""

import argparse
import numpy as np
import torch
import torchvision

# My imports
import vitcifar10

# Fix random seeds for reproducibility
seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '--data':    'Path to the CIFAR-10 data directory (required: True)',
        '--resume':  'Path to the checkpoint file (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch ViT for training/validation on CIFAR-10.')
    args.add_argument('--data', required=True, type=str,
                      help=help('--data'))
    args.add_argument('--resume', required=True, type=str,
                      help=help('--resume'))
    
    return  args.parse_args()


def load_dataset(test_preproc_tf, data_dir, num_workers: int = 8):
    # Load testing set
    test_ds = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, 
                                           transform=test_preproc_tf)
    # Create dataloader
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=1, 
                                          shuffle=False, 
                                          num_workers=num_workers)

    return test_dl


def main():
    # Parse command line parameters
    args = parse_cmdline_params()

    # Prepare preprocessing layers
    _, test_preproc_tf = vitcifar10.build_preprocessing_transforms()

    # Get dataloaders for training and testing
    test_dl = load_dataset(test_preproc_tf, data_dir=args.data)

    # Build model
    net = vitcifar10.build_model()

    # Load weights from file
    state = torch.load(args.resume) 
    net.load_state_dict(state['net'])

    # Use cross-entropy loss
    loss_func = torch.nn.CrossEntropyLoss()
    
    # Run testing
    test_loss = []
    test_acc_over_epochs = []
    test_loss, test_acc = vitcifar10.valid(net, test_dl, loss_func)


if __name__ == '__main__':
    main()
