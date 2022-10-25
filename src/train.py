"""
@brief  Main script to kick off the training.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   25 Oct 2022.
"""

import argparse
import numpy as np
import torch
import torchvision
import timm
import vitcifar10.randomaug


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
        '--lr' : 'Learning rate (required: True)',  # ResNets: 1e-3, ViT: 1e-4
        '--opt': 'Optimizer (required: True)', 
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch ViT for training/testing on CIFAR-10.')
    args.add_argument('--lr', required=True, type=float, help=help('--lr'))  
    args.add_argument('--opt', required=True, type=str, help=help('--opt'))
    #args.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    #args.add_argument('--noaug', action='store_true', help='disable use randomaug')
    #args.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
    #args.add_argument('--nowandb', action='store_true', help='disable wandb')
    #args.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    #args.add_argument('--net', default='vit')
    #args.add_argument('--bs', default='512')
    #args.add_argument('--size', default="32")
    #args.add_argument('--n_epochs', type=int, default='200')
    #args.add_argument('--patch', default='4', type=int, help="patch for ViT")
    #args.add_argument('--dimhead', default="512", type=int)
    #args.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")  

    return  args.parse_args()


def load_dataset(train_preproc_tf, test_preproc_tf, train_bs: int = 512, 
        test_bs: int = 100, num_workers: int = 8):
    """
    @brief Function that creates the dataloaders of CIFAR-10.

    @param[in]  train_bs  Training batch size.
    @param[in]  test_bs   Testing batch size.

    @returns a tuple with the training and testing dataloaders.
    """
    # Load training and testing sets
    train_ds = torchvision.datasets.CIFAR10(root='./data', train=True, 
                                            download=True, transform=train_preproc_tf)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, transform=test_preproc_tf)

    # Create dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_bs, 
                                           shuffle=True, num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_bs, 
                                          shuffle=False, num_workers=num_workers)

    return train_dl, test_dl


def build_preprocessing_transforms(size: int = 384, randaug_n: int = 2, randaug_m: int = 14):
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
    train_preproc_tf.transforms.insert(0, vitcifar10.randomaug.RandAugment(randaug_n, randaug_m))
    
    # Preprocessing for testing
    test_preproc_tf = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    return train_preproc_tf, test_preproc_tf


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

    return net


def main():
    # Parse command line parameters
    args = parse_cmdline_params()

    # Prepare preprocessing layers
    train_preproc_tf, test_preproc_tf = build_preprocessing_transforms()

    # Get dataloaders for training and testing
    train_dl, test_dl = load_dataset(train_preproc_tf, test_preproc_tf)

    # Build model
    net = build_model()


if __name__ == '__main__':
    main()
