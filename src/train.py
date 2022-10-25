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
        '--lr' :     'Learning rate (required: True)',
        '--opt':     'Optimizer (required: True)', 
        '--nepochs': 'Number of epochs (required: True)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch ViT for training/testing on CIFAR-10.')
    args.add_argument('--lr', required=True, type=float, 
                      help=help('--lr'))  
    args.add_argument('--opt', required=True, type=str, 
                      help=help('--opt'))
    args.add_argument('--nepochs', required=True, type=int, 
                      help=help('--nepochs'))

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
                                            download=True, 
                                            transform=train_preproc_tf)
    test_ds = torchvision.datasets.CIFAR10(root='./data', train=False, 
                                           download=True, 
                                           transform=test_preproc_tf)

    # Create dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_bs, 
                                           shuffle=True, 
                                           num_workers=num_workers)
    test_dl = torch.utils.data.DataLoader(test_ds, batch_size=test_bs, 
                                          shuffle=False, 
                                          num_workers=num_workers)

    return train_dl, test_dl


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


def build_optimizer(net, lr, opt: str = "adam"):
    if opt.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif opt.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    return optimizer


def train(net: torch.nn, train_dl, loss, optimizer):
    """
    @brief Train the model for a single epoch.
    
    @param[in, out]  net       PyTorch model.
    @param[in, out]  train_dl  PyTorch dataloader for the trainig data.
    @param[in]       loss      Pointer to the loss function.
    @param[in, out]  optimizer PyTorch optimizer to be used for training.
    """
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(train_dl):
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=True):
            outputs = net(inputs)
            loss = loss(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # TODO: call function to show progress bar for the epoch here

        #progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #    % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    return train_loss / (batch_idx + 1)


def test(epoch: int):
    # TODO
    pass


def main():
    # Parse command line parameters
    args = parse_cmdline_params()

    # Prepare preprocessing layers
    train_preproc_tf, test_preproc_tf = build_preprocessing_transforms()

    # Get dataloaders for training and testing
    train_dl, test_dl = load_dataset(train_preproc_tf, test_preproc_tf)

    # Build model
    net = build_model()

    # Use cross-entropy loss
    loss = torch.nn.CrossEntropyLoss()
    
    # Build optimizer
    optimizer = build_optimizer(net, args.lr, args.opt)

    # Build LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepochs)


    train(net, train_dl, loss, optimizer)

    
if __name__ == '__main__':
    main()
