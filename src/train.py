#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief  Main script to kick off the training.
        Some of this code has been inspired by: https://github.com/kentaroy47/vision-transformers-cifar10    
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   25 Oct 2022.
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


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '--lr' :     'Learning rate (required: True)',
        '--opt':     'Optimizer (required: True)', 
        '--nepochs': 'Number of epochs (required: True)',
        '--bs':      'Training batch size (required: True)',
        '--cpdir':   'Path to the checkpoint directory (required: True)', 
        '--logdir':  'Path to the log directory (required: True)',
        '--resume':  'Path to the checkpoint file (required: False)',
        '--cpint':   'Checkpoint interval (required: True)',  
        '--data':    'Path to the CIFAR-10 data directory (required: True)',
        '--seed':    'Random seed (required: False)',
    }
    return help_msg[short_option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch ViT for training/validation on CIFAR-10.')
    args.add_argument('--lr', required=True, type=float, 
                      help=help('--lr'))  
    args.add_argument('--opt', required=True, type=str, 
                      help=help('--opt'))
    args.add_argument('--nepochs', required=True, type=int, 
                      help=help('--nepochs'))
    args.add_argument('--bs', required=True, type=int,
                      help=help('--bs'))
    args.add_argument('--cpdir', required=True, type=str,
                      help=help('--cpdir'))
    args.add_argument('--logdir', required=True, type=str,
                      help=help('--logdir'))
    args.add_argument('--cpint', required=True, type=int,
                      help=help('--cpint'))
    args.add_argument('--data', required=True, type=str,
                      help=help('--data'))
    args.add_argument('--resume', required=False, type=str, default=None,
                      help=help('--resume'))
    args.add_argument('--seed', required=False, type=int, default=None,
                      help=help('--seed'))
    
    return  args.parse_args()


def load_dataset(train_preproc_tf, valid_preproc_tf, data_dir, train_bs: int = 512, 
                 valid_bs: int = 100, num_workers: int = 8, valid_size: float = 0.1):
    """
    @brief Function that creates the dataloaders of CIFAR-10.

    @param[in]  train_bs  Training batch size.
    @param[in]  valid_bs  Testing batch size.

    @returns a tuple with the training and validation dataloaders.
    """
    # Load training and validation sets
    train_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, 
                                            download=True, 
                                            transform=train_preproc_tf)
    valid_ds = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                           download=True, 
                                           transform=valid_preproc_tf)
    
    # Create train/val split of the CIFAR-10 training set
    num_train = len(train_ds)

    # FIXME: for debugging purposes, comment line below in production
    #num_train = 100

    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    # NOTE: Random shuffle of the train/val images
    np.random.shuffle(indices)

    # Create samplers 
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)

    # Create dataloaders
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=train_bs, 
                                           sampler=train_sampler,
                                           num_workers=num_workers)
    valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=valid_bs, 
                                          sampler=valid_sampler,
                                          num_workers=num_workers)

    return train_dl, valid_dl


def build_optimizer(net, lr, opt: str = "adam"):
    if opt.lower() == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif opt.lower() == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    
    return optimizer


def resume(checkpoint_path, net, optimizer, scheduler, scaler):
    """
    @param[in]       checkpoint_path  Path to the checkpoint file (extension .pt).
    @param[in, out]  net              Initialized PyTorch model.
    @param[in, out]  optimizer        Initialized solver.
    @param[in, out]  scheduler        Initialized LR scheduler.
    @param[in, out]  scaler           Initialized gradient scaler.
    """
    print('[INFO] Resuming from checkpoint ...')
     
    # Check that the checkpoint directory exists
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError('[ERROR] You want to resume from the last checkpoint, ' \
            + 'but there is not directory called "checkpoint"')
    
    # Load state
    state = torch.load(checkpoint_path)
    
    # Update model with saved weights
    net.load_state_dict(state['net'])

    # Update optimizer with saved params
    optimizer.load_state_dict(state['optimizer'])

    # Update scheduler with saved params
    scheduler.load_state_dict(state['scheduler'])

    # Update scaler with saved params
    scaler.load_state_dict(state['scaler'])
    
    print('[INFO] Resuming from checkpoint ...')

    return state['lowest_valid_loss'], state['epoch'] + 1


def train(net: torch.nn, train_dl, loss_func, optimizer, scheduler, scaler, device='cuda'):
    """
    @brief Train the model for a single epoch.
    
    @param[in, out]  net       Model.
    @param[in]       train_dl  Dataloader for the trainig data.
    @param[in]       loss_func Pointer to the loss function.
    @param[in, out]  optimizer Optimizer to be used for training.
    @param[in]       scaler    Gradient scaler.
    """

    # Set network in train mode
    net.train()

    # Create progress bar
    pbar = tqdm.tqdm(enumerate(train_dl), total=len(train_dl))

    # Run forward-backward over all the samples
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        # Train with amp
        with torch.cuda.amp.autocast(enabled=True):
            outputs = net(inputs)
            loss = loss_func(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Display loss and accuracy on the progress bar
        display_loss = train_loss / (batch_idx + 1)
        display_acc = 100. * correct / total
        pbar.set_description("Training loss: %.3f | Acc: %.3f%% (%d/%d) | LR: %.2E" % (display_loss, 
            display_acc, correct, total, scheduler.get_last_lr()[0]))

    # Add step to the LR cosine scheduler
    #scheduler.step(epoch - 1)
    scheduler.step()

    return display_loss, display_acc


def main():
    # Parse command line parameters
    args = parse_cmdline_params()

    # Fix random seeds for reproducibility
    if args.seed is None:
        args.seed = random.SystemRandom().randrange(0, 2**32)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # Prepare preprocessing layers
    train_preproc_tf, valid_preproc_tf = vitcifar10.build_preprocessing_transforms()

    # Get dataloaders for training and testing
    train_dl, valid_dl = load_dataset(train_preproc_tf, valid_preproc_tf, 
                                      args.data, train_bs=args.bs)

    # Build model
    net = vitcifar10.build_model()

    # Use cross-entropy loss
    loss_func = torch.nn.CrossEntropyLoss()
    
    # Build optimizer
    optimizer = build_optimizer(net, args.lr, args.opt)

    # Build LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.nepochs)

    # Setup gradient scaler
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    # Setup Tensorboard
    writer = vitcifar10.setup_tensorboard(args.logdir)

    # Resume from the last checkpoint if requested
    lowest_valid_loss = np.inf
    start_epoch = 0
    model_best = False
    if args.resume is not None:
        lowest_valid_loss, start_epoch = resume(args.resume, net, optimizer, scheduler, scaler)

    # Enable multi-GPU support
    net = torch.nn.DataParallel(net)
    torch.backends.cudnn.benchmark = True 

    # Create lists to store the losses and metrics
    train_loss_over_epochs = []
    train_acc_over_epochs = []
    valid_loss_over_epochs = []
    valid_acc_over_epochs = []
    
    # Training loop  
    for epoch in range(start_epoch, args.nepochs):  
        print("\n[INFO] Epoch: {}".format(epoch))
        start = time.time()

        # Run a training epoch
        train_loss, train_acc = train(net, train_dl, loss_func, optimizer, scheduler, scaler)

        # Run testing
        valid_loss, valid_acc, _, _ = vitcifar10.valid(net, valid_dl, loss_func)

        # Update lowest validation loss
        if valid_loss < lowest_valid_loss:
            lowest_valid_loss = valid_loss
            model_best = True
        else:
            model_best = False

        # Save checkpoint
        if epoch % args.cpint == 0:
            print('[INFO] Saving model for this epoch ...')
            state = {
                "net":               net.state_dict(),
                "optimizer":         optimizer.state_dict(),
                "scheduler":         scheduler.state_dict(),
                "scaler":            scaler.state_dict(),
                "lowest_valid_loss": lowest_valid_loss,
                "epoch":             epoch,
            }
            if not os.path.isdir(args.cpdir):
                os.mkdir(args.cpdir)
            checkpoint_path = os.path.join(args.cpdir, "epoch_{}.pt".format(epoch))
            torch.save(state, checkpoint_path) 
            print('[INFO] Saved.')

        # If it is the best model, let's copy it
        if model_best:
            print('[INFO] Saving best model ...')
            model_best_path = os.path.join(args.cpdir, 'model_best.pt')
            torch.save(state, model_best_path) 
            print('[INFO] Saved.')
        
        # Store training losses and metrics
        train_loss_over_epochs.append(train_loss)
        train_acc_over_epochs.append(train_acc)
        
        # Store validation losses and metrics
        valid_loss_over_epochs.append(valid_loss)
        valid_acc_over_epochs.append(valid_acc)

        # Log training losses and metrics in Tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        
        # Log validation losses and metrics in Tensorboard
        writer.add_scalar('Loss/valid', valid_loss, epoch)
        writer.add_scalar('Accuracy/valid', valid_acc, epoch)
       
        print("[INFO] Epoch: {} finished.".format(epoch))
    

if __name__ == '__main__':
    main()
