#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief  Script to run multiple training cycles with different seeds.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   28 Oct 2022.
"""
import argparse
import os
import glob
import random
import natsort
import re
import copy
import signal
import sys
import importlib
import os

# My imports
import vitcifar10


def signal_handler(sig, frame):
    print('[INFO] You pressed Ctrl+C, closing everything ...')
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


def help(option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '--niter':       'Number of training cycles to run (required: True)',
        '--lr' :         'Learning rate (required: True)',
        '--opt':         'Optimizer (required: True)', 
        '--nepochs':     'Number of epochs (required: True)',
        '--bs':          'Training batch size (required: True)',
        '--cpdir':       'Path to the checkpoint directory (required: True)', 
        '--logdir':      'Path to the log directory (required: True)',
        '--cpint':       'Checkpoint interval (required: True)',  
        '--data':        'Path to the CIFAR-10 data directory (required: True)',
        '--data-loader': 'String pointing to a particular active learning iterative dataloader (required: False)',
        #'--resume':  'Boolean to resume all the training cycles (required: False)',
    }
    return help_msg[option]


def parse_cmdline_params():
    """@returns The argparse args object."""
    desc = 'Run many training cycles of the ViT on CIFAR-10.'
    args = argparse.ArgumentParser(description=desc)
    args.add_argument('--niter', required=True, type=int, 
                      help=help('--niter'))  
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
    args.add_argument('--data-loader', required=False, default=None, type=str,
                      help=help('--data-loader'))

    # Parse arguments
    args = args.parse_args()
    
    # Get the dataloader function
    if args.data_loader is not None:
        module_name, class_name = args.data_loader.split('.')
        args.data_loader = getattr(__import__(module_name), class_name)
    
    return args 


def find_last_epoch(checkpoint_path: str) -> str:
    """
    @brief Find the .pt file corresponding to the last epoch.
    @returns the path to the file.
    """
    list_of_files = [x for x in glob.glob(checkpoint_path + '/*.pt') if 'epoch_' in x]
    latest_file = natsort.natsorted(list_of_files, alg=natsort.ns.IGNORECASE)
    if latest_file:
        return latest_file[-1]
    else:
        return None


def only_model_best(checkpoint_path: str) -> bool:
    """@returns True if 'model_best.pt' is the only file in the checkpoint folder."""
    list_of_files = [x for x in os.listdir(checkpoint_path) if not x.startswith('.')]
    if len(list_of_files) == 1 and list_of_files[0] == 'model_best.pt':
        return True
    else:
        return False


def run_cycles(args, train_dl=None, valid_dl=None):
    """@brief Loop of training cycles."""
    for train_iter in range(0, args.niter):
        print("[INFO] Iteration {} ...".format(train_iter))

        # Deep copy of the arguments, they will be modified for this iteration
        args_copy = copy.deepcopy(args)

        # Arguments we need to tweak for this specific iteration: --cpdir, --logdir, --seed
        args_copy.cpdir = args.cpdir + "/iter_" + str(train_iter)
        args_copy.logdir = args.logdir + "/iter_" + str(train_iter)
        args_copy.seed = random.SystemRandom().randrange(0, 2**32)
        
        # Discover if the current iteration is finished, and where to resume it from 
        if not only_model_best(args_copy.cpdir):
            path_to_last_checkpoint = find_last_epoch(args_copy.cpdir)
            if path_to_last_checkpoint is None:
                # If this iteration has not started at all, we run it
                args_copy.resume = None
                vitcifar10.train.main(args_copy, train_dl, valid_dl)
            else:
                # This iteration has started, let's see if it has finished or not
                pattern = "^.*epoch_([0-9]+).pt$"
                regex = re.compile(pattern)
                m = regex.match(path_to_last_checkpoint)
                epoch_number = int(m.group(1))
                if epoch_number < args.nepochs:
                    # It has not finished, let's resume it
                    print('Last checkpoint dir:', args_copy.cpdir)
                    print('Last checkpoint path:', path_to_last_checkpoint)
                    args_copy.resume = path_to_last_checkpoint
                    vitcifar10.train.main(args_copy, train_dl, valid_dl)
            
            # We have finished the training, let's remove the checkpoints
            checkpoints = [f for f in os.listdir(args_copy.cpdir) if f.startswith('epoch_')]
            for f in checkpoints:
                os.remove(os.path.join(args_copy.cpdir, f))

        print("[INFO] Iteration {} finished.".format(train_iter))


def main():
    # Parse command line parameters
    args = parse_cmdline_params()
    
    # Besides running several training cycles for reporting purposes,
    # we could run an outer loop with different dataloader configurations
    if args.data_loader is None:
        run_cycles(args)
    else:
        original_cpdir = args.cpdir
        original_logdir = args.logdir
        dl = args.data_loader(args.bs)
        for i, (train_dl, valid_dl) in enumerate(dl):
            args.cpdir = original_cpdir + '/' + str(dl)
            args.logdir = original_logdir + '/' + str(dl)
            if not os.path.isdir(args.cpdir):
                os.mkdir(args.cpdir)
            if not os.path.isdir(args.logdir):
                os.mkdir(args.logdir)

            # Reduce batch size if it is larger than the dataset
            #if args.bs > len(train_dl):
            #    args.bs = len(train_dl)

            # Run as many complete training cycles as the user wants
            run_cycles(args, train_dl, valid_dl)

    
if __name__ == '__main__':
    main()
