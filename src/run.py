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

# My imports
import vitcifar10


def help(option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '--niter':   'Number of training cycles to run (required: True)',
        '--lr' :     'Learning rate (required: True)',
        '--opt':     'Optimizer (required: True)', 
        '--nepochs': 'Number of epochs (required: True)',
        '--bs':      'Training batch size (required: True)',
        '--cpdir':   'Path to the checkpoint directory (required: True)', 
        '--logdir':  'Path to the log directory (required: True)',
        '--cpint':   'Checkpoint interval (required: True)',  
        '--data':    'Path to the CIFAR-10 data directory (required: True)',
        '--resume':  'Boolean to resume all the training cycles (required: False)',
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
    args.add_argument('--resume', required=False, type=bool, default=False,
                      help=help('--resume'))
    
    return  args.parse_args()


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


def main():
    # Parse command line parameters
    args = parse_cmdline_params()

    # Loop of training cycles
    for i in range(0, args.niter):
        print("[INFO] Iteration {} ...".format(i))

        # Get random seed 
        seed = random.SystemRandom().randrange(0, 2**32)

        # Create training command for this cycle
        iter_cpdir = args.cpdir + "/iter_" + str(i)
        cmd = "python3 -m vitcifar10.train " \
            + "--lr " + str(args.lr) + " " \
            + "--opt " + args.opt + " " \
            + "--nepochs " + str(args.nepochs) + " " \
            + "--bs " + str(args.bs) + " " \
            + "--cpdir " + iter_cpdir + " " \
            + "--logdir " + args.logdir + "/iter_" + str(i) + " " \
            + "--cpint " + str(args.cpint) + " " \
            + "--data " + args.data + " " \
            + "--seed " + str(seed)
        
        # Resume from the last checkpoint if indicated by the user
        path_to_last_checkpoint = None
        if args.resume:
            path_to_last_checkpoint = find_last_epoch(iter_cpdir)
        
        # Skip this iteration if we reached the last epoch
        if path_to_last_checkpoint is not None:
            # Resume the training of this iteration from the last epoch
            cmd += " --resume " + path_to_last_checkpoint

            # Get the epoch number and compare it with the number of epochs 
            # passed as a command line argument
            pattern = "^.*epoch_([0-9]+).pt$"
            regex = re.compile(pattern)
            m = regex.match(path_to_last_checkpoint)
            epoch_number = int(m.group(1))

            if epoch_number < args.nepochs:
                os.system(cmd)
        else:
            # Launch training in a subshell 
            #os.system(cmd + " &")
            os.system(cmd)

        print("[INFO] Iteration {} finished.".format(i))


if __name__ == '__main__':
    main()
