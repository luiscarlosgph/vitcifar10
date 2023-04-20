#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@brief  Script that launches the vitcifar10.test script over a set of 
        models (different random initialisations).

@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   1 Mar 2023.
"""

import argparse
import subprocess
import re


def help(short_option):
    """
    @returns The string with the help information for each command line option.
    """
    help_msg = {
        '--program': 'Python script to run the inference (required: True)',
        '--cifar'  : 'Path to the CIFAR-10 dataset directory (required: True)',
        '--iter'   : 'Number of iterations (required: True)',
        '--budgets': 'Number of budgets to test (required: True)',
        '--bs'     : 'Batch size (required: False). Default value is 1.',
        '--checkpoint-dir': 'Path to the checkpint directory (required: True)',
    }
    return help_msg[short_option] 


def parse_cmdline_params():
    """@returns The argparse args object."""
    args = argparse.ArgumentParser(description='PyTorch inference cross-validation.')
    args.add_argument('--program', required=True, type=str, help=help('--program'))
    args.add_argument('--cifar', required=True, type=str, help=help('--cifar'))
    args.add_argument('--iter', required=True, type=str, help=help('--iter'))
    args.add_argument('--budgets', required=True, type=str, help=help('--budgets'))
    args.add_argument('--bs', required=False, type=str, help=help('--bs'), default=1)
    args.add_argument('--checkpoint-dir', required=True, type=str, help=help('--checkpoint-dir'))

    return  args.parse_args()


def main():
    # Parse command line parameters
    args = parse_cmdline_params()
    args.budgets = eval(args.budgets)

    accuracies = []
    for budget in args.budgets:
        for j in range(int(args.iter)):
            # Build command string
            cmd = args.program + ' '
            cmd += '--data ' + args.cifar + ' ' 
            cmd += '--resume ' + args.checkpoint_dir + '/budget_' + str(budget)
            cmd += '/iter_' + str(j) + '/model_best.pt '
            cmd += '--bs ' + str(args.bs)
            
            # Run command and get output  
            print('Running command:', cmd)
            output = subprocess.check_output(cmd.split(' ')).decode('utf-8')
            
            # Get accuracy
            match = re.search("Percentage of images correctly classified: ([0-9.]+)%", output)
            accuracies.append(float(match.group(1)))
    
    # Print all the accuracies
    for acc in accuracies:
        print(acc)


if __name__ == "__main__":
    main()
