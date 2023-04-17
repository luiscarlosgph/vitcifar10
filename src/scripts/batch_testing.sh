#!/bin/bash
#
# @brief  Script that launches the vitcifar10.test script over a set of 
#         models (different random initialisations).
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   1 Mar 2023.

# Constants
BATCH_SIZE=1
PROGRAM="python3 -m vitcifar10.test"
CIFAR10_DATA_DIR="./data"
CHECKPOINT_DIR="./active_learning_2022/fully_supervised_without_pretraining/random/checkpoints"

# Create an array of budgets from the user input
echo "Reading the list of budget iterations ... "
IFS=',' 
budgets=($2)

# Create the array of iterations
iterations=("0", "1", "2", "3", "4", "5", "6", "7", "8", "9")

for budget in "${budgets[@]}"
do 
    # Build test command
    cmd="$PROGRAM --data $CIFAR10_DATA_DIR --resume $CHECKPOINT_DIR/budget_$budget/iter_$budget/model_best.pt --bs $BATCH_SIZE | grep 'Percentage of images correctly classified' | cut -d ' ' -f7 | cut -d '%' -f1"
    
    # Run inference
    #acc=$($cmd)
    print($cmd)
done

