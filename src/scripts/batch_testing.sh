# @brief  Script that launches the vitcifar10.test script over a set of 
#         models (different random initialisations).
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   1 Mar 2023.

# Constants
BATCH_SIZE=1
PROGRAM="python3 -m vitcifar10.test"
CIFAR10_DATA_DIR="./data"

$PROGRAM --data $CIFAR_10_DATA_DIR --resume active_learning_2022/fully_supervised_without_pretraining/random/checkpoints/budget_100/iter_5/model_best.pt --bs $BATCH_SIZE
