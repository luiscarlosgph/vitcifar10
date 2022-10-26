Description
-----------

This repository contains the Python package `vitcifar10`, which is a Vision Transformer (ViT) baseline code for training and testing on CIFAR-10. This implementation only supports CUDA (no CPU training). 

The idea of this repository is not to have a very flexible implementation, but one that can be used as a baseline for research with results on testing close to the state of the art.

The code in this repository relies on [timm](https://github.com/rwightman/pytorch-image-models), which is a Python package with many state-of-the-art models implemented and pretrained.


Use within a Docker container
---------------------------

If you do not have Docker, [here](https://github.com/luiscarlosgph/how-to/tree/main/docker) you can find a tutorial to install it.

1. Build `vitcifar10` Docker image:
   ```bash
   $ git clone https://github.com/luiscarlosgph/vitcifar10.git
   $ cd vitcifar10/docker
   $ docker build -t vitcifar10 .
   ```

2. Run `vitcifar10` container:
   ```bash
   $ docker run TODO
   ```


Install with pip
----------------

```bash
$ pip install vitcifar10
```


Install from source
-------------------

```bash
$ git clone https://github.com/luiscarlosgph/vitcifar10.git
$ cd vitcifar10
$ python3 setup.py install
```


Train on CIFAR-10
-----------------

* Launch training:

   ```bash
   $ python3 -m vitcifar10.train --lr 1e-4 --opt adam --nepochs 200 --bs 16 --cpdir checkpoints --logdir logs --cpint 5 --data ./data
   ```

* Resume training from a checkpoint:
   ```bash
   $ python3 -m vitcifar10.train --lr 1e-4 --opt adam --nepochs 200 --bs 16 --cpdir checkpoints --logdir logs --cpint 5 --data ./data --resume   checkpoints/epoch_21.pt
   ```

* Options:
   * `--lr`: learning rate.
   * `--opt`: optimizer, choose either `sgd` or `adam`.
   * `--nepochs`: number of epochs to execute.
   * `--bs`: training batch size.
   * `--cpdir`: checkpoint directory.
   * `--logdir`: path to the directory where the Tensorboard logs will be saved.
   * `--cpint`: interval of epochs to save a checkpoint of the training process.
   * `--resume`: path to the checkpoint file you want to resume.
   * `--data`: path to the directory where the dataset will be stored.


* Launch Tensorboard:

   ```bash
   $ python3 -m  tensorboard.main --logdir logs --bind_all
   ```


Test on CIFAR-10
----------------

```bash
$ python3 -m vitcifar10.test --data ./data --resume checkpoints/model_best.pt
```

* Options:
   * `--data`: path to the directory where the dataset will be stored.
   * `--resume`: path to the checkpoint file you want to test.

<!--
Perform inference on a single image
-----------------------------------

TODO
-->



Author
------

Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).


License
-------

This repository is shared under an [MIT license](LICENSE).


