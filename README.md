Description
-----------

This repository contains the Python package `vitcifar10`, which is a Vision Transformer (ViT) baseline code for training and testing on CIFAR-10. This implementation only supports CUDA (no CPU training).


Install
-------

```bash
$ python3 setup.py install
```


Train
-----

* Launch training:

```bash
$ python3 -m vitcifar10.train --lr 1e-4 --opt adam --nepochs 200 --bs 16 --cpdir checkpoints --logdir logs --cpint 5
```

* Resume training from a checkpoint:
```bash
$ python3 -m vitcifar10.train --lr 1e-4 --opt adam --nepochs 200 --bs 16 --cpdir checkpoints --logdir logs --cpint 5 --resume checkpoints/epoch_21.pt
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


* Launch Tensorboard:

```bash
$ python3 -m  tensorboard.main --logdir logs --bind_all
```


Test
----

```bash
$ TODO
```


