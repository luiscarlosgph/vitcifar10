Description
-----------

This repository contains the Python package `vitcifar10`, which is a Vision Transformer baseline code for training and testing on CIFAR-10.


Install
-------

```bash
$ python3 setup.py install
```


Train
-----

* Launch training:

```bash
$ python3 -m vitcifar10.train --lr 1e-4 --opt adam --nepochs 200 --bs 16 --cpdir checkpoints --logdir logs
```

* Launch Tensorboard:

```bash
$ TODO
```


Test
----

```bash
$ TODO
```


