Description
-----------

This repository contains the Python package `vitcifar10`, which is a Vision Transformer (ViT) baseline code for training and testing on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html). This implementation only supports CUDA (no CPU training). 

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

2. Launch `vitcifar10` container:
   ```bash
   $ docker run --name wild_vitcifar10 --runtime=nvidia -v /dev/shm:/dev/shm vitcifar10:latest &
   ```
   
3. Get a terminal inside the container (you can execute this command multiple times to get multiple container terminals):
   ```bash
   $ docker exec -it wild_vitcifar10 /bin/zsh
   $ cd $HOME
   ```
   
4. Launch CIFAR-10 training:
   ```
   $ python3 -m vitcifar10.train --lr 1e-4 --opt adam --nepochs 200 --bs 16 --cpdir checkpoints --logdir logs --cpint 5 --data ./data
   ```
   
5. Launch CIFAR-10 testing:
   ```
   $ python3 -m vitcifar10.test --data ./data --resume checkpoints/model_best.pt --bs 1
   ```

If you want to kill the container run `$ docker kill wild_vitcifar10`. 

To remove it execute `$ docker rm wild_vitcifar10`.


Install with pip
----------------

```bash
$ pip install vitcifar10 --user
```


Install from source
-------------------

```bash
$ git clone https://github.com/luiscarlosgph/vitcifar10.git
$ cd vitcifar10
$ python3 setup.py install
```


Download CIFAR-10 training and testing data
-------------------------------------------

```bash
$ python3 -m vitcifar10.download_data --data ./data
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
   * `--seed`: fix random seed for reproducibility.


* Launch Tensorboard:

   ```bash
   $ python3 -m  tensorboard.main --logdir logs --bind_all
   ```

Train multiple models from scratch (different random seeds)
-----------------------------------------------------------

```bash
$ python3 -m vitcifar10.run --lr 1e-4 --opt adam --nepochs 100 --bs 16 --cpdir checkpoints --logdir logs --cpint 5 --niter 5 --data data
```
* Options:
   * `--niter`: number of training cycles to run, e.g. `--niter 5` will lead to training five networks.

The rest of the options are identical to those of `vitcifar10.train`.


Train with a custom dataloader
-------------------------------

This is useful when you want to experiment with the input/labels without modifying the model. To do so, `vitcifar10.run` accepts the parameter `--data-loader`, e.g.:

```bash
$ python3 -m vitcifar10.run --lr 1e-4 --opt adam --nepochs 100 --bs 16 --cpdir checkpoints --logdir logs --cpint 10 --niter 5 --data data --data-loader <python_class_name>
```

The dataloader class that you pass must align with the following requirements:
   * Constructor `__init__` should accept the batch size as first parameter. 
   * Methods `__next__` and `__iter__` should be implemented.


Test on CIFAR-10
----------------

```bash
$ python3 -m vitcifar10.test --data ./data --resume checkpoints/model_best.pt --bs 1
```

* Options:
   * `--data`: path to the directory where the dataset will be stored.
   * `--resume`: path to the checkpoint file you want to test.


Perform inference on a single image
-----------------------------------

After training, you can classify images such as this [dog](data/dog.jpg) or this [cat](data/cat.jpg) following:

```bash
$ python3 -m vitcifar10.inference --image data/dog.jpg --model checkpoints/model_best.pt 
It is a dog!
```

```bash
$ python3 -m vitcifar10.inference --image data/cat.jpg --model checkpoints/model_best.pt
It is a cat!
```


Training | validation | testing splits
----------------------------------

This code uses [torchvision](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html) to download and load the CIFAR-10 dataset. The constructor of the class `torchvision.datasets.CIFAR10` has a boolean parameter called `train`. 

In our code we set `train=True` to obtain the images for training and validation, using 90% for **training** (45K images) and 10% for **validation** (5K images). The validation set is used to discover the best model during training (could also be used for hyperparameter tunning or early stopping). For **testing**, we set `train=False`. The testing set contains 10K images. 


Author
------

Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).


License
-------

This repository is shared under an [MIT license](LICENSE).


