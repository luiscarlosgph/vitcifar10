#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@brief Setup for the vit-cifar10 package.
@author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
"""
import setuptools
import unittest

# Read the contents of the README file
from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(name='vitcifar10',
    version='0.0.2',
    description='Python module to train and test on CIFAR-10.',
    author='Luis C. Garcia-Peraza Herrera',
    author_email='luiscarlos.gph@gmail.com',
    license='MIT License',
    url='https://github.com/luiscarlosgph/dentalseg',
    packages=[
        'vitcifar10',
    ],
    package_dir={
        'vitcifar10': 'src',
    },
    install_requires = [
        'numpy', 
        'torch',
        'torchvision',
        'timm',
        'tensorboard',
        'sklearn',
        'pillow',
    ],
    long_description=long_description,
    long_description_content_type='text/markdown',
    test_suite = 'test',
)
