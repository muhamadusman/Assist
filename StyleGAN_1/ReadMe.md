The code is an extension of StyleGAN provided by [NVlabs]([https://github.com/NVlabs/stylegan3](https://github.com/NVlabs/stylegan)). 

this code uses tfrecords as input data 

This repository contains the official TensorFlow implementation of the following paper:

> **A Style-Based Generator Architecture for Generative Adversarial Networks**<br>
> Tero Karras (NVIDIA), Samuli Laine (NVIDIA), Timo Aila (NVIDIA)<br>
> https://arxiv.org/abs/1812.04948
By changing the provided files in the directories and then running the `train.py` file you can train the model

conda environment can be created by conda env create -f environment.yml 

change the paths in config.py

The number of GPUs configuration can be selected from train.py

activate conda env and run train.py to train 
