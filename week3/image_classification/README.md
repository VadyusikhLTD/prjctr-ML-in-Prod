# Purpose

This is a part of homework to practice with WandB. 
To test WandB I've selected simple image classification task - Fashion MNIST.  
As a reference for some solution (argument processing, usage Python Data classes) was selected [text-classification pipeline](https://github.com/huggingface/transformers/tree/main/examples/pytorch/text-classification).

# Image classification

Dataset - Fashion MNIST.
Model - simple 3 layer CNN.

# Model card for Fashion MNIST image classification

## General description 
Model goal is to classify input images to 1 of 10 classes from Fashion MNIST dataset.
 - Input : gray-scale image 28x28 px
 - Output : 10 values for each class, biggest value - define detected class
 - Model architecture - simple CNN with 3 layers

## Metrics 
As a loss metric was chosen standard for simple classification tasks - cross-entropy loss.
As an evaluation metric was chosen accuracy metric and achieved 89% on validation dataset.

## Used data

For train and evaluation was used public Fashion MNIST dataset, provided by [torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.FashionMNIST.html). 

## Law and Ethical Consideration
This model is fully safe for commercial and non-commercial use. 
And limited only with Licenses of Torch library and Fashion MNIST dataset.
No related ethical consideration are known to auther at the moment.
