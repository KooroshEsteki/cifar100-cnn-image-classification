# CIFAR-100 CNN Image Classification

This mini project builds a Convolutional Neural Network CNN using Keras to classify images from the CIFAR-100 dataset.

The CIFAR-100 dataset contains 100 different classes of objects. Each image is a 32x32 RGB image.

---

## Project Goal

The goal of this project is to build, train, and test a CNN model that can classify CIFAR-100 images into 100 categories.

The code returns:

1. The trained CNN model
2. The test results, including loss and accuracy

---

## Dataset

The CIFAR-100 dataset is loaded directly from Keras using:

```python
from keras.datasets import cifar100

(x_train, y_train), (x_test, y_test) = cifar100.load_data()
