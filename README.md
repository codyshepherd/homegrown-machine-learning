# Implementations of Various Machine Learning Algorithms

## Perceptron
### A Single-Layered Perceptron Written for the MNIST Dataset
`preprocessing.py` must be run before `perceptron.py`. It will break the MNIST files out of csv format,
gets them into the format needed by the perceptron, and dumps them to .json objects. This is done to 
amortize preprocessing time over several tests.

## MLP
### A Two-Layered Neural Network Using Feed-Forward & Backpropagation Written for the MNIST Dataset
The mlp expects the .json objects produced by perceptron/preprocessing in its local directory.
---
Currently backpropagation is hard-coded to expect only two layers (one hidden layer + output layer). 
All the other components are configured for arbitrary layers, however. Eventually backprop will be
updated to manage an arbitrary number of hidden layers.

## SVM
### Measuring Sklearn's SVM with Different Versions of Spambase Dataset
`processing.py` scales the spambase data to standard mean, unit variance, shuffles it, and generates a ROC curve.
It then measures accuracy as a function of number of features; first, on the most heavily weighted features, then
on randomly-selected features.
