# Implementations of Various Machine Learning Algorithms

## Perceptron
### A Single-Layered Perceptron Written for the MNIST Dataset
`preprocessing.py` must be run before `perceptron.py`. It will break the MNIST files out of csv format,
gets them into the format needed by the perceptron, and dumps them to .json objects. This is done to 
amortize preprocessing time over several tests.

## MLP
### A Two-Layered Neural Network Using Feed-Forward and Backpropagation Written for the MNIST Dataset.
The mlp expects the .json objects produced by perceptron/preprocessing in its local directory.

Currently backpropagation is hard-coded to expect only two layers (one hidden layer + output layer). 
All the other components are configured for arbitrary layers, however. Eventually backprop will be
updated to manage an arbitrary number of hidden layers.

## SVM
### Measuring Sklearn's SVM with Different Versions of Spambase Dataset
`processing.py` scales the spambase data to standard mean, unit variance, shuffles it, and generates a ROC curve.
It then measures accuracy as a function of number of features; first, on the most heavily weighted features, then
on randomly-selected features.

## Naive Bayes
### A probablistic classifier for the Spambase Dataset
Naive Bayes makes the assumption that the probabilities of all features occuring given a class are independent.
This implementation will intermittently fail to provide good classifications. This happens when the exmaples
are shuffled such that one or more features in every example are zero for a given class in the training set, but
not the test set. This could be obviated by making the shuffle more deterministic, or by implementing Lapalace
Smoothing.

## K-Means Clustering
### An unsupervised clustering algorithm for classifying handwritten digits
This K-means clusterer will take the best out of five tries given randomly-initialized cluster centers. It
works by computing the distances between all possible example-centroid pairs given the training set, and then
resetting the cluster centroid to the mean of its cluster. It does this in a loop until accuracy stops 
changing. Higher accuracies (in the 90% range) are seen for the second iteration where K=30.