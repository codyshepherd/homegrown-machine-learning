"""
Cody Shepherd
preprocessing.py
CS 545
Homework 1

This script breaks the MNIST dataset out of its .csv files, separates out the classes,
normalizes the values in each sample, and dumps the samples and classes as separate 
matrices into .json files.
"""
import csv
import json
import numpy as np

Data = []
samples = []

with open('mnist_train.csv', 'r') as fname:
    data = csv.reader(fname)

    for item in data:
        Data.append(item)

Classes = np.array([float(item[0]) for item in Data])
Data = [item[1:] for item in Data]

samples = np.array(Data, dtype=float)

for i in xrange(len(samples)):
    samples[i] = map(lambda x: x/255, samples[i])

with open('samples.json', 'w+') as fname:
    json.dump(samples.tolist(), fname)

with open('classes.json', 'w+') as fname:
    json.dump(Classes.tolist(), fname)


Data = []

with open('mnist_test.csv', 'r') as fname:
    data = csv.reader(fname)

    for item in data:
        Data.append(item)

Classes = np.array([float(item[0]) for item in Data])
Data = [item[1:] for item in Data]

samples = np.array(Data, dtype=float)

for i in xrange(len(samples)):
    samples[i] = map(lambda x: x/255, samples[i])

with open('test_samples.json', 'w+') as fname:
    json.dump(samples.tolist(), fname)

with open('test_classes.json', 'w+') as fname:
    json.dump(Classes.tolist(), fname)
