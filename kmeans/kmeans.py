'''
Cody Shepherd
kmeans.py
CS 545: Machine Learning
Homework 5

INSTRUCTIONS ON RUNNING
In command-line: python kmeans.py
Or open in PyCharm and run kmeans.py

REQUIREMENTS
files optdigits.test and optdigits.train must be in local dir
'''

import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import itertools


class Kmeans:
    '''
    This is the Kmeans classifier.
    '''

    def __init__(self, tng, tst, K=10):
        '''
        This is the Constructor
        :param tng: String; the filename of the training data
        :param tst: String; the filename of the test data
        :param K: Int; the number of clusters desired
        '''
        self.K = K

        self.tng = pd.read_csv(tng, header=None, index_col=64)
        self.tng_labels = self.tng.index.values
        self.tng = self.tng.as_matrix()

        self.tst = pd.read_csv(tst, header=None, index_col=64)
        self.tst_labels = self.tst.index.values
        self.tst = self.tst.as_matrix()

        self.maxval = 16
        self.example_len = self.tng.shape[1]

        self.centroids = np.random.randint(0, self.maxval+1, (self.K, self.example_len))
        self.old_centroids = []

        self.keys = [x for x in xrange(self.K)]

        self.clusters = {}
        self.cluster_labels = np.zeros(self.K)

        self.distances = []
        self.min_dists = []

        self.guesses = []

    def init_clusters(self):
        '''
        This function (re) initializes all the clusters to empty lists.
        :return: None
        '''
        for key in self.keys:
            self.clusters[key] = []

    def get_distances(self, X, Y, centroids=False):
        '''
        This function computes the euclidean distances between the two sets of vectors. If kwarg 'centroids'
        is true, it checks for identical vectors, and skips that computation if encountered.
        :param X: A matrix or set of vectors
        :param Y: A matrix or set of vectors
        :param centroids: flag to turn on checking for, and skipping, identical centroids
        :return: a list of lists of shape (X.shape[0], Y.shape[1]), containing the distances from each
        vector in X to each vector in Y
        '''
        dists = []

        if centroids:
            for x in X:
                dists.append([])
                for y in Y:
                    if all([True if xi==yi else False for xi, yi in zip(x, y)]):
                        continue

                    d = np.linalg.norm(x-y)
                    dists[-1].append(d)

        else:
            for x in X:
                dists.append([])
                for y in Y:
                    d = np.linalg.norm(x-y)
                    dists[-1].append(d)

        return dists

    def assign_to_centroids(self, tng=True):
        '''
        "Assigns" vectors to centroids by dropping the index of a given vector into the appropriate
        dict slot in self.clusters
        :param tng: Flag to indicate whether to use training data or test data
        :return: None
        '''
        if tng:
            src = self.tng
        else:
            src = self.tst
        self.min_dists = []
        self.init_clusters()
        self.distances = self.get_distances(src, self.centroids)

        for example_ind, example_dists in enumerate(self.distances):
            closest_cluster = np.argmin(example_dists)
            self.min_dists.append((closest_cluster, example_dists[closest_cluster]))
            self.clusters[closest_cluster].append(example_ind)

    def update_centroids(self):
        '''
        Reassigns centroids to the mean of the vectors assigned to them.
        :return: None
        '''
        self.old_centroids = np.copy(self.centroids)
        for i in xrange(self.K):
            if len(self.clusters[i]) != 0:
                self.centroids[i] = self.average(self.clusters[i])

    def average(self, member_indices, tng=True):
        '''
        Computes the mean over the vectors indicated by the indices given in the argument
        :param member_indices: a list of indices indicating which vectors in the dataset to use for computation
        :param tng: flag to indicate which dataset to consider
        :return: a vector representing the mean of those vectors indicated by member_indices
        '''
        if tng:
            src = self.tng
        else:
            src = self.tst
        members = [src[i] for i in member_indices]
        avg = np.mean(members, axis=0)
        return avg

    def print_clusters(self):
        for key in self.clusters.keys():
            print key
            print len(self.clusters[key])
            print self.clusters[key]

    def train(self):
        '''
        This is the function that trains the Kmeans classifier by assigning vectors to centroids, and then
        updating centroids to the mean of their clusters in a loop. The loop ends when accuracy stops changing
        (note that accuracy can drop and the loop will continue).
        :return: None
        '''
        count = 0
        acc = 0
        while True:
            self.assign_to_centroids()

            old_acc = acc
            amse, mss, acc = self.get_metrics()

            print "Iteration: ", count
            print "amse: ", amse
            print "mss: ", mss
            print "acc: ", acc

            self.update_centroids()

            if old_acc == acc:
                break

            count += 1

        self.assign_to_centroids()

    def get_metrics(self, tng=True):
        '''
        This function computes average mean squared error, mean squared separation, and accuracy over the clusters
        and the indicated dataset
        :param tng: flag to indicate which dataset should be used in calculations
        :return: amse (float; average mean squared error), mss (float; mean squared separation), acc (float;
        accuracy - with the guessed label defined by the mode of the cluster a given example resides in)
        '''
        if tng:
            src = self.tng_labels
        else:
            src = self.tst_labels
        mses = []
        correct = 0

        for i in xrange(self.K):
            mse = np.sum([x[1] if x[0] == i else 0.0 for x in self.min_dists])/float(len(self.centroids[i]))
            mses.append(mse)

            if len(self.clusters[i]) != 0:
                lbls = [src[j] for j in self.clusters[i]]
                self.cluster_labels[i] = mode(lbls)[0][0]

                correct += np.sum([1 if src[j] == self.cluster_labels[i] else 0 for j in self.clusters[i]])

        amse = np.sum(mses)/float(self.K)

        mss = np.sum(self.get_distances(self.centroids, self.centroids, centroids=True))/(self.K*float((self.K-1))/2)

        acc = float(correct)/float(len(src))

        return amse, mss, acc

    def test(self):
        '''
        This function "classifies" the test data by assigning each vector in the test dataset to its appropriate
        cluster, then deriving the list of "guesses" for each vector in the test set.
        :return: None
        '''
        self.assign_to_centroids(tng=False)

        self.guesses = []
        for i in xrange(len(self.tst)):
            for j in self.keys:
                if i in self.clusters[j]:
                    self.guesses.append(self.cluster_labels[j])

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, "%d" % cm[i, j],
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    def plots(self):
        '''
        This is the wrapper for plotting the confusion matrix.
        :return: None
        '''
        nums = [i for i in xrange(10)]
        pp = PdfPages(str(self.K) + '_' + 'cm.pdf')

        cm = confusion_matrix(self.tst_labels, self.guesses, nums)
        class_names = nums
        np.set_printoptions(precision=2)

        plt.figure(1)
        self.plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix, without normalization')

        pp.savefig()
        pp.close()
        plt.close()

    def vis(self):
        '''
        This function generates 8x8 bitmaps (in the form of pdfs) of each learned centroid.
        :return: None
        '''
        scale = 256.0/16.0
        fig = 2
        for i in self.keys:
            pp = PdfPages(str(self.K) + '_' + str(i) + '.pdf')
            plt.figure(fig)
            plt.subplot(111)
            scaled = np.array([x*scale for x in self.centroids[i]])
            scaled = np.reshape(scaled, (8,8))
            plt.imshow(scaled, cmap='gray', interpolation='nearest')
            pp.savefig()
            pp.close()
            plt.close()
            fig += 1

training_file = 'optdigits.train'
test_file = 'optdigits.test'

#####################################################
#                       K = 10                      #
#####################################################

clfs = []
for iter in xrange(5):
    clf = Kmeans(training_file, test_file)
    clf.train()
    print "Classifier " + str(iter) + " Done Training"
    amse, mss, acc = clf.get_metrics(tng=True)
    clfs.append((amse, mss, acc, clf))


best = clfs[np.argmin([t[0] for t in clfs])]
amse, mss, acc, clf = best

clf.test()
tamse, tmss, tacc = clf.get_metrics(tng=False)
clf.plots()
clf.vis()

with open('K10_best_results.txt', 'w+') as fh:
    fh.write('TRAINING\n')
    fh.write('amse: ' + str(amse) + '\n' + 'mss: ' + str(mss) + '\n' + 'acc: ' + str(acc) + '\n')
    fh.write('TESTING\n')
    fh.write('amse: ' + str(tamse) + '\n' + 'mss: ' + str(tmss) + '\n' + 'acc: ' + str(tacc) + '\n')

#####################################################
#                       K = 30                      #
#####################################################

big_clfs = []
for itr in xrange(5):
    clf = Kmeans(training_file, test_file, K=30)
    clf.train()
    amse, mss, acc = clf.get_metrics(tng=True)
    big_clfs.append((amse, mss, acc, clf))


best = big_clfs[np.argmin([t[0] for t in big_clfs])]
amse, mss, acc, clf = best

clf.test()
tamse, tmss, tacc = clf.get_metrics(tng=False)
clf.plots()
clf.vis()

with open('K30_best_results.txt', 'w+') as fh:
    fh.write('TRAINING\n')
    fh.write('amse: ' + str(amse) + '\n' + 'mss: ' + str(mss) + '\n' + 'acc: ' + str(acc) + '\n')
    fh.write('TESTING\n')
    fh.write('amse: ' + str(tamse) + '\n' + 'mss: ' + str(tmss) + '\n' + 'acc: ' + str(tacc) + '\n')
