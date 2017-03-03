import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import collections

class Kmeans:

    def __init__(self, tng, tst, K=10):
        self.K = K
        self.tng = pd.read_csv(tng, header=None, index_col=64)
        self.tst = pd.read_csv(tst, header=None, index_col=64)

        self.maxval = len(self.tng)
        self.example_len = self.tng.shape[1]

        indices = np.arange(self.maxval)
        np.random.shuffle(indices)
        self.centroids = self.tng.iloc[indices[:10]].as_matrix()

        self.keys = [x for x in xrange(self.K)]
        self.clusters = collections.defaultdict(list)

        self.min_dists = []

    def __str__(self):
        return 'TRAINING\n' + self.tng.to_string() + '\nTEST\n' + self.tst.to_string()

    def get_distances(self, X, Y):
        dists = []
        for x in X:
            dists.append([])
            for y in Y:
                if all([True if xi==yi else False for xi, yi in zip(x, y)]):
                    continue

                d = np.sqrt(np.sum([(xi-yi)**2 for xi, yi in zip(x, y)]))
                dists[-1].append(d)

        return dists

    def assign_to_centroids(self):
        #self.distances = []
        self.distances = self.get_distances(self.tng.as_matrix(), self.centroids)

        '''
        for example in self.tng.as_matrix():
            self.distances.append([])
            for centroid in self.centroids:
                #d = np.linalg.norm(example-centroid)
                d = np.sqrt(np.sum([(x-y)**2 for x, y in zip(example, centroid)]))
                self.distances[-1].append(d)
        '''

        for i, example in enumerate(self.distances):
            mn = np.argmin(example)
            self.min_dists.append((mn, example[mn]))
            self.clusters[mn].append(i)

    def update_centroids(self):
        self.old_centroids = np.copy(self.centroids)
        for i in xrange(self.K):
            if self.clusters[i]:
                self.centroids[i] = self.average(self.clusters[i])

    def average(self, member_indices):
        members = [self.tng.iloc[i] for i in member_indices]
        avg = np.mean(members, axis=0)
        return avg

    def print_clusters(self):
        for key in self.clusters.keys():
            print key
            print self.clusters[key]

    def train(self):
        threshold = 10.0
        count = 0
        while threshold > 0:
            self.assign_to_centroids()

            amse, mss, acc = self.get_metrics()

            print "Iteration: ", count
            print "amse: ", amse
            print "mss: ", mss
            print "acc: ", acc

            self.update_centroids()

            diffs = []
            for i in xrange(self.K):
                #diffs.append(np.linalg.norm(self.old_centroids[i]-self.centroids[i]))
                diffs.append(np.sqrt(np.sum([(x-y)**2 for x, y in zip(self.old_centroids[i], self.centroids[i])])))
            threshold = np.mean(diffs)
            print "threshold: ", threshold

            count += 1

    def get_metrics(self):
        mses = []
        correct = 0
        vs = list(self.tng.index.values)
        print "len vs: ", len(vs)
        raw_input()
        lens = 0
        for cls in self.clusters:
            lens += len(self.clusters[cls])
        print "total length of items in centroids:"
        print lens
        raw_input()
        for i in xrange(self.K):
            mse = np.sum([x[1] if x[0]==i else 0.0 for x in self.min_dists])/float(len(self.centroids[i]))
            mses.append(mse)

            #TODO: Problem with computation of num correct
            correct += np.sum([1 if vs[x]==i else 0 for x in self.clusters[i]])

        amse = np.sum(mses)/float(self.K)

        mss = np.sum(self.get_distances(self.centroids, self.centroids))/(self.K*float((self.K-1))/2)

        acc = float(correct)/float(len(self.tng))
        print "num correct: ", correct
        print "num examples: ", len(self.tng)

        return amse, mss, acc

    def predict(self):
        pass

training_file = 'optdigits.train'
test_file = 'optdigits.test'

clf = Kmeans(training_file, test_file)

clf.train()
#clf.print_clusters()

