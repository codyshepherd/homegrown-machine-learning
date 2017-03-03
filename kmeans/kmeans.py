import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import collections

class Kmeans:

    short_fns = {
        'distance': lambda x, y: np.sqrt((x[0] - y[0])**2 + (x[1] - y[1])**2)
    }

    def __init__(self, tng, tst, K=10):
        self.K = K
        self.tng = pd.read_csv(tng, header=None, index_col=64)
        self.tst = pd.read_csv(tst, header=None, index_col=64)
        #self.maxval = max(self.tng.values.max(), self.tst.values.max())
        self.maxval = len(self.tng)
        self.example_len = self.tng.shape[1]
        #self.centroids = np.random.randint(0,self.maxval,(self.K,self.example_len))
        indices = np.arange(self.maxval)
        np.random.shuffle(indices)
        self.centroids = self.tng.iloc[indices[:10]].as_matrix()

        self.keys = [x for x in xrange(self.K)]
        #self.clusters = dict.fromkeys(self.keys,[])
        self.clusters = collections.defaultdict(list)

    def __str__(self):
        return 'TRAINING\n' + self.tng.to_string() + '\nTEST\n' + self.tst.to_string()

    def assign_to_centroids(self):
        self.distances = []
        for example in self.tng.as_matrix():
            self.distances.append([])
            for centroid in self.centroids:
                d = np.linalg.norm(example-centroid)
                self.distances[-1].append(d)

        for i, example in enumerate(self.distances):
            mx = np.argmax(example)
            self.clusters[mx].append(i)

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
        measure = 1.0
        count = 1
        while measure > 0:
            self.assign_to_centroids()
            self.update_centroids()
            diffs = []
            for i in xrange(self.K):
                diffs.append(np.linalg.norm(self.old_centroids[i]-self.centroids[i]))
            measure = np.mean(diffs)
            print "iteration: ", count
            print "mean of diffs"
            print measure
            count += 1

    def predict(self):
        pass

training_file = 'optdigits.train'
test_file = 'optdigits.test'

clf = Kmeans(training_file, test_file)

clf.train()
clf.print_clusters()

