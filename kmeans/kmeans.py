import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import collections

class Kmeans:

    def __init__(self, tng, tst, K=10):
        self.K = K

        self.tng = pd.read_csv(tng, header=None, index_col=64)
        self.tng_labels = self.tng.index.values
        self.tng = self.tng.as_matrix()

        self.tst = pd.read_csv(tst, header=None, index_col=64)
        self.tst_labels = self.tst.index.values
        self.tst = self.tst.as_matrix()

        self.maxval = 16
        self.example_len = self.tng.shape[1]

        '''
        indices = np.arange(self.maxval)
        np.random.shuffle(indices)
        self.centroids = self.tng.iloc[indices[:10]].as_matrix()
        '''

        self.centroids = np.random.randint(0, self.maxval+1, (self.K, self.example_len))

        self.keys = [x for x in xrange(self.K)]

        self.clusters = {}


    def __str__(self):
        return 'TRAINING\n' + self.tng.to_string() + '\nTEST\n' + self.tst.to_string()

    def init_clusters(self):
        #self.clusters = collections.defaultdict(list)
        for key in self.keys:
            self.clusters[key] = []


    def get_distances(self, X, Y, centroids=False):
        dists = []

        if centroids==True:
            for x in X:
                dists.append([])
                for y in Y:
                    if all([True if xi==yi else False for xi, yi in zip(x, y)]):
                        continue

                    #d = np.sqrt(np.sum([(xi-yi)**2 for xi, yi in zip(x, y)]))
                    #d = np.sqrt(np.sum((x-y)**2,axis=0))
                    d = np.linalg.norm(x-y)
                    dists[-1].append(d)

        else:
            for x in X:
                dists.append([])
                for y in Y:
                    d = np.linalg.norm(x-y)
                    #d = np.sqrt(np.sum([(x-y)**2 for x, y in zip(example, centroid)]))
                    dists[-1].append(d)

        return dists

    def assign_to_centroids(self):
        self.min_dists = []
        self.init_clusters()
        self.distances = self.get_distances(self.tng, self.centroids)

        for example_ind, example_dists in enumerate(self.distances):
            closest_cluster = np.argmin(example_dists)
            self.min_dists.append((closest_cluster, example_dists[closest_cluster]))
            self.clusters[closest_cluster].append(example_ind)

    def update_centroids(self):
        self.old_centroids = np.copy(self.centroids)
        for i in xrange(self.K):
            if len(self.clusters[i]) != 0:
                self.centroids[i] = self.average(self.clusters[i])

    def average(self, member_indices):
        members = [self.tng[i] for i in member_indices]
        avg = np.mean(members, axis=0)
        return avg

    def print_clusters(self):
        for key in self.clusters.keys():
            print key
            print len(self.clusters[key])
            print self.clusters[key]

    def train(self):
        count = 0
        while True:
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

            count += 1

    def get_metrics(self, tng=True):
        if tng==True:
            src = self.tng_labels
        else:
            src = self.tst_labels
        mses = []
        correct = 0

        for i in xrange(self.K):
            mse = np.sum([x[1] if x[0]==i else 0.0 for x in self.min_dists])/float(len(self.centroids[i]))
            mses.append(mse)



        amse = np.sum(mses)/float(self.K)

        mss = np.sum(self.get_distances(self.centroids, self.centroids, centroids=True))/(self.K*float((self.K-1))/2)

        acc = float(correct)/float(len(src))
        print "num correct: ", correct
        print "num examples: ", len(src)

        return amse, mss, acc

    def predict(self):
        pass

training_file = 'optdigits.train'
test_file = 'optdigits.test'

clf = Kmeans(training_file, test_file)

clf.train()
#clf.print_clusters()

