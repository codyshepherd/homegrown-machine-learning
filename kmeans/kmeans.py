import pandas as pd
import numpy as np

class Kmeans:

    def __init__(self, tng, tst, K=10):
        self.K = K
        self.tng = pd.read_csv(tng, header=None, index_col=64)
        self.tst = pd.read_csv(tst, header=None, index_col=64)
        self.maxval = max(self.tng.values.max(), self.tst.values.max())
        self.example_len = self.tng.shape[1]
        self.centroids = np.random.randint(0,self.maxval,(self.K,self.example_len))

    def __str__(self):
        return 'TRAINING\n' + self.tng.to_string() + '\nTEST\n' + self.tst.to_string()

    def train(self):
        # initialize K random centroids within bounds
        pass

    def predict(self):
        pass

training_file = 'optdigits.train'
test_file = 'optdigits.test'

clf = Kmeans(training_file, test_file)

