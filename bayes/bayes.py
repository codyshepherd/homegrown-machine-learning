import pandas as pd
import sklearn.model_selection
import sklearn.utils
import numpy as np
import math

logs = []
def log(tup):
    logs.append(tup)

def save(filename='bayes.log'):
    with open(filename, "w") as fh:
        fh.write("start of log\n")
        for tup in logs:
            for item in tup:
                fh.write(str(item) + '\n')
        fh.write("end of log\n")

# ===================================================================
# Preparation
# ===================================================================

# read, shuffle, and split data
filename = 'spambase.data'
raw = pd.read_csv(filename, header=None, index_col=57)
data = sklearn.utils.shuffle(raw)
train_set, test_set, train_cls, test_cls = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5, random_state=42)

train_set = train_set.as_matrix()
test_set = test_set.as_matrix()

num_train = float(len(train_set))
num_test = float(len(test_set))
example_len = len(train_set[0])

# compute priors
tng_pct_spm = sum([x==1 for x in train_cls])/num_train
tst_pct_spm = sum([x==1 for x in test_cls])/num_test
prior_p1 = tng_pct_spm
prior_p0 = 1.0-tng_pct_spm

log(('prior p(1)', prior_p1, 'prior p(0)', prior_p0))

mean_vector = []
stdev_vector = []

# compute mean vector
for i in xrange(example_len):
    mean = 0.0
    for example in train_set:
        mean += example[i]
    mean /= float(num_train)
    mean_vector.append(mean)

chk_mean = np.mean(train_set, axis=0, dtype=np.float64)

print mean_vector[:10]
print chk_mean[:10]

# compute stdev vector
for i in xrange(example_len):
    stddev = 0.0
    for example in train_set:
        #stddev += (example[i] - mean_vector[i])**2.0
        stddev += (example[i] - mean_vector[i])**2.0
    stddev /= float(num_train)
    stdev_vector.append(stddev)

chk_std = np.std(train_set, axis=0, dtype=np.float64)

# validate
print "my stdev vector"
print stdev_vector[:10]
print chk_std[:10]
save()
