"""
bayes.py
Cody Shepherd
CS 545 Homework 4
"""

import pandas as pd
import itertools
import sklearn.model_selection
import sklearn.utils
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
import random

errors = 0
logs = []
def logme(tup):
    """
    :param tup: - a tuple of items to be written to the log
    :return: None

    This function is a hedge against needing more complicated functionality.
    """
    logs.append(tup)

def save(filename='bayes.log'):
    """
    This function writes the global logs list to a file
    :param filename: - the filename to which to dump logs
    :return: None
    """
    with open(filename, "w") as fh:
        fh.write("start of log\n")
        for tup in logs:
            for item in tup:
                fh.write(str(item) + '\n')
        fh.write("end of log\n")

def validate(type, v0, v1, dataset, classes,):
    """
    This function is for verifying correct computation of means and stdevs.
    :param type: - one of ['mean', 'stdev']
    :param v0: - vector of feature means/stdevs given label == 0
    :param v1: - vector of feature means/stdevs given label == 1
    :param dataset: - set of original examples on which to compute mean/stdev
    :param classes: -  class labels corresponding to dataset
    :return: check_v0, check_v1 - two vectors which represent the correctly
        calculated mean/stdev given each class label
    """
    ones = []
    zeros = []
    for i in xrange(len(dataset)):
        if classes[i] != 1:
            zeros.append(dataset[i])
        else:
            ones.append(dataset[i])
    if type=='mean':
        #train_0_col_means = X_train.loc[0].describe().loc['mean']
        #train_1_col_means = X_train.loc[1].describe().loc['mean']
        chk_mean_0 = np.mean(zeros, axis=0, dtype=np.float64)
        chk_mean_1 = np.mean(ones, axis=0, dtype=np.float64)

        print "Mean =========="
        print "Homegrown"
        print v0[:10]
        print v1[:10]
        print "Numpy"
        #print train_0_col_means
        #print train_1_col_means
        print chk_mean_0[:10]
        print chk_mean_1[:10]
        return chk_mean_0, chk_mean_1

    if type=='stdev':
        #train_0_col_std = X_train.loc[0].describe().loc['std']
        #train_1_col_std = X_train.loc[1].describe().loc['std']
        chk_dev_0 = np.std(zeros, axis=0, dtype=np.float64)
        chk_dev_1 = np.std(ones, axis=0, dtype=np.float64)

        print "Std Dev =========="
        print "Homegrown"
        print v0[:10]
        print v1[:10]
        print "Numpy"
        #print train_0_col_std
        #print train_1_col_std
        print chk_dev_0[:10]
        print chk_dev_1[:10]
        return chk_dev_0, chk_dev_1

def post_v(x, m, s):
    """
    This Gaussian pdf function calculates the posterior probabilities of each feature
    given a class, sums them, and then applies the log function. This is the version
    that uses numpy's broadcasting functions.
    :param x: the feature vector
    :param m: vector of means given a class for each feature
    :param s: vector of stdevs given a class for each feature
    :return: a real value representing the sum of logs of pdfs.
    """
    ns = np.array(s)
    nm = np.array(m)

    first_term = np.true_divide(1.0, np.multiply(np.sqrt(2 * math.pi), ns))

    second_term = np.exp(-np.true_divide(np.power((np.subtract(x, nm)), 2), (np.multiply(2.0, np.power(ns, 2)))))

    ret = np.multiply(first_term, second_term)

    ret = np.sum(np.log(ret))

    return ret

def post(x, m, s):
    """
    This is the same as post_v(), except it uses homebrewed math on single values, rather than
    numpy broadcast math over vectors.
    :param x: the x value in Gauss
    :param m: the mu value in Gauss
    :param s: the sigma value in Gauss
    :return: a scalar representing the log of the pdf
    """
    #if s == 0.0:
    if s == 0.0 or x == 0.0:
        return 0.0

    #ret = ((1.0/(np.sqrt(2.0*math.pi)*s)) * np.exp( (-1.0) * ( (np.power((x-m), 2.0))/( 2.0*(np.power(s, 2.0)) ) ) ) )
    first_term = 1.0/(np.sqrt(2 * math.pi) * s)
    second_term = np.exp(-(np.power((x-m),2)/(2.0 * np.power(s, 2))))

    ret = first_term * second_term

    if ret != 0.0:
        ret = np.log(ret)
    return ret

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
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

def getARP(preds, actuals):
    """
    This function computes accuracy, recall, and precision given a vector of predictions
    and a vector of actual values.
    :param preds: the vector of guessed classes on {0,1}
    :param actuals: the vector of actual labels
    :return: accuracy, recall, precision
    """
    num_actuals = len(actuals)
    num_preds = len(preds)
    if num_actuals != num_preds:
        raise Exception("Different lengths of args given to getARP()")
    TP = 0.0
    TN = 0.0
    FP = 0.0
    FN = 0.0
    for i in xrange(num_actuals):
        if preds[i] == actuals[i] and preds[i] == 1:
            TP += 1.0
        elif preds[i] == actuals[i] and preds[i] == 0:
            TN += 1.0
        elif preds[i] != actuals[i] and preds[i] == 1:
            FP += 1.0
        elif preds[i] != actuals[i] and preds[i] == 0:
            FN += 1.0
        else:
            raise Exception("?????")

    a = (TP + TN)/float(num_actuals)
    r = TP/(TP + FN)
    p = TP/(TP + FP)

    return a, r, p

# ===================================================================
# Preparation
# ===================================================================

# read, shuffle, and split data
filename = 'spambase.data'
raw = pd.read_csv(filename, header=None, index_col=57)
data = sklearn.utils.shuffle(raw)
X_train, X_test, train_cls, test_cls = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5, random_state=42)

train_set = X_train.as_matrix()
test_set = X_test.as_matrix()

# get commonly-used figures
num_train = len(train_set)
num_test = len(test_set)
example_len = len(train_set[0])
num_tn_0 = sum([x==0 for x in train_cls])
num_tn_1 = sum([x==1 for x in train_cls])


# ===================================================================
# Look for all-null features
# ===================================================================
# This section is used for identifying all-null features across examples
# It just prints results to stdout

null_features_0 = []
null_features_1 = []
for i in xrange(example_len):
    null_feature_0 = True
    null_feature_1 = True
    for j in xrange(len(train_set)):
        if train_set[j][i] != 0 and train_cls[j] == 0:
            null_feature_0 = False
        if train_set[j][i] != 0 and train_cls[j] == 1:
            null_feature_1 = False
    if null_feature_0 == True:
        null_features_0.append(i)
    if null_feature_1 == True:
        null_features_1.append(i)

print "train 0"
print null_features_0
print "train 1"
print null_features_1
null_features_0 = []
null_features_1 = []
for i in xrange(example_len):
    null_feature_0 = True
    null_feature_1 = True
    for j in xrange(len(test_set)):
        if test_set[j][i] != 0 and test_cls[j] == 0:
            null_feature_0 = False
        if test_set[j][i] != 0 and test_cls[j] == 1:
            null_feature_1 = False
    if null_feature_0 == True:
        null_features_0.append(i)
    if null_feature_1 == True:
        null_features_1.append(i)

print "test 0"
print null_features_0
print "test 1"
print null_features_1

# ===================================================================
# Build Model
# ===================================================================
######################## compute priors
tng_pct_spm = num_tn_1/float(num_train)
tst_pct_spm = sum([x==1 for x in test_cls])/float(num_test)
prior_p1 = tng_pct_spm
prior_p0 = 1.0-tng_pct_spm

logme(('prior p(1)', prior_p1, 'prior p(0)', prior_p0))

mean_vector_0 = []
mean_vector_1 = []
stdev_0_vector = []
stdev_1_vector = []

######################## compute mean vectors given each class
for i in xrange(example_len):
    j = 0
    mean_0 = 0.0
    mean_1 = 0.0
    for example in train_set:
        if train_cls[j] == 0:
            mean_0 += example[i]
        else:
            mean_1 += example[i]
        j += 1
    mean_0 /= float(num_tn_0)
    mean_1 /= float(num_tn_1)
    mean_vector_0.append(mean_0)
    mean_vector_1.append(mean_1)

#mean_vector_0, mean_vector_1 = validate('mean', mean_vector_0, mean_vector_1, train_set, train_cls)

####################### compute stdev vectors given each class
for i in xrange(example_len):
    j = 0
    stddev_0 = 0.0
    stddev_1 = 0.0
    for example in train_set:
        if train_cls[j] == 0:
            stddev_0 += ((example[i] - mean_vector_0[i])**2.0)
        else:
            stddev_1 += ((example[i] - mean_vector_1[i])**2.0)
        j += 1
    stddev_0 /= float(num_tn_0-1.0)
    stddev_1 /= float(num_tn_1-1.0)

    stddev_0 = np.sqrt(stddev_0)
    stddev_1 = np.sqrt(stddev_1)

    stdev_0_vector.append(stddev_0)
    stdev_1_vector.append(stddev_1)

#stdev_0_vector, stdev_1_vector = validate('stdev', stdev_0_vector, stdev_1_vector, train_set, train_cls)

##################### get guess for each example
guesses = []
for example in test_set:

    #sum_0 = np.sum([post(x, m, s) for x, m, s in zip(example, mean_vector_0, stdev_0_vector)])
    #sum_1 = np.sum([post(x, m, s) for x, m, s in zip(example, mean_vector_1, stdev_1_vector)])

    sum_0 = post_v(example, mean_vector_0, stdev_0_vector)
    sum_1 = post_v(example, mean_vector_1, stdev_1_vector)

    guess_0 = np.log(prior_p0) + sum_0
    guess_1 = np.log(prior_p1) + sum_1

    guesses.append(0 if guess_0 > guess_1 else 1)

#################### compute accuracy, precision, and recall
accuracy, recall, precision = getARP(guesses, test_cls)

print "Naive Bayes"
print "accuracy: ", accuracy
print "recall: ", recall
print "precision: ", precision
logme(('accuracy', accuracy, 'recall', recall, 'precision', precision))

#################### generate Confusion Matrix
pp = PdfPages('cm.pdf')

cm = confusion_matrix(test_cls, guesses, [i for i in xrange(2)])
class_names = [x for x in xrange(2)]
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cm, classes=class_names, title='Confusion matrix, without normalization')

pp.savefig()
pp.close()

#=====================================
# Logistic Regression
#====================================

clf = LogisticRegression()
clf.fit(train_set, train_cls)
predictions = clf.predict(test_set)

acc, rec, prec = getARP(predictions, test_cls)

print "log reg"
print "accuracy: ", acc
print "recall: ", rec
print "precision: ", prec
logme(('==== linear regression =====',))
logme(('accuracy', acc, 'recall', rec, 'precision', prec))

print "errors: ", errors
save()
