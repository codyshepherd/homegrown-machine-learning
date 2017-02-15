"""
'processing.py'
Cody Shepherd
CS 545: Machine Learning
Homwork 3
"""
import pandas as pd
import sklearn.preprocessing
import sklearn.utils
import sklearn.model_selection
import sklearn.svm
from sklearn.metrics import roc_curve
import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import random

# Helper function for doing feature selection
def features_only(features, examples):
    """
    ARGS:
    features - a list of integers representing indices of features to be selected.
    examples - a list of example instances on which feature selection will be performed.

    RETURNS:
    lst - an altered copy of 'examples' arg, with each example having only the features
        indicated by 'features' arg.

    This function makes copies of each example from the examples argumentcontaining only 
    those features specified by the features argument. 
    """
    lst = []
    for example in examples:
        new_example = [example[i] for i in features]
        lst.append(new_example)

    return lst

# ===================================================================
# Preparation
# ===================================================================

# read, shuffle, and split data
filename = 'spambase.data'
raw = pd.read_csv(filename, header=None, index_col=57)
data = sklearn.utils.shuffle(raw)
train_set, test_set, train_cls, test_cls = sklearn.model_selection.train_test_split(data, data.index.values, test_size=0.5, random_state=42)

num_train = len(train_set)
num_test = float(len(test_set))
print "num_test: ", num_test

# fit scaler, and scale both sets
scaler = sklearn.preprocessing.StandardScaler().fit(train_set)
train_set = scaler.transform(train_set)
test_set = scaler.transform(test_set)

# ===================================================================
# Experiment 1: Full-feature testing and ROC Curve
# ===================================================================

# fit SVM and make predictions
clf = sklearn.svm.SVC(kernel='linear')
clf = clf.fit(train_set, train_cls)
predictions = clf.predict(test_set)

# get svm outputs pre-signum
distances = clf.decision_function(test_set)

# compute accuracy, precision, and recall
TP = 0.0
TN = 0.0
FP = 0.0
FN = 0.0
for i in xrange(len(predictions)):
    if predictions[i] == test_cls[i] and predictions[i] == 1:
        TP += 1.0
    elif predictions[i] == test_cls[i] and predictions[i] == 0:
        TN += 1.0
    elif predictions[i] != test_cls[i] and predictions[i] == 1:
        FP += 1.0
    elif predictions[i] != test_cls[i] and predictions[i] == 0:
        FN += 1.0
    else:
        raise Exception("?????")

accuracy = (TP + TN)/float(len(test_cls))
recall = TP/(TP + FN)
precision = TP/(TP + FP)

# write results to file
with open("svm.log", "w+") as fh:
    fh.write("accuracy: " + str(accuracy) + '\n')
    fh.write("recall: " + str(recall) + '\n')
    fh.write("precision: " + str(precision) + '\n')

# compute ROC curve and save to file
fprs, tprs, thrsh = roc_curve(test_cls, distances)

pp1 = PdfPages('roc_curve.pdf')
print "plotting roc..."

plt.figure(1)
plt.plot(fprs, tprs, color='blue')

plt.title('Experiment 1: ROC Curve ' + str(len(thrsh)) + ' thresholds')
plt.xlabel('FPR')
plt.ylabel('TPR')

pp1.savefig()
pp1.close()

# ===================================================================
# Experiment 2: Feature Selection
# ===================================================================

# get weight vector of previously trained model
weights = clf.coef_[0]

# zip weights with indices and sort
indices = [x for x in xrange(len(weights))]
zipped = zip(indices, weights)
sorted_weights = sorted(zipped, key = lambda x: x[1])

with open("svm.log", "a") as fh:
    fh.write("Indices of highest weighted features: \n")
    for i in xrange(5):
        fh.write(str(sorted_weights[i][0]) + '\n')

# loop over m values; train, test, and record accuracy
results = []
for m in xrange(2, 58, 1):
    indices = []
    for i in xrange(m):
        indices.append(sorted_weights[i][0])

    new_train_set = features_only(indices, train_set)
    new_test_set = features_only(indices, test_set)

    new_clf = sklearn.svm.SVC(kernel='linear')
    new_clf = new_clf.fit(new_train_set, train_cls)

    new_preds = new_clf.predict(new_test_set)
    correct = (new_preds == test_cls).sum()
    accuracy = float(correct)/float(len(new_test_set))

    results.append((m,accuracy))

# plot results
pp2 = PdfPages('feature_selection.pdf')
print "plotting feature selection..."

plt.figure(2)
plt.plot([x[0] for x in results], [x[1] for x in results], color='blue')

plt.title('Experiment 2: Feature Selection')
plt.xlabel('Number of features')
plt.ylabel('Accuracy')

pp2.savefig()
pp2.close()

# ===================================================================
# Experiment 3: Random Feature Selection
# ===================================================================

# loop over m values, selecting random features each time; train, test, and compute accuracy
rand_results = []
for m in xrange(2, 58, 1):
    indices = random.sample(range(57), m)

    new_train_set = features_only(indices, train_set)
    new_test_set = features_only(indices, test_set)

    new_clf = sklearn.svm.SVC(kernel='linear')
    new_clf = new_clf.fit(new_train_set, train_cls)

    new_preds = new_clf.predict(new_test_set)
    correct = (new_preds == test_cls).sum()
    accuracy = float(correct)/float(len(new_test_set))

    rand_results.append((m,accuracy))

# plot results
pp3 = PdfPages('random_feature_selection.pdf')
print "plotting random feature selection..."

plt.figure(3)
plt.plot([x[0] for x in rand_results], [x[1] for x in rand_results], color='blue')

plt.title('Experiment 3: Random Feature Selection')
plt.xlabel('Number of features')
plt.ylabel('Accuracy')

pp3.savefig()
pp3.close()


print "Done"
