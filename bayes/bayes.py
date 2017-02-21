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

def validate(type, v0, v1, dataset, classes,):
    ones = []
    zeros = []
    for i in xrange(len(dataset)):
        if classes[i] != 1:
            zeros.append(dataset[i])
        else:
            ones.append(dataset[i])
    if type=='mean':
        chk_mean_0 = np.mean(zeros, axis=0, dtype=np.float64)
        chk_mean_1 = np.mean(ones, axis=0, dtype=np.float64)

        print "Mean =========="
        print "Homegrown"
        print v0[:10]
        print v1[:10]
        print "Numpy"
        print chk_mean_0[:10]
        print chk_mean_1[:10]

    if type=='stdev':
        chk_dev_0 = np.std(zeros, axis=0, dtype=np.float64)
        chk_dev_1 = np.std(ones, axis=0, dtype=np.float64)

        print "Std Dev =========="
        print "Homegrown"
        print v0[:10]
        print v1[:10]
        print "Numpy"
        print chk_dev_0[:10]
        print chk_dev_1[:10]

def post(x, m, s):
    if x == 0.0 or m == 0.0 or s == 0.0:
        return 0.0
    ret = (1.0/(math.sqrt(2.0*math.pi)*s)*math.exp( (-1.0) * ( ((x-m)**2.0)/( 2.0*(s**2.0) ) ) ) )
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
    num_actuals = len(actuals)
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

    accuracy = (TP + TN)/float(num_actuals)
    recall = TP/(TP + FN)
    precision = TP/(TP + FP)

    return accuracy, recall, precision

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

num_train = len(train_set)
num_test = len(test_set)
example_len = len(train_set[0])
num_tn_0 = sum([x==0 for x in train_cls])
num_tn_1 = sum([x==1 for x in train_cls])

# compute priors
tng_pct_spm = num_tn_1/float(num_train)
tst_pct_spm = sum([x==1 for x in test_cls])/num_test
prior_p1 = tng_pct_spm
prior_p0 = 1.0-tng_pct_spm

log(('prior p(1)', prior_p1, 'prior p(0)', prior_p0))

mean_vector_0 = []
mean_vector_1 = []
stdev_0_vector = []
stdev_1_vector = []

# compute mean vector
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

#validate('mean', mean_vector_0, mean_vector_1, train_set, train_cls)

# compute stdev vectors given each class
for i in xrange(example_len):
    j = 0
    stddev_0 = 0.0
    stddev_1 = 0.0
    for example in train_set:
        if train_cls[j] == 0:
            stddev_0 += (example[i] - mean_vector_0[i])**2.0
        else:
            stddev_1 += (example[i] - mean_vector_1[i])**2.0
        j += 1
    stddev_0 /= float(num_tn_0-1)
    stddev_1 /= float(num_tn_1-1)

    stddev_0 = math.sqrt(stddev_0)
    stddev_1 = math.sqrt(stddev_1)

    stdev_0_vector.append(stddev_0)
    stdev_1_vector.append(stddev_1)

#validate('stdev', stdev_0_vector, stdev_1_vector, train_set, train_cls)

# get class of each item
guesses = []
for i in xrange(num_test):
    example = test_set[i]
    sum_0 = sum([post(x, m, s) for x, m, s in zip(example, mean_vector_0, stdev_0_vector)])
    sum_1 = sum([post(x, m, s) for x, m, s in zip(example, mean_vector_1, stdev_1_vector)])

    guess_0 = np.log(prior_p0) + sum_0
    guess_1 = np.log(prior_p1) + sum_1

    if guess_0 > guess_1:
        guess = 0
    else:
        guess = 1

    guesses.append(guess)

# compute accuracy, precision, and recall
accuracy, recall, precision = getARP(guesses, test_cls)

print "accuracy: ", accuracy
print "recall: ", recall
print "precision: ", precision
log(('accuracy', accuracy, 'recall', recall, 'precision', precision))

pp = PdfPages('cm.pdf')

cm = confusion_matrix(test_cls, guesses, [i for i in xrange(2)])
class_names = [x for x in xrange(2)]
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
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
log(('==== linear regression =====',))
log(('accuracy', acc, 'recall', rec, 'precision', prec))


save()
