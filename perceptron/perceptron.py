"""
Cody Shepherd
perceptron.py
CS 545 
Homework 1

This script requires command line arguments passed in.
Usage: python perceptron.py [learning rate] [epochs] [convergence]

learning rate (float): specifies the magnitude of weight updates for incorrect guesses
epochs (int): number of passes through the training data to take during training
convergence (float): the difference between epoch accuracies that will cause the program to halt early

NOTE: this program relies on data preprocessing performed by another file, which should be included with
this: preprocessing.py. The preprocessing work done by that file includes:
    - separating training data from test data
    - separating classifications from each sample
    - normalizing the values in each sample
    - dumping separated, normalized data to .json files

Preprocessing.py must be run in the working directory prior this program.
"""
import json
import itertools
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix
import sys
import timeit

# Check CLI Args
if len(sys.argv) < 3:
    print "Usage: python perceptron.py [learning rate] [epochs] [convergence]"
    sys.exit(1)


class Perceptron:

    def __init__(self, inputs=784, outputs=10, learning_rate=0.01, bias=1, convergence=0.0001):
        """
        Constructor
        KWARGS:
        inputs (int): the number of features in each sample
        outputs (int): the number of possible classes any sample could fall into
        learning_rate (float): the magnitude of weight updates
        bias (int): controls the threshold for "correct" guesses
        convergence (float): the difference in epoch accuracies, under which the program will halt

        RETURNS:
        Perceptron class object

        One of the most important things this constructor does is initialize the matrix of weight
        vectors. When it is done, self.weights will be a matrix (2-D array) of 10 weight vectors, each made
        of 785 features (self.inputs + len(self.bias)) from a uniform distribution over [-0.05, 0.05].
        """
        self.bias = bias
        self.convergence = convergence
        self.weights = np.random.uniform(-0.05, 0.05, (outputs, inputs+1))
        self.learning_rate = learning_rate
        self.logs = []  # for bookeeping
        
    def train(self, samples, classes, test_samples, test_classes, epochs=100):
        """
        ARGS:
        samples (2-D array of floats): shape (num_samples, num_features). This is the array that holds
            the training data
        classes (1-D array of ints): shape (num_samples,). This is the array that holds the correct
            classifications for each respective sample in the samples matrix.
        test_samples (2-D array of floats): same as samples, except this matrix should hold the
            test data.
        test_classes (1-D array of ints): same as classes, except this matrix should hold the correct
            classifications for each respective item in test_samples.

        KWARGS:
        epochs (int): the number of times to loop over the training data

        RETURNS:
        None

        This is the function that serves as the training loop. For each epoch, it loops over
        each sample in the samples set, making a guess and checking. Incorrect guesses are 
        logged and trigger a call to update() to update weights. Correct guesses trigger no 
        further action.

        This function also loops over the samples in the test set during each epoch. Incorrect
        guesses are only logged in support of getting an accuracy rating - they do not trigger 
        weight updates. 

        At the end of each epoch, the difference between the accuracy of the current epoch and
        the previous one will be checked. A difference below the self.convergence member will
        cause the loop to halt and final results to be plotted.

        This function calls the output() function before its termination in order to dump
        a .pdf of the plot of the accuracy results over epochs.
        """
        print "Learning rate: ", self.learning_rate
        print "Epochs: ", epochs
        if len(samples) != len(classes):
            raise Exception('Samples and classes have different shapes')

        done_flag = False
        num_samples = len(samples)
        num_test_samples = len(test_samples)

        for epoch in xrange(epochs):
            num_correct = 0.0
            num_test_correct = 0.0

            # guess & update loop
            for i in xrange(num_samples):
                x_with_bias = np.concatenate(([self.bias], samples[i]))     # attach the bias
                y = [np.dot(weight_vector, x_with_bias) for weight_vector in self.weights]  # generate guesses
                
                answer = np.argmax(y)   # get the guess

                if answer != classes[i]:
                    self.update(answer, classes[i], samples[i], y)  # if the guess was wrong, update weights

            # loop for tallying the number of correct after weight updates
            for i in xrange(num_samples):
                x_with_bias = np.concatenate(([self.bias], samples[i]))
                y = [np.dot(weight_vector, x_with_bias) for weight_vector in self.weights]

                answer = np.argmax(y)

                if answer == classes[i]:
                    num_correct += 1.0
    
            # loop for checking the test set at the end of each epoch
            for i in xrange(num_test_samples):
                test_x_with_bias = np.concatenate(([self.bias], test_samples[i]))
                test_y = [np.dot(weight_vector, test_x_with_bias) for weight_vector in self.weights]

                test_answer = np.argmax(test_y)

                if test_answer == test_classes[i]:
                    num_test_correct += 1.0

            accuracy = float(num_correct) / float(num_samples)
            test_accuracy = float(num_test_correct) / float(num_test_samples)
            self.log(epoch, accuracy, test_accuracy)

            # check for convergence
            if len(self.logs) > 1 and (abs(self.logs[-1][1] - self.logs[-2][1]) < self.convergence):
                print "Convergence reached"
                print "Most recent accuracy: ", self.logs[-1][1]
                print "previous accuracy: ", self.logs[-2][1]
                print "difference: ", self.logs[-1][1] - self.logs[-2][1]
                print "self.convergence: ", self.convergence
                done_flag = True

            print "Epoch :", epoch
            print "Accuracy: ", accuracy
            print "Test Accuracy: ", test_accuracy
            if done_flag:
                break

        # plot results and dump to .pdf
        self.output()

    def update(self, answer, actual, x, y):
        """
        ARGS: 
        answer (int): the guessed class for x
        actual (int): the true class for x
        x (1-D array of floats): shape (784,) the array representing the data sample currently being considered
            by the perceptron
        y (1-D array of ints): shape (10,) the array representing the "guess values" generated by the perceptron for
            each possible class

        RETURNS:
        None

        The purpose of this function is to update the weight vectors according to
        the perceptron learning rule. 

        It generates two sparse vectors: 
            d: all zeros except for a 1 in the place representing the "true" class
            y: all zeros except for 1s in whichever places the perceptron guessed above 0

        These vectors are used to apply the perceptron learning rule across each feature per weight vector.

        This function loops over each weight vector.
        """

        d = np.zeros(len(self.weights[0]))
        d[actual] = 1

        y = np.zeros(len(self.weights[0]))
        y = [(lambda z: 1 if z > 0 else 0)(z) for z in y]

        x_with_bias = np.concatenate([[self.bias], x])

        for i in xrange(len(self.weights)):
            self.weights[i] += (self.learning_rate*(d[i] - y[i])*x_with_bias)

    def test(self, test_samples, test_classes):
        """
        ARGS:
        test_samples: see description of arg of same name in train()
        test_classes: see description of arg of same name in train()

        The purpose of this function is to generate a confusion matrix by running a final
        test of the perceptron's accuracy on the test data.

        It performs a guess-and-check in the same way as the train() function, but only 
        on the test data, and does no updating of weights. 
        """
        def plot_confusion_matrix(cm, classes,
                                  normalize=False,
                                  title='Confusion matrix',
                                  cmap=plt.cm.Blues):
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

        y_pred = np.zeros(len(test_classes))
        y_true = test_classes

        pp = PdfPages('cm_' + str(self.learning_rate) + '_' + str(self.convergence) + '.pdf')

        for i in xrange(len(test_samples)):
            test_x_with_bias = np.concatenate(([self.bias], test_samples[i]))
            test_y = [np.dot(weight_vector, test_x_with_bias) for weight_vector in self.weights]

            y_pred[i] = np.argmax(test_y)

        cm = confusion_matrix(y_true, y_pred, [i for i in xrange(10)])
        class_names = [x for x in xrange(10)]
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names,
                              title='Confusion matrix, without normalization')

        pp.savefig()
        pp.close()


    def log(self, epoch, accuracy, test_accuracy):
        """
        ARGS:
        epoch (int): which epoch is currently being logged
        accuracy (float): the accuracy (a real value on [0, 1]) demonstrated by the perceptron on training
            data for the specified epoch
        test_accuracy (float): a real value, as above, representing the accuracy demonstrated by the
            perceptron on test data for the specified epoch

        RETURNS:
        None

        This function just appends the arguments to a list. It is included as a hedge against more complicated
        logging requirements in the future.
        """
        self.logs.append([epoch, accuracy, test_accuracy])

    def output(self):
        """
        This function generates plots from the self.logs list and saves them to a .pdf.
        """
        pp = PdfPages('plt_' + str(self.learning_rate) + '_' + str(self.convergence) + '.pdf')
        print "plotting..."
        x = [t[0] for t in self.logs]
        y = [t[1] for t in self.logs]
        y_a = [t[2] for t in self.logs]

        plt.plot(x, y, color='blue')
        plt.plot(x, y_a, color='green')

        blue_line = mlines.Line2D([], [], color='blue', label='training')
        green_line = mlines.Line2D([], [], color='green', label='test')
        plt.legend(handles=[blue_line, green_line], loc=4)

        plt.title('Learning rate: ' + str(self.learning_rate) + '  Convergence: ' + str(self.convergence))
        plt.ylabel('% Correct')
        plt.xlabel('Epoch')

        pp.savefig()
        pp.close()


# unpack preprocessed data from .json files
with open('samples.json', 'r+') as fname:
    samples = np.array(json.load(fname))

with open('classes.json', 'r+') as fname:
    classes = np.array(json.load(fname))

with open('test_samples.json', 'r+') as fname:
    test_samples = np.array(json.load(fname))

with open('test_classes.json', 'r+') as fname:
    test_classes = np.array(json.load(fname))

# get CLI args, initialize Perceptron accordingly
lr = float(sys.argv[1])
ep = int(sys.argv[2])
if len(sys.argv) > 3:
    cv = float(sys.argv[3])
    p = Perceptron(learning_rate=lr, convergence=cv)
else:
    p = Perceptron(learning_rate=lr)

print "p: ", p

# train and test
t = timeit.timeit(stmt='p.train(samples, classes, test_samples, test_classes, epochs=ep)', setup='from __main__ import p, samples, classes, test_samples, test_classes, ep', number=1)
t += timeit.timeit(stmt='p.test(test_samples, test_classes)', setup='from __main__ import p, test_samples, test_classes', number=1)
#p.train(samples, classes, test_samples, test_classes, epochs=ep)
#p.test(test_samples, test_classes)
print "final time: ", t
