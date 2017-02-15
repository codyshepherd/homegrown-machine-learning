import json
import itertools
import numpy as np
import csv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.lines as mlines
from sklearn.metrics import confusion_matrix
import sys


class MLP:
    
    def __init__(self, hidden, inputs=784, outputs=10, learning_rate=0.1, bias=1, convergence=0.0001, activation='sigmoid', alpha=0.9):
        """
        The constructor initializes possible activation functions ( I was never able to get reLU to work), 
        and initializes the weights of all the hidden layers. 

        This constructor should be able to handle an arbitrary number of hidden layers, each with an 
        arbitrary number of neurons.

        This constructor also initializes the "delts" from epoch-1, i.e. the changes made on the previous 
        epoch, to which the alpha will be applied. For the zeroth epoch, these delts are initialized to zeros.
        """
        self.correct_target = 0.9
        self.incorrect_target = 0.1
        self.num_hidden_layers = len(hidden)
        self.inputs = inputs
        self.outputs = outputs
        self.learning_rate = learning_rate
        self.bias = bias
        self.convergence = convergence
        self.alpha = alpha
        
        self.activation = {
            'sigmoid': lambda x: [1/(1 + np.exp(-1*x_i)) for x_i in x],
            'relu': lambda x: [0.0 if x_i < 0 else x_i for x_i in x]
        }[activation]

        self.hidden = hidden # tuple of ints indicating number of neurons in hidden layer

        self.weights = []


        self.weights.append(np.random.uniform(-0.5, 0.5, (hidden[0]+1, inputs+1)))
        if self.num_hidden_layers > 1:
            for i in xrange(self.num_hidden_layers-1):
                self.weights.append(np.random.uniform(-0.5, 0.5, (hidden[i+1]+1, len(self.weights[i]))))
        self.weights.append(np.random.uniform(-0.5, 0.5, (outputs, hidden[-1]+1)))

        self.logs = []

        self.delts = {}
        self.delts[str(0)] = np.zeros((hidden[0]+1, inputs+1))
        if self.num_hidden_layers > 1:
            for i in xrange(self.num_hidden_layers-1):
                self.delts[str(i+1)] = np.zeros((hidden[i+1]+1, len(self.delts[str(i)])))
        self.delts[str(self.num_hidden_layers)] = np.zeros((outputs, hidden[-1]+1))

    def train(self, samples, classes, test_samples, test_classes, epochs=50):
        """
        This is the primary training method.

        It loops over all epochs, first doing a guess->backprop loop over all training
        examples using SGD, then a guess->tally over all training examples without backprop, 
        then guess->tally over all test examples.

        It loops for a minimum of the number of epochs given in the argument. After that it
        checks for convergence.

        Before it exits, this function calls MLP.output()
        """

        print "Hidden layers: ", [i for i in self.hidden]
        print "Learning rate: ", self.learning_rate
        print "Epochs: ", epochs
        if len(samples) != len(classes):
            raise Exception('Samples and classes have different shapes')

        epochs_done_flag = False
        current_epoch = 0
        num_samples = len(samples)
        num_test_samples = len(test_samples)

        #for epoch in xrange(epochs):
        while epochs_done_flag == False:
            num_correct = 0.0
            num_test_correct = 0.0

            # guess & update loop (don't execute for epoch 0)
            if current_epoch != 0:
                for i in xrange(num_samples):
                    y = []

                    x_with_bias = np.concatenate(([self.bias], samples[i]))     # attach the bias

                    y_i = self.activation([np.dot(weight_vector, x_with_bias) for weight_vector in self.weights[0]])
                    y.append(y_i)

                    for i in xrange(self.num_hidden_layers-1):
                        y_i = self.activation([np.dot(weight_vector, y[i-1]) for weight_vector in self.weights[i+1]])
                        y.append(y_i)

                    y_i = self.activation([np.dot(weight_vector, y[-1]) for weight_vector in self.weights[-1]])
                    y.append(y_i)

                    self.backprop(x_with_bias, y, classes[i])

            # tally loop with logging
            for i in xrange(num_samples):
                y = []

                x_with_bias = np.concatenate(([self.bias], samples[i]))     # attach the bias

                y_i = self.activation([np.dot(weight_vector, x_with_bias) for weight_vector in self.weights[0]])
                y.append(y_i)

                for i in xrange(self.num_hidden_layers-1):
                    y_i = self.activation([np.dot(weight_vector, y[i-1]) for weight_vector in self.weights[i+1]])
                    y.append(y_i)

                y_i = self.activation([np.dot(weight_vector, y[-1]) for weight_vector in self.weights[-1]])
                y.append(y_i)

                answer = np.argmax(y[-1])
                if answer == classes[i]:
                    num_correct += 1

            # test data loop
            for i in xrange(num_test_samples):
                y = []

                x_with_bias = np.concatenate(([self.bias], test_samples[i]))     # attach the bias

                y_i = self.activation([np.dot(weight_vector, x_with_bias) for weight_vector in self.weights[0]])
                y.append(y_i)

                for i in xrange(self.num_hidden_layers-1):
                    y_i = self.activation([np.dot(weight_vector, y[i-1]) for weight_vector in self.weights[i+1]])
                    y.append(y_i)

                y_i = self.activation([np.dot(weight_vector, y[-1]) for weight_vector in self.weights[-1]])
                y.append(y_i)

                answer = np.argmax(y[-1])
                if answer == test_classes[i]:
                    num_test_correct += 1

            # calculate results
            accuracy = float(num_correct) / float(num_samples)
            test_accuracy = float(num_test_correct) / float(num_test_samples)
            self.log(current_epoch, accuracy, test_accuracy)
           
            print "Epoch: ", current_epoch
            print "Accuracy: ", accuracy
            print "Test Accuracy: ", test_accuracy

            # check for convergence
            if current_epoch >= epochs and (abs(self.logs[-1][1] - self.logs[-2][1]) < self.convergence):
                print "Convergence reached"
                print "self.convergence: ", self.convergence
                epochs_done_flag = True

                if len(self.logs) > 1:
                    print "Most recent accuracy: ", self.logs[-1][1]
                    print "previous accuracy: ", self.logs[-2][1]
                    print "difference: ", self.logs[-1][1] - self.logs[-2][1]
                else:
                    print "No logs!"
            else:
                current_epoch += 1
                
        # call routine to dump logs to pdfs
        self.output()


    def backprop(self,x, y, target_class):
        """
        This is the backpropagation function. It adjusts the weights of the neural net's
        hidden layers according to, surprise, the backpropagation algorithm. i.e. it 
        calculates the error at each layer, then updates the weights of each layer:
        w_k = w_k + delta-w_k
            where delta-w_k = eta*error*activation + alpha*prior_delta

        more or less. I'm not going to get crazy with notation.

        NOTE: this function is currently hard-coded to handle only one hidden layer. it cannot
        manage more or less than that.
        """
        o = y[-1]

        # d_out: 10 real values
        d_out = [(lambda o_k, i: \
                                o_k*(1 - o_k)*(self.correct_target - o_k) if i==target_class else \
                                o_k*(1 - o_k)*(self.incorrect_target - o_k))(item, o.index(item)) for item in o]
        d_out = np.array(d_out, dtype=np.float32).reshape(10,1)

        d_hidden = [(lambda h_k, w_kj: \
                                h_k*(1-h_k)*np.dot(w_kj, d_out))(item, w) for item, w in zip(y[-2], \
                                np.array(self.weights[-1], dtype=np.float32).reshape(self.hidden[-1]+1, 10))]
        d_hidden = np.array(d_hidden, dtype=np.float32).reshape(len(d_hidden), 1)
                                    
        #TODO: Extend to handle arbitrary number of hidden layers

        delt_w_kj = self.learning_rate*np.matmul(d_out, np.array(y[-2]).reshape(1,self.hidden[-1]+1))

        kj_key = str(self.num_hidden_layers)
        self.weights[-1] += ( delt_w_kj + self.alpha*self.delts[kj_key] )

        self.delts[kj_key] = delt_w_kj


        delt_w_ji = self.learning_rate*np.matmul(d_hidden, x.reshape(1,785))

        ji_key = str(self.num_hidden_layers-1)
        self.weights[-2] += ( delt_w_ji + self.alpha*self.delts[ji_key] )

        self.delts[ji_key] = delt_w_ji

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
        global nm
        pp = PdfPages('plt_' + str(self.hidden) + '_' + str(self.alpha) + '_' + str(nm) + '.pdf')
        print "plotting..."
        x = [t[0] for t in self.logs]
        y = [t[1] for t in self.logs]
        y_a = [t[2] for t in self.logs]

        plt.plot(x, y, color='blue')
        plt.plot(x, y_a, color='green')

        blue_line = mlines.Line2D([], [], color='blue', label='training')
        green_line = mlines.Line2D([], [], color='green', label='test')
        plt.legend(handles=[blue_line, green_line], loc=4)

        plt.title('Hidden neurons: ' + str(self.hidden) + '  Momentum: ' + str(self.alpha) + ' Tng Examples: ' + str(nm))
        plt.ylabel('Fraction Correct')
        plt.xlabel('Epoch')

        pp.savefig()
        pp.close()

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

        num_test_samples = len(test_samples)
        y_pred = np.zeros(len(test_classes))
        y_true = test_classes

        global nm
        pp = PdfPages('cm_' + str(self.hidden) + '_' + str(self.alpha) + '_' + str(nm) + '.pdf')

        # test data loop
        for i in xrange(num_test_samples):
            y = []

            x_with_bias = np.concatenate(([self.bias], test_samples[i]))     # attach the bias

            y_i = self.activation([np.dot(weight_vector, x_with_bias) for weight_vector in self.weights[0]])
            y.append(y_i)

            for i in xrange(self.num_hidden_layers-1):
                y_i = self.activation([np.dot(weight_vector, y[i-1]) for weight_vector in self.weights[i+1]])
                y.append(y_i)

            y_i = self.activation([np.dot(weight_vector, y[-1]) for weight_vector in self.weights[-1]])
            y.append(y_i)

            y_pred[i] = np.argmax(y[-1])


        cm = confusion_matrix(y_true, y_pred, [i for i in xrange(10)])
        class_names = [x for x in xrange(10)]
        np.set_printoptions(precision=2)

        # Plot non-normalized confusion matrix
        plt.figure()
        plot_confusion_matrix(cm, classes=class_names,
                              title='Confusion matrix, without normalization')

        pp.savefig()
        pp.close()



# Check CLI Args
if len(sys.argv) < 9:
    print "Usage: python perceptron.py [Hidden neurons] [learning rate] [epochs] [convergence] [alpha] [output classes] [activation] [num_samples]"
    sys.exit(1)


# unpack preprocessed data from .json files
with open('samples.json', 'r+') as fname:
    samples = np.array(json.load(fname))

with open('classes.json', 'r+') as fname:
    classes = np.array(json.load(fname))

with open('test_samples.json', 'r+') as fname:
    test_samples = np.array(json.load(fname))

with open('test_classes.json', 'r+') as fname:
    test_classes = np.array(json.load(fname))

hid = map(int, sys.argv[1].split(','))
lr = float(sys.argv[2])
ep = int(sys.argv[3])
cv = float(sys.argv[4])
al = float(sys.argv[5])
out = int(sys.argv[6])
act = str(sys.argv[7])
nm = int(sys.argv[8])

inp = len(samples[0])

m = MLP(hid, inputs=inp, outputs=out, learning_rate=lr, convergence=cv, activation=act, alpha=al)

m.train(samples[:nm], classes[:nm], test_samples, test_classes, epochs=ep)
m.test(test_samples, test_classes)
