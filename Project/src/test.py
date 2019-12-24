import numpy as np
import libsvm
import random

class Perceptron(object):

    # Initialization of w, b and other hyper-parameters
    # happens in the train methods of respective perceptron variants

    def __init__(self, num_features):
        self.num_features = num_features

    def predict(self, x, w, b):
        predictions = np.array([])

        if np.shape(x) == (1, self.num_features):
            dot_product = x.dot(np.transpose(w))
            if dot_product + b >= 0:
                predictions = np.append(predictions, 1)
            else:
                predictions = np.append(predictions, 0)
        else:
            for example in x:
                dot_product = example.dot(np.transpose(w))
                if dot_product + b >=0:
                    predictions = np.append(predictions, 1)
                else:
                    predictions = np.append(predictions, 0)

        # if np.shape(X) == (2,):
        #     X = np.array([X])
        #
        # for example in X:
        #     if np.dot(np.transpose(w), example) + b >= 0:
        #         predictions = np.append(predictions, 1)
        #     else:
        #         predictions = np.append(predictions, -1)
        return predictions

    def accuracy(self, x, y, w, b):
        predictions = self.predict(x, w, b)
        accuracy = np.sum(predictions == y) / len(predictions)
        return accuracy

    def get_num_predictions(self, x, y, w, b):
        predictions = self.predict(x, w, b)
        return np.sum(predictions == y)

    def update(self,x, y, w, b, lr):
        # FILL IN
        if y == 0:
            y = -1
        w_new = w + lr * (y * x)
        b_new = b + lr * y
        return (w_new, b_new)

    def shuffle_arrays(self, x, y):
        idx = np.arange(x.shape[0])
        np.random.shuffle(idx)
        return x[idx], y[idx]

    # SIMPLE PERCEPTRON

    def simple_perceptron_train(self, X_train, y_train, epochs=10, lr=0.01):
        # w = np.random.uniform(0, 1, size=X_train.shape[1])  # initialize w
        # b = 0  # initialize bias

        np.random.seed(0)
        random_number = np.random.uniform(-0.01, 0.01)
        w = np.full((1, self.num_features), random_number)
        b = random_number
        #update_counter = 0
        for i in range(0, epochs):
            X_train, y_train = self.shuffle_arrays(X_train, y_train)
            for j in range(0, len(y_train)):
                prediction = self.predict(X_train[j], w, b)
                if prediction[0] != y_train[j]:
                    #update_counter += 1
                    w, b = self.update(X_train[j], y_train[j], w, b, lr)
                else:
                    continue
        #print("Total updates =",update_counter)
        return w, b

    def averaged_perceptron_train(self, X_train, y_train, epochs=10, lr=0.01):
        # w = np.random.uniform(0, 1, size=X_train.shape[1])  # initialize w
        # b = 0  # initialize bias

        #np.random.seed(0)
        #random_number = np.random.uniform(-0.01, 0.01)
        w = np.full((1, self.num_features), 0)
        a = np.full((1, self.num_features), 0)
        b = 0
        b_a = 0
        update_counter = 0
        counter = 0
        for i in range(0, epochs):
            X_train, y_train = self.shuffle_arrays(X_train, y_train)
            for j in range(0, len(y_train)):
                prediction = self.predict(X_train[j], w, b)
                if prediction[0] != y_train[j]:
                    update_counter += 1
                    w, b = self.update(X_train[j], y_train[j], w, b, lr)
                    counter += 1
                    a = a + w
                    b_a = b_a + b
                else:
                    counter += 1
                    a = a + w
                    b_a = b_a + b
                    continue

        # print("Total updates =",update_counter)
        return a/counter, b_a/counter


def train_test():
    X_train, y_train, num_features = libsvm.read_libsvm(
        "C:\\Users\\Abhi\\Documents\\MyFiles\\AB\\GradSchool\\Fall19\\ML\\Project"
        "\\data\\data\\data-splits\\data.train")
    # log transforming
    X_train = X_train.log1p()
    # X_train_array = np.array([])
    # print("Number of features =", num_features)

    # adding all sparse matrices to a numpy array
    # for x in X_train:
        # X_train_array = np.append(X_train_array, x)

    simple_perceptron = Perceptron(num_features)
    decay_perceptron = Perceptron(num_features)
    average_perceptron = Perceptron(num_features)
    pocket_perceptron = Perceptron(num_features)
    margin_perceptron = Perceptron(num_features)

    learning_rates = [0.01, 0.1, 1]
    margins = [1, 0.1, 0.01]

    print("Accuracies on training set for 20 epochs")
    print("--" * 50)

    print("Simple Perceptron")
    print("--" * 50)
    for r in learning_rates:
        w, b = simple_perceptron.simple_perceptron_train(X_train, y_train,
                                                         epochs=20, lr=r)
        print("Simple perceptron accuracy on training set for lr =", r, ":",
              simple_perceptron.accuracy(X_train, y_train, w, b) * 100)
    # print()
    # print("Decay Perceptron")
    # print("--" * 50)
    # for r in learning_rates:
    #     w, b = decay_perceptron.decaying_perceptron_train(X_train_array, y_train, epochs=20, lr=r)
    #     print("Decay perceptron accuracy on training set for lr =", r, ":",
    #           decay_perceptron.accuracy(X_train_array, y_train, w, b) * 100)
    print()
    print("Averaged Perceptron")
    print("--" * 50)
    for r in learning_rates:
        w, b = average_perceptron.averaged_perceptron_train(X_train, y_train, epochs=20, lr=r)
        print("Average perceptron accuracy on training set for lr =", r, ":",
              average_perceptron.accuracy(X_train, y_train, w, b) * 100)
    # print()
    # print("Margin Perceptron")
    # print("--" * 50)
    # for r in learning_rates:
    #     for mu in margins:
    #         w, b = margin_perceptron.margin_perceptron_train(X_train_array, y_train, mu, epochs=20, lr=r)
    #         print("Margin perceptron accuracy on training set for mu =", mu, "lr =", r, ":",
    #               margin_perceptron.accuracy(X_train_array, y_train, w, b) * 100)
    #
    # print()
    # print("Pocket Perceptron")
    # print("--" * 50)
    # for r in learning_rates:
    #     w, b = pocket_perceptron.pocket_perceptron_train(X_train_array, y_train, epochs=20, lr=r)
    #     print("Pocket perceptron accuracy on training set for lr =", r, ":", pocket_perceptron.accuracy(X_train_array, y_train, w,b)*100)

train_test()

