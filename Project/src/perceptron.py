import numpy as np
import libsvm
import csv
import time

class Perceptron(object):

    # Initialization of w, b and other hyper-parameters
    # happens in the train methods of respective perceptron variants

    def __init__(self, num_features):
        self.num_features = num_features

    def predict(self, X, w, b):
        predictions = np.array([])

        if np.shape(X) == (1,self.num_features):
            dot_product = X.dot(np.transpose(w))
            if dot_product + b >= 0:
                predictions = np.append(predictions, 1)
            else:
                predictions = np.append(predictions, 0)
        else:
            for example in X:
                dot_product = example.dot(np.transpose(w))
                if dot_product + b >=0:
                    predictions = np.append(predictions, 1)
                else:
                    predictions = np.append(predictions, 0)
        return predictions

    def accuracy(self, X, y, w, b):
        predictions = self.predict(X, w, b)
        accuracy = np.sum(predictions == y) / len(predictions)
        return accuracy

    def get_num_predictions(self, X, y, w, b):
        predictions = self.predict(X, w, b)
        return np.sum(predictions == y)

    def update(self,x, y, w, b, lr):
        # FILL IN
        if y == 0:
            y = -1
        w_new = w + lr * (y * x)
        b_new = b + lr * y
        return (w_new, b_new)

    def shuffle_arrays(self, X, y):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], y[idx]

    # SIMPLE PERCEPTRON

    def simple_perceptron_train(self, X_train, y_train, epochs=10, lr=0.01):
        # w = np.random.uniform(0, 1, size=X_train.shape[1])  # initialize w
        # b = 0  # initialize bias

        np.random.seed(0)
        random_number = np.random.uniform(-0.01, 0.01)
        w = np.full((1, self.num_features), random_number)
        b = random_number
        update_counter = 0
        for i in range(0, epochs):
            X_train, y_train = self.shuffle_arrays(X_train, y_train)
            for j in range(0, X_train.shape[0]):
                prediction = self.predict(X_train[j], w, b)
                if prediction[0] != y_train[j]:
                    update_counter += 1
                    w, b = self.update(X_train[j], y_train[j], w, b, lr)
                else:
                    continue
        print("Total updates =",update_counter)
        return w, b



    #AVERAGED PERCPETRON

    def averaged_perceptron_train(self, X_train, y_train, epochs=10, lr=0.01):
        # w = np.random.uniform(0, 1, size=X_train.shape[1])  # initialize w
        # b = 0  # initialize bias

        np.random.seed(0)
        random_number = np.random.uniform(-0.01, 0.01)
        w = np.full((1, self.num_features), 0)
        a = np.full((1, self.num_features), 0)
        b = 0
        b_a = 0
        update_counter = 0
        counter = 0
        for i in range(0, epochs):
            X_train, y_train = self.shuffle_arrays(X_train, y_train)
            for j in range(0, X_train.shape[0]):
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

        print("Total updates =",update_counter)
        return a/counter, b_a/counter

    def test_perceptron(self, X_test, w, b, id_list):
        with open('../predictions/perceptron_predictions_log.csv', 'w', newline='') \
                as \
                csvfile:
            writer = csv.writer(csvfile)
            rows = []
            writer.writerow(["example_id","label"])
            print("Generating predictions...")
            for i in range(0, X_test.shape[0]):
                prediction = self.predict(X_test[i],w,b)
                id_pred = id_list[i]
                rows.append([id_pred, str(int(prediction[0]))])
            writer.writerows(rows)

        print("Done.")


def get_id_list():

    id_list = []
    with open(r"../data/data/data-splits/eval.id") as file:
        lines = file.readlines()

    for line in lines:
        id_list.append(line.strip())

    return id_list


if __name__ == "__main__":

    X_train, y_train, num_features = libsvm.read_libsvm(r"../data/data/data-splits"
                                                        r"/data.train")
    X_test, y_test, num_features = libsvm.read_libsvm(r"../data/data/data-splits"
                                                        r"/data.test")
    X_anon, y_anon, num_features = libsvm.read_libsvm(r"../data/data/data-splits"
                                                        r"/data.eval.anon")

    X_train = X_train.log1p()
    X_test = X_test.log1p()
    X_anon = X_anon.log1p()

    simple_perceptron = Perceptron(num_features)
    average_perceptron = Perceptron(num_features)

    learning_rates = [10**-4]

    start = time.time()

    print("PERCEPTRON")
    print("--" * 50)
    for lr in learning_rates:
        w, b = simple_perceptron.simple_perceptron_train(X_train, y_train,
                                                         epochs=50, lr=lr)
        print("Simple Perceptron Accuracy on training set for lr =", lr, ":",
        simple_perceptron.accuracy(X_train, y_train, w, b) * 100)
        print("Simple Perceptron Accuracy on training set for lr =", ":",
              simple_perceptron.accuracy(X_test, y_test, w, b) * 100)
        print("Simple Perceptron Accuracy on anon set for lr =", ":",
        simple_perceptron.accuracy(X_anon, y_anon, w, b) * 100)
        print()
    end = time.time()
    # print(end - start)

    id_list = get_id_list()
    simple_perceptron.test_perceptron(X_anon, w, b, id_list)





