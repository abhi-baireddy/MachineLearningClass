import libsvm
import numpy as np
from scipy import sparse
import csv
import time

class LogisticRegression(object):

    def __init__(self, num_features):
        self.num_features = num_features

    def sigmoid(self, w, X_train):

        z = np.exp(-np.dot(X_train, w.T))
        return 1/(1+z)

    def loss_function(self, w, X_train, y_train):

        z = self.sigmoid(w, X_train)
        return -((np.dot(y_train, np.log(z))) + (np.dot(1-y_train, np.log(
            1-z))))/X_train.shape[0]

    def predict(self, w, X_train):

        probabilities = self.sigmoid(w, X_train)
        predictions = np.where(probabilities >= 0.5, 1, 0)
        return predictions

    def accuracy(self, w, X_train, y_train):

        predictions = self.predict(w, X_train)
        acc = np.sum(y_train == predictions.T)/len(predictions)
        return acc

    def gradient(self, w, X_train, y_train):

        z = self.sigmoid(w, X_train)
        return np.dot((z-y_train.T).T, X_train)/X_train.shape[0]

    def fit(self, X_train, y_train, lr=0.01, threshold=0.5):

        # initialize weights
        np.random.seed(0)
        random_number = np.random.uniform(-0.01, 0.01)
        w = np.full((1, self.num_features+1), random_number)
        loss = self.loss_function(w, X_train, y_train)
        change_in_loss = 1
        t = 0
        while t < 10**5:
            old_loss = loss
            # lr = lr/(1+t)
            w = w - (lr * self.gradient(w, X_train, y_train))
            loss = self.loss_function(w, X_train, y_train)
            change_in_loss = old_loss - loss
            t += 1
        return w

    def test(self, X_test, w, id_list):
        with open('../predictions/regression_predictions_log.csv', 'w', newline='') \
                as \
                csvfile:
            writer = csv.writer(csvfile)
            rows = []
            writer.writerow(["example_id","label"])
            print("Generating predictions...")
            predictions = self.predict(w, X_test)
            output = []
            for p in predictions:
                output.append(p[0])

            for i in range(0, X_test.shape[0]):
                id_pred = id_list[i]
                rows.append([id_pred, str(int(output[i]))])
            writer.writerows(rows)

        print("Done.")


def add_bias_term(X_train):

    b = np.full((1, X_train.shape[0]), 1)
    b = sparse.csr_matrix(b).T
    X_train_with_biases = sparse.csr_matrix(sparse.hstack([b, X_train]))
    return X_train_with_biases

def get_id_list():

    id_list = []
    with open(r"../data/data/data-splits/eval.id") as file:
        lines = file.readlines()

    for line in lines:
        id_list.append(line.strip())

    return id_list

if __name__ == "__main__":

    X_train, y_train, num_features = libsvm.read_libsvm(
        "../data/data/data-splits/data.train")

    X_test, y_test, num_features = libsvm.read_libsvm(
        "../data/data/data-splits/data.test")

    X_anon, y_anon, num_features = libsvm.read_libsvm(
        "../data/data/data-splits/data.eval.anon")

    X_train = X_train.log1p()
    X_test = X_test.log1p()
    X_anon = X_anon.log1p()

    X_train = add_bias_term(X_train)
    X_test = add_bias_term(X_test)
    X_anon = add_bias_term(X_anon)

    X_train = X_train.todense()
    X_test = X_test.todense()
    X_anon = X_anon.todense()

    y_train = np.array([y_train])
    y_test = np.array([y_test])
    y_anon = np.array([y_anon])

    model = LogisticRegression(num_features)
    start = time.time()
    print("Fitting...")
    w = model.fit(X_train, y_train)
    print("Fitting done. Now predicting...")
    learning_rates = [10**-2]
    for lr in learning_rates:
        print("Accuracy of logistic regressor on train set for lr =", lr, ":",
              model.accuracy(w, X_train, y_train))
        print("Accuracy of logistic regressor on test set for lr =", lr, ":",
              model.accuracy(w, X_test, y_test))
        print("Accuracy of logistic regressor on anon set for lr =", lr, ":",
              model.accuracy(w, X_anon, y_anon))
    end = time.time()
    print(end - start)
    id_list = get_id_list()
    model.test(X_anon, w, id_list)




