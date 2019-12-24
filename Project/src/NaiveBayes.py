import libsvm
import numpy as np
from sklearn import preprocessing
import csv
import time


class NaiveBayesClassifier(object):

    def __init__(self, X_train, y_train, num_features, l=1):
        self.num_features = num_features
        self.numPos = (y_train == 1).sum()
        self.numNeg = y_train.size - self.numPos
        self.posPrior = self.numPos/y_train.size
        self.negPrior = 1 - self.posPrior
        self.X_train = X_train
        self.y_train = y_train
        self.l = l
        self.positive_probs = [[],[]]
        self.negative_probs = [[],[]]

    def countProbabilities(self, feature_index, feature_value, label):

        feature_count = 0
        for i in range(0, self.X_train.shape[0]):
            if self.X_train[i,feature_index] == feature_value and self.y_train[i] \
                    == \
                    label:
                feature_count += 1
        if label == 1:
            label_count = self.numPos
        else:
            label_count = self.numNeg

        return (feature_count + self.l)/(label_count + 2*self.l)

    def likelihoodEstimate(self, data_point, label):

        estimate = 1
        for i in range(0, self.num_features):
            if label == 1:
                if data_point[0,i] == 1:
                    estimate *= self.positive_probs[1][i]
                elif data_point[0,i] == 0:
                    estimate *= self.positive_probs[0][i]
            elif label == 0:
                if data_point[0,i] == 1:
                    estimate *= self.negative_probs[1][i]
                elif data_point[0,i] == 0:
                    estimate *= self.negative_probs[0][i]
        return estimate



    def fit(self): # popoulates self.positive_probs and self.negative_provs
        for feature in range(0,num_features):
            self.positive_probs[0].append(self.countProbabilities(feature, 0, 1))
            self.negative_probs[0].append(self.countProbabilities(feature, 0, 0))

            self.positive_probs[1].append(self.countProbabilities(feature, 1, 1))
            self.negative_probs[1].append(self.countProbabilities(feature, 1, 0))



    def predict(self, data):

        predictions = np.array([])
        for i in range(0, data.shape[0]):
            positiveProbability = self.posPrior * self.likelihoodEstimate(
                data[i], 1)
            negativeProbability = self.negPrior * self.likelihoodEstimate(
                data[i], 0)

            if positiveProbability < negativeProbability:
                predictions = np.append(predictions, 0)
            else:
                predictions = np.append(predictions, 1)

        return predictions

    def accuracy(self, X, y):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(predictions)
        return accuracy

    def test_nb(self, X_test, id_list):
        with open('../predictions/nb_predictions_log.csv', 'w', newline='') as \
                csvfile:
            writer = csv.writer(csvfile)
            rows = []
            writer.writerow(["example_id","label"])
            print("Generating predictions...")
            predictions = self.predict(X_test)
            for i in range(0, X_test.shape[0]):
                id_pred = id_list[i]
                rows.append([id_pred, str(int(predictions[i]))])
            writer.writerows(rows)

        print("Done.")


def get_id_list():

    id_list = []
    with open(r"../data/data/data-splits/eval.id") as file:
        lines = file.readlines()

    for line in lines:
        id_list.append(line.strip())

    return id_list


def discretize(data):

    for i in range(0, data.shape[0]):
        threshold = data[i].mean()
        binarizer = preprocessing.Binarizer(threshold).fit(data)
        data[i] = binarizer.transform(data[i])


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

    print("Discretizing...")
    discretize(X_train)
    discretize(X_test)
    discretize(X_anon)
    print("Done\n")

    lambdas = [1]
    start = time.time()
    for l in lambdas:
        nbc = NaiveBayesClassifier(X_train, y_train, num_features, l)
        print("Fitting...")
        nbc.fit()
        print("Fitting done. Now predicting...")
        print("Accuracy of NBC on train data set for l =", l, ": ", nbc.accuracy(
            X_train, y_train)*100)
        print("Accuracy of NBC on test data set for l =", l, ": ", nbc.accuracy(
            X_test, y_test)*100)
        print("Accuracy of NBC on anon data set for l =", l, ": ", nbc.accuracy(
            X_anon, y_anon)*100)
    end = time.time()
    print(end - start)
    # id_list = get_id_list()
    # nbc.test_nb(X_anon, id_list)


