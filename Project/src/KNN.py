import libsvm
import numpy as np
import sklearn.metrics.pairwise
import csv

class KNN(object):

    def __init__(self, num_features, k=20):

        self.num_features = num_features
        self.k = k

    def fit(self, X_train, y_train):

        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):

        predictions = []
        for i in range(0, X_test.shape[0]):
            distances = []
            for j in range(0, self.X_train.shape[0]):
                d = sklearn.metrics.pairwise.pairwise_distances(X_test[i],
                                                                X_train[j])
                d_tup = (d, j)
                distances.append(d_tup)
            distances = sorted(distances, key=lambda x: x[0])
            labels = distances[0:self.k]
            nearest_labels = []
            for l in range(0, len(labels)):
                nearest_labels.append(self.y_train[labels[l][1]])
            positive_label_count = nearest_labels.count(1)
            negative_label_count = nearest_labels.count(0)
            if positive_label_count > negative_label_count:
                predictions.append(1)
            else:
                predictions.append(0)

        return predictions

    def accuracy(self, X_test, y_test):
        predictions = self.predict(X_test)
        return np.sum(predictions == y_test) / len(predictions)

    def test_knn(self, X_test, id_list):
        with open('../predictions/knn_predictions_log.csv', 'w', newline='') as \
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

    knn = KNN(num_features, 20)
    knn.fit(X_train, y_train)
    K = [20]

    for k in K:
        print("Accuracy of logistic regressor on train set for k =", k, ":",
              knn.accuracy(X_train, y_train))
        print("Accuracy of logistic regressor on test set for k =", k, ":",
              knn.accuracy(X_test, y_test))
        print("Accuracy of logistic regressor on anon set for k =", k, ":",
              knn.accuracy(X_anon, y_anon))

    id_list = get_id_list()
    knn.test_knn(X_anon, id_list)
