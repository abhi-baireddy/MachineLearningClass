import numpy as np
import libsvm
import csv
import time
from sklearn.preprocessing import MaxAbsScaler
class SVM(object):


    def __init__(self, num_features):
        self.num_features = num_features



    def predict(self, X, w, b):
        predictions = np.array([])

        if np.shape(X) == (1, self.num_features):
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
        if y == 0:
            y = -1
        w_new = w + lr * (y * x)
        b_new = b + lr * y
        return (w_new, b_new)

    def shuffle_arrays(self, X, y):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        return X[idx], y[idx]

    def svm_train(self, X_train, y_train, epochs=10, lr=0.01, c=1):
        # w = np.random.uniform(0, 1, size=X_train.shape[1])  # initialize w
        # b = 0  # initialize bias
        # initial_lr = lr
        prev_lr = lr
        w = np.full((1, self.num_features), 0)
        b = 0

        update_counter = 0
        for i in range(0, epochs):
            # lr_t = prev_lr/(1+(prev_lr*i)/c)
            # prev_lr = lr_t
            lr_t = lr/(1+i)
            X_train, y_train = self.shuffle_arrays(X_train, y_train)
            for j in range(0, len(y_train)):
                if y_train[j] == 0:
                    y = -1
                else:
                    y = 1
                if y*X_train[j].dot(np.transpose(w)) + b <= 1:
                    w = (1-lr_t)*w + lr_t * c * y * X_train[j]
                    b = (1-lr_t)*b + lr_t * c * y

                else:
                    w = (1 - lr_t) * w
                    b = (1 - lr_t) * b
        return w, b

    def test_svm(self, X_test, w, b, id_list):
        with open('../predictions/svm_predictions_log.csv', 'w', newline='') as \
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

    # scaler = MaxAbsScaler()

    X_train, y_train, num_features = libsvm.read_libsvm(r"../data/data/data-splits"
                                                        r"/data.train")
    X_test, y_test, num_features = libsvm.read_libsvm(r"../data/data/data-splits"
                                                        r"/data.test")
    X_anon, y_anon, num_features = libsvm.read_libsvm(r"../data/data/data-splits"
                                                        r"/data.eval.anon")

    # scaler.fit(X_train)
    # X_train = scaler.transform(X_train)
    #
    # scaler.fit(X_test)
    # X_test = scaler.transform(X_test)
    #
    # scaler.fit(X_anon)
    # X_anon = scaler.transform(X_anon)
    X_train = X_train.log1p()
    X_test = X_test.log1p()
    X_anon = X_anon.log1p()

    learning_rates = [10**-4]
    tradeoff_param = [1]
    svm = SVM(num_features)
    start = time.time()
    print("SVM")
    print("--" * 50)
    for lr in learning_rates:
        for c in tradeoff_param:
            w, b = svm.svm_train(X_train, y_train, epochs=50, lr=lr, c=c)
            print("SVM Accuracy on training set for lr =", lr, "c =", c,
                  ":",
                  svm.accuracy(X_train, y_train, w, b) * 100)
            print("SVM Accuracy on test set for lr =", lr, "c =", c, ":",
                  svm.accuracy(X_test, y_test, w, b) * 100)
            print("SVM Accuracy on anon set for lr =", lr, "c =", c, ":",
                  svm.accuracy(X_anon, y_anon, w, b) * 100)
    end = time.time()

    id_list = get_id_list()
    svm.test_svm(X_anon, w, b, id_list)



