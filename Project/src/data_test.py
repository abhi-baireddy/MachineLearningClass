import libsvm
import numpy as np
import random
from sklearn import preprocessing
from scipy import sparse
from sklearn.preprocessing import MaxAbsScaler

X_train, y_train, num_features = libsvm.read_libsvm(
    "C:\\Users\\Abhi\\Documents\\MyFiles\\AB\\GradSchool\\Fall19\\ML\\Project"
    "\\data\\data\\data-splits\\data.train")


# log_x = X_train.log1p()
# print(type(log_x))

#print(X_train[1])
#print(log_x[1])

# normalizing the data

# normal_x = preprocessing.normalize(X_train)
# print(normal_x[2])


#min max scaling

scaler = MaxAbsScaler()
scaler.fit(X_train)
X_minmax = scaler.transform(X_train)


# print(X_train[0])
# print(binarized_X[0])







X_train = X_train.log1p()


# def discretize(data):
#     for i in range(0, data.shape[0]):
#         threshold = data[i].mean()
#         binarizer = preprocessing.Binarizer(threshold).fit(data)
#         data[i] = binarizer.transform(data[i])
#
# print(X_train[0])
# discretize(X_train)
# print(X_train[0])

print(X_train[0])
b = np.full((1, X_train.shape[0]), 1)
X_train_with_biases = X_train
b = sparse.csr_matrix(b)
b = b.T
X_train_with_biases = sparse.csr_matrix(sparse.hstack([b, X_train_with_biases]))
print(X_train_with_biases[0])



