from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import libsvm
import time
import csv
from sklearn.preprocessing import KBinsDiscretizer

def get_id_list():

    id_list = []
    with open(r"../data/data/data-splits/eval.id") as file:
        lines = file.readlines()

    for line in lines:
        id_list.append(line.strip())

    return id_list

enc = KBinsDiscretizer(n_bins=4, encode="onehot", strategy="uniform")

X_train, y_train, num_features = libsvm.read_libsvm(
    "../data/data/data-splits/data.train")
X_train_binned = enc.fit_transform(X_train.toarray())

X_test, y_test, num_features = libsvm.read_libsvm(
    "../data/data/data-splits/data.test")
X_test_binned = enc.fit_transform(X_test.toarray())

X_test_eval, y_test_eval, num_features = libsvm.read_libsvm(
    "../data/data/data-splits/data.eval.anon")

X_eval_binned = enc.fit_transform(X_test_eval.toarray())
id_list = get_id_list()

print("*" * 50 + "DTREE" + "*" * 50)
depths = [2]
print("Accuracies for non binned data")

start = time.time()
for d in depths:
    dtree = DecisionTreeClassifier(random_state=1, max_depth=d)
    dtree = dtree.fit(X_train, y_train)
    y_train_predict = dtree.predict(X_train)
    print("Accuracy of dtree on train set for depth = ", d ,":",
           accuracy_score(y_train, y_train_predict))

    y_test_predict = dtree.predict(X_test)
    print("Accuracy of dtree on test set for depth = ", d ,":",
                 accuracy_score(y_test, y_test_predict))


    y_test_predict_eval = dtree.predict(X_test_eval)
    print("Accuracy of dtree on eval set: for depth = ", d ,":",
                accuracy_score(y_test_eval, y_test_predict_eval))
    print()

    id_list = get_id_list()

    print("Generating predictions for data.eval.anon")

    X_test, y_test, num_features = libsvm.read_libsvm(r"C:\Users\Abhi\Documents\AB\Fall19\ML\Project\data\data\data"
                                                       "-splits\data.eval.anon")
    y_test_predict = dtree.predict(X_test)
    print(print("Accuracy of dtree on test set: ", accuracy_score(y_test, y_test_predict)))
    with open('decision_tree_predictions_2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        rows = []
        writer.writerow(["example_id", "label"])
        for i in range(0, len(y_test_predict)):
            id_pred = id_list[i]
            rows.append([id_pred, str(int(y_test_predict[i]))])
        writer.writerows(rows)
