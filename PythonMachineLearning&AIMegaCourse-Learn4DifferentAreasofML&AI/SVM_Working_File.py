# Supporting Vector Machine
import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import  KNeighborsClassifier

cancer = datasets.load_breast_cancer()

# print(cancer.feature_names)
# print(cancer.target_names)

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(cancer.data, cancer.target, 
                                             test_size=0.2)

# print(x_train, y_train)
classes = ['maligant', 'benign']

# clf = svm.SVC() # prediction value 0.8947368421052632
# clf = svm.SVC(kernel="linear") # prediction value 0.9912280701754386
# clf = svm.SVC(kernel="poly") # prediction value 0.8947368421052632
# clf = svm.SVC(kernel="poly", degree=2) # prediction value 0.9122807017543859
# clf = svm.SVC(kernel="linear", C=1) # prediction value0.9824561403508771
# clf = svm.SVC(kernel="linear", C=2) # prediction value 0.9736842105263158
clf = KNeighborsClassifier(n_neighbors=9) # prediction value 0.9298245614035088
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

acc = metrics.accuracy_score(y_test, y_pred)
print(acc)