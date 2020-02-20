import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sklearn
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle


# read csv file
data = pd.read_csv("student-mat.csv", sep=";")
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G3"

X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, test_size=0.1)
"""
best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # model definition
    # commented due pickle model is loaded instead of created it
    linear = linear_model.LinearRegression()
    # model creation
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)
    # accuracy
    print(acc)

    if acc > best:
        best = acc
        # saving pickle model
        with  open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

# load pickle in our linear model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# print accuracy, coeficient, intercept
print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# predictions 
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# attribute retrived from data
p = "absences"
# plot style
style.use("ggplot")
plt.scatter(data[p], data["G3"])
plt.xlabel(p); plt.ylabel("Final Grade");
plt.show()
