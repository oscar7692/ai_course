# Linear Regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import sklearn
from matplotlib import style
from sklearn import linear_model
from sklearn.utils import shuffle


# Loading in Our Data
data = pd.read_csv("student-mat.csv", sep=";")

"""Trimming Our Data
Since we have so many attributes and not all are relevant we need to select 
the ones we want to use. We can do this by typing the following."""
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

""" Separating Our Data
Now that we've trimmed our data set down we need to separate it into 4 arrays. 
However, before we can do that we need to define what attribute we are trying 
to predict. This attribute is known as a label. The other attributes that will 
determine our label are known as features. Once we've done this we will use 
numpy to create two arrays. One that contains all of our features and one that 
contains our labels.
"""
predict = "G3"
X = np.array(data.drop([predict], 1))
y = np.array(data[predict])

""" 
After this we need to split our data into testing and training data. We will 
use 90% of our data to train and the other 10% to test. The reason we do this 
is so that we do not test our model on data that it has already seen.
"""
x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, test_size=0.1)


"""
# TRAIN MODEL MULTIPLE TIMES FOR BEST SCORE
best = 0
for _ in range(30):

    x_train, x_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # defining the model which we will be using.
    linear = linear_model.LinearRegression()

    # train and score our model using the arrays we created
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)

    # accuracy
    print(acc)

    # If the current model has a better score than one we've already trained 
    # then save it
    if acc > best:
        best = acc

        # Saving Our Model
        # To save our model we will write to a new file using pickle.dump().
        with  open("studentmodel.pickle", "wb") as f:
            pickle.dump(linear, f)"""

# load pickle in our linear model
pickle_in = open("studentmodel.pickle", "rb")
linear = pickle.load(pickle_in)

# see the constants used to generate the line we can type the following.
print("-------------------------")
print("Co: \n", linear.coef_)
print("Intercept: \n", linear.intercept_)
print("-------------------------")

# Gets a list of all predictions
predictions = linear.predict(x_test)
for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])


# attribute retrived from data
# Drawing and plotting model
p = "absences" # Change this to G1, G2, studytime or absences to see other graphs
# plot style
style.use("ggplot")
plt.scatter(data[p], data["G1"])
plt.legend(loc=4)
plt.xlabel(p); plt.ylabel("Final Grade");
plt.show()