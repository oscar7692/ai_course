# K-Nearest Neighbors
import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier # classification algorithm
from sklearn.utils import shuffle

# loading data
data = pd.read_csv("car.data")

# creating a label encoder object and then use that to encode each column 
# of our data into integers.
le = preprocessing.LabelEncoder()

# The method fit_transform() takes a list (each of our columns) and will 
# return to us an array containing our new values.
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clas = le.fit_transform(list(data["class"]))

predict = "class"

# we need to recombine our data into a feature list and a label list. 
# We can use the zip()
X =  list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clas)

# split train testing
# we will split our data into training and testing data
x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# print(x_train, y_test)

model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)

predicted = model.predict(x_test)

names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x],
          "Actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 9, True)
    print("Neighbors: ", n)