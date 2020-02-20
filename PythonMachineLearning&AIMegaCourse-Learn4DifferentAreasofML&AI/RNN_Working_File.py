import numpy as np
import pandas as pd
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.neighbors import KNeighborsClassifier # classification algorithm
from sklearn.utils import shuffle


data = pd.read_csv("car.data")

# converting non numerical data
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
clas = le.fit_transform(list(data["class"]))

predict = "class"

X =  list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(clas)

# split train testing
x_train, x_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(X, y, test_size=0.1)
# print(x_train, y_test)

