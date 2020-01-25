import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras


# source https://www.youtube.com/watch?v=6g4O5UOH304
# Just disables the warning, doesn't enable AVX/FMA
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# data that will be donwloaded
data = keras.datasets.fashion_mnist

# loading data in variables
(train_images, train_labels), (test_images, test_labels) = data.load_data()

# creating label names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# dividing image pixels to create a more handable working data
train_images = train_images/255.0
test_images = test_images/255.0

# # print the image
# print(train_images[7])

# # display image in a plot table
# plt.imshow(train_images[7], cmap=plt.cm.binary)
# plt.show()

#############################creating a model###################################

# here is defined the archiecture of the model layers
# keras.Sequential --> creates a sequence of layers  
# keras.layers.Flatten --> is used to make data passable to neurons
# keras.layers.Dense --> make a fully connected neural network and softmax will
# pick values for each neurons so that all of those values add up to one, 
# essentially is the probability of the network  thinking it's a certain value
# relu rectified linear unit

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

# declaring some parameter for the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

# training model
# epochs essentially is how many times the model is gonna see this information,
# is a randomly pick images and corresponding labels
model.fit(train_images, train_labels, epochs=5)

# evaluate model
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("tested accuracy: ", test_acc)

################### using model to make predictions ############################

prediction = model.predict([test_images[7]])

for i in range(5):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Acutal: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()
print(class_names[np.argmax(prediction[0])])