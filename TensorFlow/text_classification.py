import numpy as np
import os
import tensorflow as tf
from tensorflow import keras


# source https://www.youtube.com/watch?v=6g4O5UOH304
# Just disables  AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# call data set from imdb
data = keras.datasets.imdb

# load data set
# num_words takes the most frequent 10000 words in data set
(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# print(train_data[0])

word_index = data.get_word_index()

# assing our own values to list that are contained in the data set model
word_index = {k:(v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["UNSUED>"] = 3

# will reverse the key vales that are contained in the dictionary nad every
# word will point to a numeric value
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# maxlen that tensorflow handle is 256
train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=250
)
test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=250
)

# this def formt output data
def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# print(decode_review(test_data[0]))

# # model is declared here
# model = keras.Sequential()
# model.add(keras.layers.Embedding(88000, 16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation="relu"))
# model.add(keras.layers.Dense(1, activation="sigmoid"))
# model.summary()

# # declaing neuron
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# # cut training data to 10000 values
# # hyper parameters or hyper tuning
# # changes parameters each time until we get a model that is better and accurate
# x_val = train_data[:10000]
# x_train = train_data[10000:]

# y_val = train_labels[:10000]
# y_train = train_labels[10000:]

# # fit the model
# # batch_size is how many movies reviews gonna do each time and load in at once
# fitModel = model.fit(x_train, y_train, epochs=40,
#                      batch_size=512,
#                      validation_data=(x_val, y_val),
#                      verbose=1)
# results = model.evaluate(test_data, test_labels)
# print(results)


# # saving and load models
# # to save model we need to comment the previous code form line 83 - 89
# # the model will be saved as binary data which meas we'll be able to read it
# # really quickly and usewhen we want ti actually make predictons
# model.save("model.h5")
model = keras.models.load_model("model.h5")


def review_encode(s):
    encoded = [1]
    
    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


# # getting test file
with open("test.txt", encoding="utf-8",) as f:
    for line in f.readlines():
        nline = line.replace(",","").replace(".","").replace("(","").replace(")","").replace(":","").replace("\"","").strip().split(" ")
        encode = review_encode(nline)
        encode = keras.preprocessing.sequence.pad_sequences(
            [encode],
            value=word_index["<PAD>"],
            padding="post",
            maxlen=250
            )
        predict = model.predict(encode)
        print(line)
        print(encode)
        print(predict[0])


# # reviews of prediction
# test_review = test_data[0]
# predict = model.predict([test_review])
# print("review: ")
# print(decode_review(test_review))
# print("prediction: " + str(predict[0]))
# print("actual: " + str(test_labels[0]))
print(results)