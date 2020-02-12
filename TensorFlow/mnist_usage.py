import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_core.examples.tutorials.mnist import input_data


"""
I'm using tensorflow-cpu==2.1.0 and I had the same issue

ModuleNotFoundError: No module named 'tensorflow.examples'

I solve this downloading manually the directory called "tutorials" from tensorflow repo
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials

and placed it in my virtual env directory
myenv\Lib\site-packages\tensorflow_core\examples\

after that it works fine
"""


# import dataset
mnist = input_data.read_data_sets("MNIST/", one_hot=True)

# see nuber of examples to train
print(mnist.train.num_examples)
# show an image
imagen = mnist.train.images[1]
# change image size
imagen = imagen.reshape(28,28)
print(imagen)
plt.imshow(imagen)
plt.show()