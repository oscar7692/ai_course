import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 


"""
erorr en la matriz
"""

# constants
numero_entradas = 2
numero_neuronas = 3

# placeholders for data entries for layers
#                    data_type   n_rows  n_columns
x0 = tf.placeholder(tf.float32, [None, numero_entradas])
x1 = tf.placeholder(tf.float32, [None, numero_entradas])

# variables 
# wx --> links weight, layer one
# wy --> entry for layer two
wx = tf.Variable(tf.random_normal(shape=[numero_entradas, numero_neuronas]))
wy = tf.Variable(tf.random_normal(shape=[numero_entradas, numero_neuronas]))
b = tf.Variable(tf.zeros([1, numero_neuronas]))

# activation function
y0 = tf.tanh(tf.matmul(x0, wx) + b)
y1 = tf.tanh(tf.matmul(y0, wy) + tf.matmul(x1, wx) + b)

# data creation
lote_x0 = np.array([ [0, 1], [2, 3], [4, 5] ])
lote_x1 = np.array([ [2, 4], [3, 9], [4, 1] ])

# initialice variable will be used
init = tf.global_variables_initializer()

# sesion creation
with tf.Session() as session:
    session.run(init)
    salida_y0, salida_y1 = session.run([y0, y1], 
                                       feed_dict={x0:lote_x0, x1:lote_x1})