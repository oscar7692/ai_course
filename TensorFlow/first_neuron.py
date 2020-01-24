import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import os


# Just disables  AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# neural network example
caracteristicas = 10
neuronas = 4

# filas x columnas
x = tf.placeholder(tf.float32, (None, caracteristicas))
w = tf.Variable(tf.random.normal([caracteristicas, neuronas]))
b = tf.Variable(tf.ones([neuronas]))

multiplicacion = tf.matmul(x, w)
z = tf.add(multiplicacion, b)

# funcion de activacion
activacion = tf.sigmoid(z)
# inicializa las variables
inicializacion = tf.global_variables_initializer()

valores_x = np.random.random([1, caracteristicas])

with tf.Session() as session:
    session.run(inicializacion)
    resultado = session.run(activacion, feed_dict={x:valores_x})
print(resultado)