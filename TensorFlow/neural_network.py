import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import os


# Just disables  AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# variables
aleatorio_a = np.random.uniform(0,50,(4,4))

aleatorio_b = np.random.uniform(0,50,(4,1))

# placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

# operations
suma = a + b
multiplicacion = a * b

# session creation
with tf.Session() as session:
    resultado_suma = session.run(suma, feed_dict={a:10,b:20})
    print(resultado_suma)

with tf.Session() as session:
    resultado_suma = session.run(suma, feed_dict={a:aleatorio_a,b:aleatorio_b})
    print(resultado_suma)

with tf.Session() as session:
    resultado_multiplicacion = session.run(multiplicacion, 
                                           feed_dict={a:10,b:20})
    print(resultado_multiplicacion)

with tf.Session() as session:
    resultado_multiplicacion = session.run(multiplicacion, 
                                           feed_dict={
                                               a:aleatorio_a,
                                               b:aleatorio_b
                                               })
    print(resultado_multiplicacion)