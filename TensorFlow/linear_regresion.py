import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
import numpy as np
import os
# Just disables  AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# creating array
datos_x = np.linspace(0,10,10) + np.random.uniform(-1,1,10)
datos_y = np.linspace(0,10,10) + np.random.uniform(-1,1,10)

# creating graphic
plt.plot(datos_x, datos_y, "*")

np.random.rand(2)
m = tf.Variable(0.81)
b = tf.Variable(0.87)
error = 0

for x,y in zip(datos_x, datos_y):
    y_pred = m*x+b
    error = error + (y - y_pred)**2

optimizador = tf.train.GradientDescentOptimizer(learning_rate=0.001)
entrenaminto = optimizador.minimize(error)

inicializacion = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(inicializacion)
    pasos = 1
    for i in range(pasos):
        session.run(entrenaminto)
    final_m, final_b = session.run([m, b])
    
x_test = np.linspace(-1,11,10)
y_pred_2 = (final_m * x_test) + final_b

plt.plot(x_test, y_pred_2, 'r')
plt.plot(datos_x, datos_y, '*')
plt.show()

