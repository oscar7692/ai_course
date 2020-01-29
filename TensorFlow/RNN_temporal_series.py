import matplotlib as plt
import matplotlib.pyplot as dis
import numpy as np
import pandas as pd
# compat.v1 enables placeholder 
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.preprocessing import MinMaxScaler


# reading csv file
leche = pd.read_csv("produccion-leche.csv", index_col="Month")
# print(leche, "\n", leche.info())

leche.index = pd.to_datetime(leche.index)
print(leche.plot())


# variables will store train data
conjunto_entrenamiento = leche.head(150)
conjunto_pruebas = leche.tail(18)

normalizacion =MinMaxScaler()
entrenamiento_normalizado = normalizacion.fit_transform(conjunto_entrenamiento)
pruebas_normalizado = normalizacion.transform(conjunto_pruebas)

# creating lotes with data
def lotes(datos_entrenamiento, tamanio_lote, pasos):
    comienzo = np.random.randint(0, len(datos_entrenamiento) - pasos)
    lote_y = np.array(datos_entrenamiento[comienzo:comienzo+pasos+1]).reshape(1, pasos+1)
    return lote_y[:,:-1].reshape(-1,pasos,1), lote_y[:,1:].reshape(-1,pasos,1)

# constants
numero_entradas = 1
numero_pasos = 18
numero_neuronas = 120
numero_salidas = 1
tasa_aprendizaje = 0.001
numero_iteraciones_entrenamiento = 5000
tamanio_lote = 1

# placeholders
x = tf.placeholder(tf.float32, [None, numero_pasos, numero_entradas])
y = tf.placeholder(tf.float32, [None, numero_pasos, numero_salidas])

# creating neural layer
capa = tf.contrib.rnn.OutputProjectionWrapper(
    tf.contrib.rnn.BasicLSTMCe11(num_units=numero_neuronas,
                                 activation=tf.nn.relu), 
    output_size=numero_salidas)

salidas, estados = tf.nn.dynamic_rnn(capa, x, dtype=tf.float32)

# error function and optimizer function
funcion_error = tf.reduce_mean(tf.square(salidas - y))
optimizador = tf.train.AdamOptimizer(learnin_rate=tasa_aprendizaje)
entrenamiento = optimizador.minimize(funcion_error)

inint = tf.global_variables_initializer()
saver = tf.train.Saver()

# x and y lotes creation using tensorflow session
with tf.Session() as sesion:
    sesion.run(init)
    for iteracion in range(numero_iteraciones_entrenamiento):
        lote_x, lote_y = lotes(entrenamiento_normalizado, tamanio_lote, numero_pasos)
        sesion.run(entrenamiento, feed_dict={x:lote_x, y:lote_y})
        if iteracion %100 == 0:
            error = funcion_error.eval(feed_dict={x:lote_x, y:lote_y})
            print(iteracion, "\t Error: ", error)
        # saves current tensorflow session 
        saver.save(sesion, "./modelo_series_temporales")