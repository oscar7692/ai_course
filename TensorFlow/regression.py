import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
# Just disables  AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

"""
aprendizaje supervisado (regresion):
patiendo de una lista de precios (etiquetas) de casas con sus caracteristicas,
crearemos un algoritmo de aprendizaje supervisado  que sea capaz de predecir el
precio de una casa en funcion de sus caracteristicas
(tamaño, numero de habitaciones y zona)
"""


casas = pd.read_csv("precios-casas.csv")

casas_x = casas.drop("median_house_value", axis=1)
casas_y = casas["median_house_value"]

x_train, x_test, y_train, y_test = train_test_split(casas_x, casas_y,
                                                    test_size=0.30)
# print(x_test.head(),"\n")
# print(y_test.head())

# overwrites variables to normalize data to work with tensorflow
normalizador = MinMaxScaler()
normalizador.fit(x_train)

x_train = pd.DataFrame(data=normalizador.transform(x_train),
                       columns=x_train.columns,
                       index=x_train.index)

x_test = pd.DataFrame(data=normalizador.transform(x_test),
                      columns=x_test.columns,
                      index=x_test.index)
# print(x_train.head(),"\n\n", x_test.head())

# creating variables with categorical columns
longitude = tf.feature_column.numeric_column("longitude")
latitude = tf.feature_column.numeric_column("latitude")
housing_median_age = tf.feature_column.numeric_column("housing_median_age")
total_rooms = tf.feature_column.numeric_column("total_rooms")
total_bedrooms = tf.feature_column.numeric_column("total_bedrooms")
population = tf.feature_column.numeric_column("population")
households = tf.feature_column.numeric_column("households")
median_income = tf.feature_column.numeric_column("median_income")

columnas = [longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income]

# emtry function
funcion_entrada = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train,
                                                                y=y_train,
                                                                batch_size=10,
                                                                num_epochs=1000,
                                                                shuffle=True)

# creating tensorflow model
modelo = tf.estimator.DNNRegressor(hidden_units=[10,10,10], 
                                   feature_columns=columnas)

# training model
modelo.train(input_fn=funcion_entrada, steps=8000)

# generate predicton function
funcion_entrada_prediccion = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,
                                                                           batch_size=10,
                                                                           num_epochs=1,
                                                                           shuffle=False)

# predicton generator
generador_predicciones = modelo.predict(funcion_entrada_prediccion)
predicciones = list(generador_predicciones)
# print(predicciones)

# estimated values
predicciones_finales = [prediccion["predictions"] for prediccion in predicciones]
# print(predicciones_finales)

# estimate cuadratic medium error
# show the cuadratic medium error over current houses values 
print(mean_squared_error(y_test, predicciones_finales)**0.5)