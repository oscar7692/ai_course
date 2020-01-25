import os
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
# Just disables  AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


"""
aprendizaje supervisado (clasificacion):
partiendo de una lista de hombres y mujeres con sus caracteristicas de altura y
peso, crearemos un algortimo para averiguar el sexo de una nueva persona en
funcion de sus datos de altura y peso
"""


ingresos = pd.read_csv("ingresos.csv")
# return an array
ingresos["income"].unique()

# this function changes values for 0 and 1 to work with tensorflow
def cambio_valor(valor):
    if valor == "<=50K":
        return 0
    else:
        return 1

ingresos["income"] = ingresos["income"].apply(cambio_valor)
# print(ingresos.head())

# generate data excepting the one that we want predic
# drops the column income and place it in column 1
datos_x = ingresos.drop("income", axis=1)
datos_y = ingresos["income"]

# divide datos_x and datos_y in a tuple for train_data and test_data
# savin 30% for each one
x_train, x_test, y_train, y_test = train_test_split(datos_x, 
                                                    datos_y, 
                                                    test_size=0.8)
# print("\n", x_test.head())

# create variables to store caracteristics values in function if those are
# numeric values or text values for the headers in csv
gender = tf.feature_column.categorical_column_with_vocabulary_list("gender",
                                                                   ["Female, Male"])

occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation",
                                                                   hash_bucket_size=1000)

marital_status = tf.feature_column.categorical_column_with_hash_bucket("marital-status",
                                                                   hash_bucket_size=1000)

relationship = tf.feature_column.categorical_column_with_hash_bucket("relationship",
                                                                   hash_bucket_size=1000)

education = tf.feature_column.categorical_column_with_hash_bucket("education",
                                                                   hash_bucket_size=1000)

native_country = tf.feature_column.categorical_column_with_hash_bucket("native-country",
                                                                   hash_bucket_size=1000)

workclass = tf.feature_column.categorical_column_with_hash_bucket("workclass",
                                                                   hash_bucket_size=1000)

age = tf.feature_column.numeric_column("age")

fnlwgt = tf.feature_column.numeric_column("fnlwgt")

educational_num = tf.feature_column.numeric_column("educational-num")

capital_gain = tf.feature_column.numeric_column("capital-gain")

capital_loss = tf.feature_column.numeric_column("capital-loss")

hours_per_week = tf.feature_column.numeric_column("hours-per-week")

# creating a list that holds all the abvove columns 
columnas_categorias = [gender, occupation, marital_status, relationship,
                       education, native_country, workclass, age, fnlwgt,
                       educational_num, capital_gain, capital_loss,
                       hours_per_week]

# cretaing entry function for estimation
funcion_entrada = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_train,
                                                      y=y_train,
                                                      batch_size=100,
                                                      num_epochs=None,
                                                      shuffle=True)

# model creation
modelo = tf.estimator.LinearClassifier(feature_columns=columnas_categorias)

# train model
modelo.train(input_fn=funcion_entrada, steps=8000)

# predicton function
funcion_prediccion = tf.compat.v1.estimator.inputs.pandas_input_fn(x=x_test,
                                                         batch_size=len(x_test),
                                                         shuffle=False)

# creation of predictions generator
generador_predicciones = modelo.predict(input_fn=funcion_prediccion)

# prediction list generated of generador_predicciones
predicciones = list(generador_predicciones)
# print predicciones
# print(predicciones)

# take values of predicciones 
predicciones_finales = [prediccion["class_ids"][0] for prediccion in predicciones]

# generate an inform to evaluete model performance
print(classification_report(y_test, predicciones_finales))