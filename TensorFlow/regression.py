import os
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
# Just disables  AVX/FMA warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


casas = pd.read_csv("precios-casas.csv")

casas_x = casas.drop("median_house_value", axis=1)
casas_y = casas["median_house_value"]

x_train, x_test, y_train, y_test = train_test_split(casas_x, casas_y, test_size=0.30)
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

print(x_train.head(),"\n\n", x_test.head())