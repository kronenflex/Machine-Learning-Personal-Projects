# -*- coding: utf-8 -*-
"""
Created on Sun Jun 14 21:45:19 2020

@author: DiegoIgnacioPavezOla
"""

#Plantilla  de pre procesado
#Importar las libreria

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importar el dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 3].values

#tratamientos de los NAs
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Codificar datos categoricos
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:, 0] = le_X.fit_transform(X[:,0])


#transformar variable categorica a dummy
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Variable X
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[0])],
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)

#Variable Y
le_Y = preprocessing.LabelEncoder()
Y[:] = le_Y.fit_transform(Y)

#Dividir el dataset en conjunto de entrenamiento y testing

from sklearn.model_selection  import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
