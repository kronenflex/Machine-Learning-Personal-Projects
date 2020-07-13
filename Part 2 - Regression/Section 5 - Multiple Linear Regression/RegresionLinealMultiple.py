# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 21:52:22 2020

@author: DiegoIgnacioPavezOla
"""
# Regresion Lineal Multiple

import numpy as np
import pandas as pd
import matplotlib as plt

#Importar dataset
dataset = pd.read_csv("50_Startups.csv")

#variables dependientes e independientes
X = dataset.iloc[:, :4].values
y = dataset.iloc[:, -1].values

#Codificar datos categoricos
from sklearn import preprocessing
le_X = preprocessing.LabelEncoder()
X[:, 3] = le_X.fit_transform(X[:,3])

#Transformar variable en dummy
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

#Transformar variable X[4] en Dummy
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(categories='auto'),[3])],
    remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype = np.float)

#EVITAR LA TRAMPA DE LAS VARIABLES DUMMY
X = X[:, 1:]

#Test de training y testing

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#ajustar el modelo de regresion lineal multiple con el conjunto de entrenamiento
from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Prediccion de los resultados en el conjunto de testing
y_pred = regression.predict(X_test)

#Mejororar el modelo

# Construccion del modelo optimo de la regresion lineal multiple utilizando eliminacion hacia atras
#El modelo requiere de una columna de 1 al principio para poder calcular los distintos p valor
import statsmodels.api as sm
X =  np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)

# el modelo dice que el sl es de 00.5
SL = 0.05
#se crea todas la variables posibles inclyendo la columna de 1 (modelo1) se elimina la que tiene el p valor mas alto
X_opt = X[:, [0,1,2,3,4,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
print(regression_OLS.summary())

#se crea todas la variables posibles inclyendo la columna de 1 (modelo2) se elimina la que tiene el p valor mas alto
X_opt = X[:, [0,1,3,4,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
print(regression_OLS.summary())

#se crea todas la variables posibles inclyendo la columna de 1 (modelo3) se elimina la que tiene el p valor mas alto
X_opt = X[:, [0,3,4,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
print(regression_OLS.summary())

#se crea todas la variables posibles inclyendo la columna de 1 (modelo4) se elimina la que tiene el p valor mas alto
X_opt = X[:, [0,3,5]]
regression_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regression_OLS.summary()
print(regression_OLS.summary())

from sklearn.model_selection import train_test_split
X_train_opt, X_test_opt, y_train_opt, y_test_opt = train_test_split(X_opt, y, test_size = 0.2, random_state = 0)

y_pred2 = regression_OLS.predict(X_test_opt)
