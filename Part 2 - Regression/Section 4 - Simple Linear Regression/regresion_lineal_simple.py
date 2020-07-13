# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:04:14 2020

@author: DiegoIgnacioPavezOla
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importar el dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values

#Dividir el dataset en entrenamiento y test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

X_train = X_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# Crear modelo de regresion lineal simple
'''from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
regression = linear_model.LinearRegression()'''


from sklearn.linear_model import LinearRegression
regression = LinearRegression()
regression.fit(X_train, y_train)

#Predecir el conjunto de test

y_pred = regression.predict(X_test)

# Visualizar los resultados de entrenamientos
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
#titulo
plt.title("Sueldo vs Años de Experiencia (Conjuntos de Entrenamiento)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo $")
plt.show()

#Visualizar los datos de testing
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regression.predict(X_train), color = "blue")
#titulo
plt.title("Sueldo vs Años de Experiencia (Conjuntos de Testing)")
plt.xlabel("Años de Experiencia")
plt.ylabel("Sueldo $")
plt.show()
y_pred2 = regression.predict([[2]])
