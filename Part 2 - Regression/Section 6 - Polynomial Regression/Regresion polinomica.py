# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 20:41:13 2020

@author: DiegoIgnacioPavezOla
"""
#Regresion polinomica

import pandas as pd
import numpy as np
import matplotlib as plt

#importar el data set
dataset = pd.read_csv('Position_Salaries.csv')
# elegir las variables
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values

# Dividir el dataset no se realiza por la poca cantidad de data

# Visualizar los resultados de entrenamientos
import matplotlib.pyplot as plt1
plt1.scatter(X,y)
#titulo
plt1.title("Lugares de trabajo")
plt1.xlabel("Posicion")
plt1.ylabel("Sueldo $")
plt1.show()

#ajustar la regresion lineal con el dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


#Ajustar la regresion polinomica con el dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

#Visualizacion de los modelos linear
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "blue")
plt.title("Modelo de regresion lineal")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show

#Visualizacion de los modelos polinomicos
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "blue")
plt.title("Modelo de regresion polinomica")
plt.xlabel("Posicion del empleado")
plt.ylabel("Sueldo en $")
plt.show

#Prediccion de nuestros modelos
lin_reg.predict([[6.5]])
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))


