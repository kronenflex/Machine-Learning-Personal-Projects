# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 16:55:57 2020

@author: DiegoIgnacioPavezOla
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importar el dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Ajustar la regresión con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)


# Predicción de nuestros modelos
y_pred = regression.predict([[6.5]])
print(y_pred)

# Visualización de los resultados del Modelo Polinómico
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X, regression.predict(X), color = "blue")
plt.title("Modelo de Regresión DTR")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()

# Opcional, no es muy bueno realizarlo
# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))

# Ajustar la regresión con el dataset
from sklearn.tree import DecisionTreeRegressor
regression = DecisionTreeRegressor(random_state = 0)
regression.fit(X, y)


# Visualizacion de los resultados del SVR con X_grid donde se suaviza la curva, si no lo quiero se coloca una X en vez de una X grid
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title("Modelo de Regresion DTR")
plt.xlabel('Posicion del empleado')
plt.ylabel("Sueldo (en $)")
plt.show()

# Visualización de los resultados del SVR (con valores intermedios y sin las variables escaladas)
X_2 = sc_X.inverse_transform(X)
y_2 = sc_y.inverse_transform(y)
X_grid = np.arange(min(X_2), max(X_2), 0.1)
X_grid = X_grid.reshape(-1, 1)
plt.scatter(X_2, y_2, color = "red")
plt.plot(X_grid, sc_y.inverse_transform(regression.predict(sc_X.transform(X_grid))), color = "blue")
plt.title("Modelo de Regresión DTR (con valores intermedios y sin las variables escaladas)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()
