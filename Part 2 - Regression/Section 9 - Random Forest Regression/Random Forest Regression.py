# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 17:59:01 2020

@author: DiegoIgnacioPavezOla
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importar el dataset
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Ajustar el Random Forest con el dataset
from sklearn.ensemble import RandomForestRegressor
regression = RandomForestRegressor(n_estimators = 300, random_state = 0)
regression.fit(X, y)

# Predicción de nuestros modelos con Random Forest
y_pred = regression.predict([[6.5]])
print(y_pred)

# Visualización de los resultados del Random Forest
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regression.predict(X_grid), color = "blue")
plt.title("Modelo de Regresión con Random Forest")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()