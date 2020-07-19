# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 12:16:06 2020

@author: DiegoIgnacioPavezOla
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Escalado de variables
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1,1))


# Ajustar la regresion con el dataset
from sklearn.svm import SVR

# La sigmoide no funciona bien
#regression =  SVR(kernel = 'sigmoid')
# la gausiana si funciona mejor
regression =  SVR(kernel = 'rbf')
regression.fit(X,y)

# Prediccion de nuestros modelos con SVR y se cambia la variable X escalandola
#y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5, .., .., .]])))) # de forma generica para predecir los que quiera
y_pred = regression.predict(sc_X.transform([[6.5]]))
y_pred = sc_y.inverse_transform(y_pred)


# Visualizacion de los resultados del SVR con X_grid donde se suaviza la curva, si no lo quiero se coloca una X en vez de una X grid
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title("Modelo de Regresion (SVR)")
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
plt.title("Modelo de Regresión (SVR) (con valores intermedios y sin las variables escaladas)")
plt.xlabel("Posición del empleado")
plt.ylabel("Sueldo (en $)")
plt.show()



# Mala prediccion
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importar el dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Ajustar la regresion con el dataset
from sklearn.svm import SVR
regression =  SVR(kernel = 'sigmoid')
regression.fit(X,y)

# Prediccion de nuestros modelos con SVR
#y_pred = sc_y.inverse_transform(regression.predict(sc_X.transform(np.array([[6.5]]))))
y_pred = regression.predict([[6.5]])
 
# Visualizacion de los resultados del SVR
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regression.predict(X_grid), color = 'blue')
plt.title("Modelo de Regresion (SVR)")
plt.xlabel('Posicion del empleado')
plt.ylabel("Sueldo (en $)")
plt.show()'''