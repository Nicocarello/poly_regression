### REGRESION POLINOMIAL

##   IMPORTO LAS LIBRERIAS

import numpy as np
from sklearn import datasets, linear_model
import matplotlib.pyplot as plt

##   PREPARO LOS DATOS

# IMPORTO LOS DATOS DE LA LIBRERIA DE SCIKIT-LEARN
boston = datasets.load_boston()
print(boston)

##  ENTENDIMIENTO DE LA DATA

# VERIFICO LA INFORMACION DEL DATASET

print("Informacion dataset:")
print(boston.keys())
print()

# CARACTERISTICAS DEL DATASET

print("Caracteristicas del dataset:")
print(boston.DESCR)
print()

# VERIFICO LA CANTIDAD DE DATOS 

print('Cantidad de datos:')
print(boston.data.shape)
print()

# VER EL NOMBRE DE LAS COLUMNAS

print('Nombre de columnas:')
print(boston.feature_names)
print()

### PREPARAR LA DATA REGRESION POLINOMIAL

# SELECCIONO SOLAMENTE LA COLUMNA 6 DEL DATASET

X_p = boston.data[:,np.newaxis,5]

# DEFINO LOS DATOS CORESPONDIENTES A LAS ETIQUETAS

y_p = boston.target

# GRAFICO LOS DATOS CORRESPONDIENTES

plt.scatter(X_p,y_p)
plt.show()

### IMPLEMENTACION DE REGRESION POLINOMIAL

from sklearn.model_selection import train_test_split

# SEPARO LOS DATOS DE TRAIN EN ENTRENAMIENTO Y PRUEBA PARA PROBAR LOS ALGORITMOS

X_train_p,X_test_p,y_train_p,y_test_p = train_test_split(X_p,y_p,test_size = 0.2)

from sklearn.preprocessing import PolynomialFeatures

# DEFINO DE QUE GRADO VA A SER EL POLINOMIO

poli_reg = PolynomialFeatures(degree=2)

# TRANSFORMO LAS CARACTERISTICAS EN CARACTERISTICAS DE MAYOR GRADO

X_train_poli = poli_reg.fit_transform(X_train_p)
X_test_poli = poli_reg.fit_transform(X_test_p)

# DEFINO EL ALGORITMO QUE VOY A USAR

pr = linear_model.LinearRegression()

# ENTREO EL MODELO

pr.fit(X_train_poli,y_train_p)

# REALIZO UNA PREDICCION

Y_pred_pr = pr.predict(X_test_poli)
print(Y_pred_pr)
print()
# GRAFICO LOS DATOS JUNTO CON EL MODELO

plt.scatter(X_test_p,y_test_p)
plt.plot(X_test_p,Y_pred_pr,color = 'red',linewidth=3)
plt.show()

print('DATOS DEL MODELO REGRESION POLINOMIAL')
print()
print('Valor dependiente o coeficientes a:')
print(pr.coef_)
print()
print('Valor de la ordenada origen')
print(pr.intercept_)
print()
print('Presicion modelo')
print(pr.score(X_train_poli,y_train_p))