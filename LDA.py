# -*- coding: utf-8 -*-
"""
@author: Andrey Ferreira de Almeida
"""

#bibliotecas
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

#carrega a base de dados
iris = load_iris()

#documentacao do dataset da iris
#print(load_iris.__doc__)

#cria a base com as dimensões (X)
db = pd.DataFrame(iris.data, columns=iris.feature_names)

#insere na base base a classe (Y) [0=setosa, 1=versicolor, 2=virginica]
db['classe'] = iris.target

#Separa os dados da primeira classe e remove a coluna da classe
setosa = db[db['classe']==0]
del setosa['classe']

#Separa os dados da segunda classe e remove a coluna da classe
versicolor = db[db['classe']==1]
del versicolor['classe']

#Separa os dados da terceira classe e remove a coluna da classe
virginica = db[db['classe']==2]
del virginica['classe']

#matriz de covariancia para a primeira classe
cov_setosa = pd.DataFrame.cov(setosa)[:].values

#matriz de covariancia para a segunda classe
cov_versicolor = pd.DataFrame.cov(versicolor)[:].values

#matriz de covariancia para a terceira classe
cov_virginica = pd.DataFrame.cov(virginica)[:].values

#cria a matriz de zeros e soma as matrizes de covariancia das classes
sw = pd.DataFrame([[float(0) for a in range(4)] for b in range(4)])[:].values

for i in range(4):
    for j in range(4):
        sw[i][j] = cov_setosa[i][j] + cov_versicolor[i][j] + cov_virginica[i][j]
 
#crio a inversa da minha matriz sw
swi = np.linalg.inv(sw)
       
#cria uma referencia e remove a classe do dataset
iris = db.copy()
del iris['classe']

#matriz de covariancia no dataset principal
sb = pd.DataFrame.cov(iris)[:].values

#cria a matriz de zeros e multiplico a matriz inversa pela matriz SB pelo numero de classes
swisb = pd.DataFrame([[float(0) for a in range(4)] for b in range(4)])[:].values

for i in range(4):
    for j in range(4):   
        swisb[i][j] = swi[i][j] * ( 3 * sb[i][j])
        
#calculo os eigenvalues e os eigenvectors da matriz (4x4) acima
eigenvalues, eigenvectors = np.linalg.eig(swisb)

#plota o gráfico
cores = ['red', 'orange', 'blue']
classes = ['setosa', 'versicolor', 'virginica']

for i in range(3):
    
    base = db[db['classe'] == i]
    
    plt.scatter(
        x=base['sepal length (cm)']
        ,y=base['petal length (cm)']
        ,color=cores[i]
        ,marker='+'
        ,label=classes[i]
        )
    
plt.xlabel('sepal length (cm)')
plt.ylabel('petal length (cm)')

plt.title('petal length vs sepal length')

plt.legend(loc='lower right')

plt.show()