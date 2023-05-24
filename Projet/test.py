# Import de librairies
import numpy as np # Algèbre liéaire
import pandas as pd # Data rocessig
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from scipy.stats import spearmanr

# Import des donnée et exploration
X = pd.read_csv('Data_X.csv')
Y = pd.read_csv('Data_Y.csv')

# Vérifier s’il y a des valeurs manquantes dans les données.
# Retirer les lignes avec des valeurs manquantes, et les lignes de prix correspondantes
# Pour cela, on commence par concaténer les deux jeux de données
XY = pd.concat([X,Y], axis=1)
XY = XY.drop(['COUNTRY'], axis=1) # On supprime la colonne COUNTRY, j'ai pas compris pourquoi elle était là et plus elle est chiante parce que c'est des strings
# On retire les lignes avec des valeurs manquantes
XY = XY.dropna()
XY = XY.drop_duplicates()
XY = XY[(np.abs(stats.zscore(XY)) < 3).all(axis=1)]

# On affiche les valeurs moyennes et les écarts types
# On sépare ensuite les deux jeux de données
X = XY.iloc[:,:-2]
Y = XY.iloc[:,-2:]


# On commence par concaténer les deux jeux de données
XY = pd.concat([X,Y], axis=1)
# Nous normalisons d'abord les données
"""
for column in XY:
    XY[column]= (XY[column] - XY[column].mean()) / XY[column].std()"""
# On calcule la matrice de corrélation
corr = XY.corr()
# On affiche la matrice de corrélatio
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, random_state=101)
best_coef = 0
a = 0.1
best_alpha = 0
while a < 3:
    ridge = Ridge(alpha=a)
    ridge.fit(X_train[['DE_WINDPOW','DE_NET_EXPORT','DE_NET_IMPORT','DE_RESIDUAL_LOAD','FR_WINDPOW']],pd.DataFrame(Y_train["TARGET"]))
    prediction = ridge.predict(pd.DataFrame(X_test[['DE_WINDPOW','DE_NET_EXPORT','DE_NET_IMPORT','DE_RESIDUAL_LOAD','FR_WINDPOW']]))
    corr_ridge, pval_ridge = spearmanr(Y_test['TARGET'],prediction)
    if corr_ridge > best_coef:
        best_coef = corr_ridge
        best_alpha = a
    a += 0.001

print(best_coef, best_alpha)