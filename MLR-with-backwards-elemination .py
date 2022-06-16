

#1. kutuphaneler
from enum import auto
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#from rich.console import Console
#2. Veri Onisleme

#2.1. Veri Yukleme
veriler = pd.read_csv('odev_tenis.csv')

#veri on isleme

#encoder:  Kategorik -> Numeric

veriler2 = veriler.apply(LabelEncoder().fit_transform)

c = veriler2.iloc[:,:1]

#print(veriler)
#print(veriler2)

ohe = preprocessing.OneHotEncoder()
c=ohe.fit_transform(c).toarray()

havadurumu = pd.DataFrame(data = c, index = range(14), columns=['o','r','s'])
sonveriler = pd.concat([havadurumu,veriler.iloc[:,1:3]],axis = 1)
sonveriler = pd.concat([ veriler2.iloc[:,-2:], sonveriler], axis = 1)
print(sonveriler)

#verilerin egitim ve test icin bolunmesi
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(sonveriler.iloc[:,:-1],sonveriler.iloc[:,-1:],test_size=0.33, random_state=0)
#print(x_train)
#print(y_test)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#regressor.fit(x_train,y_train)


#y_pred = regressor.predict(x_test)


import statsmodels.regression.linear_model as sm
X_l = sonveriler.iloc[:,[0,1,2,3,4,5]].values
r_ols = sm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
#print(r.summary())
#print(X_l)
sonveriler = sonveriler.iloc[:,1:]
#print(sonveriler)

import statsmodels.regression.linear_model as lm 
X = np.append(arr = np.ones((14,1)).astype(int), values=sonveriler.iloc[:,:-1], axis=1 )

X_l = sonveriler.iloc[:,[0,1,2,3,4]].values
r_ols = lm.OLS(endog = sonveriler.iloc[:,-1:], exog =X_l)
r = r_ols.fit()
#print(r.summary())

x_train = x_train.iloc[:,1:]
x_test = x_test.iloc[:,1:]

regressor.fit(x_train,y_train)


y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
