# -*- coding: utf-8 -*-

import pandas as pd
from numpy import arange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.model_selection import RepeatedKFold
#%%
data = pd.read_csv("data.csv")

#column names
print(data.keys())

#%% dropping unncessary columns
data.drop(['Unnamed: 0','Id','groupId', 'matchId','matchType'],axis = 1, inplace = True)

#%%
subset = data.head(50000)
print(subset.shape)

#%%
subset = subset.dropna()
subsets =subset.astype('int')
subsets.info()

#%%
scaler = StandardScaler()
subsets = scaler.fit_transform(subsets)

#%%

X = subset.iloc[:,:24]
Y = subset.iloc[:,24]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=10)

#%%
reg = Lasso(alpha=0.001, max_iter = 5000)

#%%
reg.fit(X_train,Y_train)

#%%

print('Lasso Regression: R^2 score on training set', reg.score(X_train, Y_train)*100)
print('Lasso Regression: R^2 score on test set', reg.score(X_test, Y_test)*100)



