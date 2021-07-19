# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import seaborn as sns
import matplotlib.pyplot as plt

#%%
data = pd.read_csv("data.csv")

#column names
print(data.keys())

#%% dropping unncessary columns
data.drop(['Unnamed: 0','Id','groupId', 'matchId','matchType'],axis = 1, inplace = True)

#%% 
data.isna().sum()

#%%
subset = data.head(50000)
subset = pd.DataFrame(subset)
print(subset.shape)

#%%
subset.info()
subsets =subset.astype('float')
subsets.info()

#%%

corr = subsets.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

#%%
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

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

#%% feature importance

print(reg.coef_)
coefficients = reg.coef_
coefficients = pd.Series(coefficients)
results = pd.DataFrame()
results = results.assign(names = subset.keys(), coefficients = coefficients)

#%% A Second way to do the same thing
lasso_reg = linear_model.Lasso(alpha = .001, max_iter = 1000, tol = 0.01)
lasso_reg.fit(X_train, Y_train)

#%%
print(lasso_reg.score(X_test,Y_test))
print(lasso_reg.score(X_train,Y_train))


