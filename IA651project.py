#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 08:36:08 2021

IA 651 Project

@author: Cody Cox
"""
#%%
#Import modules

import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import numpy as np
from numpy import mean
from numpy import std
from numpy import absolute
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Load dataset
md = pd.read_csv("pubgdata.csv")
# columns names
print(md.keys())
# drop columns we won't use
md.drop(['Unnamed: 0','Id','groupId','matchId','matchType'], axis=1, inplace=True)
# create a subset of data
subset = md[::1000]
#%%

# Summarize shape
print(subset.shape)
# Summarize first 5 lines of data
subset.head()

#%%
target_column = ['winPlacePerc']
predictors = list(set(list(subset.columns))-set(target_column))
subset[predictors] = subset[predictors]/subset[predictors].max()
subset.describe()
#%%
X = subset[predictors].values
y = subset[target_column].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=40)
print(X_train.shape); print(X_test.shape)
#%% From tutorial on https://www.pluralsight.com/guides/linear-lasso-ridge-regression-scikit-learn
# Define model
rr = Ridge(alpha=.01)
rr.fit(X_train, y_train) 
pred_train_rr= rr.predict(X_train)
print(np.sqrt(mean_squared_error(y_train,pred_train_rr)))
print(r2_score(y_train, pred_train_rr))

pred_test_rr= rr.predict(X_test)
print(np.sqrt(mean_squared_error(y_test,pred_test_rr))) 
print(r2_score(y_test, pred_test_rr))

# Outputs
# 0.12100612256456438
# 0.8363935078547146
# 0.14031565301551008
# 0.781658136898814
#%% From tutorial on https://machinelearningmastery.com/ridge-regression-with-python/
# define model evaluation method
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

# evaluate model
scores = cross_val_score(model, X, y, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
# force scores to be positive
scores = absolute(scores)
print('Mean MAE: %.3f (%.3f)' % (mean(scores), std(scores)))