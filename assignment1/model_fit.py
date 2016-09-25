# !/usr/bin/env python
# title           :model.py
# description     :This will create a header for a python script.
# author          :"Arun Verma"
# copyright       :"Copyright 2016, Arun Verma CS 725 Course"
# credits         :["Arun Verma"]
# date            :27/8/16 5:50 PM
# license         :"Apache License Version 2.0"
# version         :0.1.0
# usage           :python model.py
# python_version  :2.7.11  
# maintainer      :"Arun Verma"
# email           :"v.arun@iitb.ac.in"
# status          :"D" ["Development(D) or Production(P)"]
# last_update     :27/8/16
# ==============================================================================

# Import the modules needed to run the script.
import pandas as pd
from sklearn import cross_validation, ensemble
from sklearn.grid_search import GridSearchCV

####################################################################################
# Reading train data
train_data = pd.read_csv("train_data.csv")

# Data Cleasing
x = train_data.ix[ :, 1:60 ]

train = x[0:len(x)/2]
test = x[len(x)/2: len(x)]
# print x
y = train_data.ix[ :, 60 ]
# print y

####################################################################################
# # Model Tuning for Random Forest regression
kf_total = cross_validation.KFold(len(x), n_folds=10, shuffle=True, random_state=4)

# specify parameters and distributions to sample from
param_dict = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
              'max_depth': [4, 5],
              'max_features':[1.0, 0.3, 0.1]}

model=ensemble.RandomForestRegressor()
gs_cv = GridSearchCV(model, param_dict, scoring="accuracy", n_jobs=4)
gs_cv.fit(x, y)
best_estimator = gs_cv.best_estimator_
print 'Best hyperparameters:' + str(gs_cv.best_params_)
print 'Best estimator:', best_estimator

####################################################################################
####################################################################################
# # Model Tuning for Gradient
# kf_total = cross_validation.KFold(len(x), n_folds=10, shuffle=True, random_state=4)
#
# # specify parameters and distributions to sample from
# # param_dict = {'learning_rate': [0.1, 0.05, 0.02, 0.01],
# #               'max_depth': [4, 5],
# #               'max_features':[1.0, 0.3, 0.1]}
#
# param_dict = {'learning_rate': [0.1],
#               'max_depth': [4, 5]}
#
# model=ensemble.GradientBoostingRegressor(n_estimators=3000)
# gs_cv = GridSearchCV(model, param_dict, cv=5, scoring="accuracy", n_jobs=4)
# gs_cv.fit(x, y)
# best_estimator = gs_cv.best_estimator_
# print 'Best hyperparameters:' + str(gs_cv.best_params_)
# print 'Best estimator:', best_estimator

####################################################################################

# Build model
# Fitting model
# params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 1,
#           'learning_rate': 0.01, 'loss': 'ls'}
# model=ensemble.GradientBoostingRegressor(**params)

# model.fit(x,y)

####################################################################################
# Reading test data
test_data=pd.read_csv("test_data.csv")
# print test_data

# Predicting on test data
# predict_result = model.predict(test_data.ix[:, 1:60]
predict_result = best_estimator.predict(test_data.ix[:, 1:60])

# Storing Result into csv file
result = pd.DataFrame({'shares': predict_result})
result.index.name = 'id'
result.to_csv("output.csv")