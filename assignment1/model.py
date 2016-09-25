# !/usr/bin/env python
# title           :model.py
# description     :This will create a header for a python script.
# author          :"Arun Verma"
# copyright       :"Copyright 2016, Arun Verma, CS 725"
# credits         :["Arun Verma"]
# date            :27/8/16 5:50 PM
# license         :"Apache License Version 2.0"
# version         :0.1.0
# usage           :python model.py
# python_version  :2.7.11  
# maintainer      :"Arun Verma"
# email           :"v.arun@iitb.ac.in"
# status          :"D" ["Development(D) or Production(P)"]
# last_update     :3/9/16
# ==============================================================================

# Import the modules needed to run the script.
from sys import argv
import pandas as pd
import numpy as np
from sklearn import cross_validation, linear_model, metrics, svm

# ############################## Reading train data ##################################

train_data = pd.read_csv("train_data.csv")

# Data Cleansing, Removing 'url' column from data
# First 60 columns as independent features except first column url
X = train_data.ix[:, 1:60]
# shares columns as dependent features which need to predicted using given data
Y = train_data.ix[:, 60]


# Creating different set of train and test data for cross validation of model
X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.30, random_state=10)

# ############################# Modeling Data ##########################################
model_type = ''
if len(argv) <= 1:
    print "Model is Linear Regression"
    model = linear_model.LinearRegression()

    # Features selection using RFE
    # rfe = RFE(model, n_features_to_select=1)
    # rfe.fit(X_train, Y_train)
    # print rfe.ranking_
    # print "Features sorted by their rank:"
    # print sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), list(X.columns.values)))


    # Fitting model with train data
    model.fit(X_train, Y_train)

    # Predict on the test data
    Y_predicted = model.predict(X_test)

    # Compute the root-mean-square
    rms = np.sqrt(metrics.mean_squared_error(Y_test, Y_predicted))
    print "Root Mean Square Error =", rms
    model.fit(X, Y)


else:
    model_type = argv[1]

    if model_type == 'L' or model_type == 'l':
        print "Model is Lasso Regression"
        model = linear_model.Lasso()
        model.fit(X_train, Y_train)

        # Predict on the test data
        Y_predicted = model.predict(X_test)

        # Compute the root-mean-square
        rms = np.sqrt(metrics.mean_squared_error(Y_test, Y_predicted))
        print "Root Mean Square Error =", rms

    elif model_type == "R" or model_type == "r":
        print "Model is Ridge Regression"

        model = linear_model.Ridge()
        model.fit(X_train, Y_train)

        # Predict on the test data
        Y_predicted = model.predict(X_test)

        # Compute the root-mean-square
        rms = np.sqrt(metrics.mean_squared_error(Y_test, Y_predicted))
        print "Root Mean Square Error =", rms

    elif model_type == "S" or model_type == "s":
        print "Model is Support Vector Regression"
        model = svm.SVR()
        model.fit(X_train, Y_train)

        # Predict on the test data
        Y_predicted = model.predict(X_test)

        # Compute the root-mean-square
        rms = np.sqrt(metrics.mean_squared_error(Y_test, Y_predicted))
        print "Root Mean Square Error =", rms

    else:
        print "PLease enter valid model type."
        exit()

# ############################ Test data Reading and prediction ################################
# Reading test data
test_data=pd.read_csv("test_data.csv")

# Predicting on test data
predict_result = model.predict(test_data.ix[:, 1:60])

# Storing Result into csv file
result = pd.DataFrame({'shares': predict_result})
result.index.name = 'id'
result.to_csv("output.csv")