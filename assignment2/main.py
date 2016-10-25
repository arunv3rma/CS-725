# !/usr/bin/env python
# title           :model
# description     :This will create a header for a python script.
# author          :"Arun Verma"
# copyright       :"Copyright 2016, Arun Verma, CS-725 Course"
# credits         :["Arun Verma"]
# date            :23/10/16 11:41 AM
# license         :"Apache License Version 2.0"
# version         :0.1.0
# usage           :python model.py
# python_version  :2.7.11  
# maintainer      :"Arun Verma"
# email           :"v.arun@iitb.ac.in"
# status          :"P" ["Development(D) or Production(P)"]
# last_update     :25/10/16
# ==============================================================================

# Import the modules needed to run the script.
import numpy as np


# ######################### Activation Functions and their derivative ##########################
# Return Sigmoid activation function value
def get_sigmoid_value(x):
    return 1.0 / (1.0 + np.exp(-x))


# Return Sigmoid derivative used for weights' update
def get_sigmoid_derivative(x):
    return x * (1 - x)


# Return Softmax activation function value
def get_softmax_value(x):
    return np.exp(x) / np.sum(np.exp(x))


# Return Softmax derivative used for weights' update
def get_softmax_derivative(x):
    return x * (1 - x)


# ############################## Neural Network functions ##################################
# Return next mini batch
def get_next_mini_batch(x_features, y_lables, batch_no, batch_size, total_batches):
    if batch_no != total_batches:
        x_mini_batch = x_features[batch_no * batch_size:(batch_no + 1) * batch_size, :]
        y_mini_batch = y_lables[batch_no * batch_size:(batch_no + 1) * batch_size, :]
    else:
        x_mini_batch = x_features[batch_no * batch_size:, :]
        y_mini_batch = y_lables[batch_no * batch_size:, :]

    return x_mini_batch, y_mini_batch


# Training neural network - a full iteration with sigmoid function
def train(x_train, y_label, in_hid_layer_weights, hid_out_layer_weights, eta):
    # Forward propagation
    input_values = x_train
    hidden_values = get_sigmoid_value(np.dot(input_values, in_hid_layer_weights))
    output_values = get_sigmoid_value(np.dot(hidden_values, hid_out_layer_weights))

    # Calculating mean error used in Gradient Decent
    output_values_error = y_label - output_values
    avg_iteration_error = np.mean(np.abs(output_values_error))

    # Backward propagation
    output_values_delta = output_values_error * get_sigmoid_derivative(output_values)
    hidden_values_error = output_values_delta.dot(hid_out_layer_weights.T)
    hidden_values_delta = hidden_values_error * get_sigmoid_derivative(hidden_values)

    # Weights updating
    hid_out_layer_weights += eta * hidden_values.T.dot(output_values_delta)
    in_hid_layer_weights += eta * input_values.T.dot(hidden_values_delta)

    return in_hid_layer_weights, hid_out_layer_weights, avg_iteration_error


# ############################## Reading train data ##################################
# Using Numpy reading data from train.csv
print "Reading Train Data"
train_data = np.genfromtxt('train.csv', delimiter=',', skip_header=1)

print "Selecting Features"
# Separating feature from data
features = train_data[:, :-1]

# Adding bias to features by adding one column with value 1
X = np.c_[features, np.ones(len(features))]

print "Selecting labels"
# Separating labels from data
labels = train_data[:, 10]

print "Doing one hot encoding"
# One hot encoding
y = np.zeros((len(X), 9))
for i in range(0, len(X)):
    y[i, int(labels[i])] = 1.0

# print "Normalizing data"
# # Normalize data: Centering(mean=0) and Whitening(variance=1)
# X = (X - np.mean(X)) / np.std(X)
# y = (y - np.mean(y)) / np.std(y)

# ############################## Training Model ##################################
print "Start training"

# randomly initialize our small weights
in_hid_weights = 0.01 * np.random.randn(len(X[0]), 60)
hid_out_weights = 0.01 * np.random.randn(60, len(y[0]))

# Mini batches of train data for handling large gradient matrices
batchSize = 1600
batches = len(X) / batchSize
if len(X) % batchSize != 0:
    batches += 1

print "Batches: ", batches
print "Model learning started"

# learning rate, Note: Given l_rate gave best result
l_rate = 0.0035
# Maximum number of iterations on full data
max_epoch = 5000

previous_epoch_error = 0
no_change_count = 0
l_rate_change_count = 0


# ############################## Learning Start ##################################
for epoch in xrange(max_epoch):
    current_epoch_error = 0

    for iteration in xrange(batches):
        x_batch, y_batch = get_next_mini_batch(X, y, iteration, batchSize, batches)
        in_hid_weights, hid_out_weights, iteration_error = train(x_batch, y_batch, in_hid_weights, hid_out_weights,
                                                                 l_rate)
        current_epoch_error += iteration_error

    print "Error in epoch " + str(epoch) + " : " + str(current_epoch_error)
    error_changed = abs(current_epoch_error - previous_epoch_error)
    print "Change in error: " + str(error_changed) + "\n"

    previous_epoch_error = current_epoch_error

    # Dynamic learning rate: decrease learning rate for
    # if error_changed < 1.0e-04:
    #     no_change_count += 1
    #     l_rate -= 0.0001
    #     if no_change_count >= 10:
    #         l_rate -= 0.001
    #         l_rate_change_count += 1
    #         no_change_count = 0
    #
    #         if l_rate_change_count >= 100:
    #             print "Not changing much error"
    #             print "No. of epoch: " + str(epoch)
    #             break


# ############################ Test data Reading and prediction ################################
print "Start prediction on test data"
# Reading test data
test_data = np.genfromtxt('test1.csv', delimiter=',', skip_header=1)

# Adding bias in test data
test = np.c_[test_data[:, 1:], np.ones(len(test_data))]

# Predicting on test data
hidden_layer_values = get_sigmoid_value(np.dot(test, in_hid_weights))
output_layer_values = get_sigmoid_value(np.dot(hidden_layer_values, hid_out_weights))

# Storing Result into csv file
output = np.empty((0, 2), int)
for i in range(0, len(output_layer_values)):
    row = np.array([i, output_layer_values[i].argmax()])
    output = np.vstack((output, row))

np.savetxt('output.csv', output, header="id,CLASS", delimiter=',', fmt='%i', comments='')
print "Done!, Result stored in output.csv"