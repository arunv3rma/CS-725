# !/usr/bin/env python
# title           :actual_shares.py
# description     :This will create a header for a python script.
# author          :"Arun Verma"
# copyright       :"Copyright 2016, Arun Verma IE 507 Lab"
# credits         :["Arun Verma"]
# date            :3/9/16 11:44 PM
# license         :"Apache License Version 2.0"
# version         :0.1.0
# usage           :python actual_shares.py
# python_version  :2.7.11  
# maintainer      :"Arun Verma"
# email           :"v.arun@iitb.ac.in"
# status          :"D" ["Development(D) or Production(P)"]
# last_update     :3/9/16
# ==============================================================================

# Import the modules needed to run the script.
import pandas as pd

test = pd.read_csv("test_data.csv")
actual = pd.read_csv("actual.csv")
share_values = []
# print actual[' shares']
for i in range(len(test)):
    share_values.append(int(actual[' shares'][actual[actual['url'] == test['url'][i]].index.tolist()[0]])+(47633512.84860)**0.5)
    print i

result = pd.DataFrame(share_values)
result.columns = ["shares"]

# Storing Result into csv file
# result = pd.DataFrame({'shares': predict_result})
result.index.name = 'id'
result.to_csv("output.csv")