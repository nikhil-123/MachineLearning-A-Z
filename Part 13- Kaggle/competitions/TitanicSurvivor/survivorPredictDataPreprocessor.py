#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 10:21:50 2018

@author: nikhil
"""

import pandas as pd
import numpy as  np
#----Data Pre-Processing Begins----------------------------------------------#

#importing daatsets--- test and train sets are different
dataset_train = pd.read_csv("../../datasets/titanicsurvivors/train.csv")
dataset_test = pd.read_csv("../../datasets/titanicsurvivors/test.csv")
X_test = dataset_test.iloc[:,[1,3,4,5,6,8,10]]
X_train = dataset_train.iloc[:,[2,4,5,6,7,9,11]]
y_train = dataset_train.iloc[:,[1]].values

#Function to check if a value in column is missing or not if it is then making it 'n'(not a cabin crew) if not then making it 'y'(cabin crew)
#def cabinCrew(x):
 #   if pd.notnull(x):
  #      return 'y'
   # else:
    #    return 'n'
#X_train['Cabin'] = X_train['Cabin'].map(cabinCrew)
#X_test['Cabin'] = X_test['Cabin'].map(cabinCrew)

#Using pandas get_dummies to get dummy variable for categorical values can't use label encoder because column has nan vaues which are float so cant compare float and string
X_train = pd.get_dummies(X_train, prefix=['_1'],columns=['Embarked'])
X_test = pd.get_dummies(X_test, prefix=['_1'],columns=['Embarked'])

#Turning dataframe to matrix because machine learning model only accepts matrix
X_train = X_train.values
X_test = X_test.values

#Handling categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
X_train[:,1] = label.fit_transform(X_train[:,1])
X_train[:,6] = label.fit_transform(X_train[:,6])
X_test[:,1] = label.fit_transform(X_test[:,1])
X_test[:,6] = label.fit_transform(X_test[:,6])

#Mean and most frequent strategy to deal with missing values in data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values=np.nan,strategy='mean')
imputer = imputer.fit(X_train[:,2:3])
X_train[:,2:3] = imputer.transform(X_train[:,2:3])
X_test[:,2:3] = imputer.transform(X_test[:,2:3])
imputer = Imputer(strategy="most_frequent")
X_train[:,3:6] = imputer.fit_transform(X_train[:,3:6])
X_test[:,3:6] = imputer.transform(X_test[:,3:6])

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train[:,2:3] =scaler.fit_transform(X_train[:,2:3])
X_train[:,5:6] = scaler.transform(X_train[:,5:6])
X_test[:,2:3] =scaler.transform(X_test[:,2:3])
X_test[:,5:6] = scaler.transform(X_test[:,5:6])
#-------------------Data Pre-processing end---------------------------#


df = pd.DataFrame(X_test)


