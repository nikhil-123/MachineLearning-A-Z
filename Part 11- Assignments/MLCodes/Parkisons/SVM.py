#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 30 23:24:19 2018

@author: nikhil
"""

# Classification template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('../../DataSets/Parkinson-disease-data-updated')
X = dataset.iloc[:, [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20,21,22,23,24]].values
y = dataset.iloc[:, 18].values

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

#Splitting into training and test sets
from sklearn.model_selection import train_test_split
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)

#Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#Fitting classifier to training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=0)
classifier.fit(X_train,y_train)

#Predicting the test results
y_pred = classifier.predict(X_test)

#Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

#K-Cross validation
from sklearn.model_selection import cross_val_score
accuracy = cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=10)

#Grid search to find the best modeland parameter
from sklearn.model_selection import GridSearchCV
parameters = [{'C' : [10000,20000],'kernel' : ['rbf'],'gamma' : [0.005,0.01,0.02,0.04]},
{'C' : [10000,20000],'kernel' : ['linear']}
]
grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10,
                           n_jobs = -1)
grid_search =grid_search.fit(X_train,y_train)
grid_search.best_estimator_
grid_search.best_params_
grid_search.best_score_
