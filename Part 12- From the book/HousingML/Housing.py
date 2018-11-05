#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 15 10:35:09 2018

@author: nikhil
"""

import pandas as pd
import matplotlib.pyplot as plt
housing = pd.read_csv("../datasets/housing.csv")
housing["ocean_proximity"].value_counts()
hous_desc=housing.describe()

housing.hist(bins=50, figsize=(20,15))

