# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Fitting the Linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# Fitting and tranforming polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(4)
X_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly,y)
lin_reg2.predict(poly_reg.fit_transform(6.5))
#visualizing linear regression result
plt.scatter(X,y,color="red")
plt.plot(X,lin_reg.predict(X),color="blue")
plt.title("Truth Or Bluff: Linear")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.show

#visualizing poly regression result
plt.scatter(X,y,color="red")
plt.title("Truth Or Bluff: Poly")
plt.xlabel("Position Level")
plt.ylabel("Salary")
plt.plot(X,lin_reg2.predict(X_poly), color="blue")
plt.show