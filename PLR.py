#polynomial linear regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values


#linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

#polynomial regressor
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
poly_reg.fit(x_poly, y)

reg2 = LinearRegression()
reg2.fit(x_poly,y)



#plotting linear regression
plt.scatter(x,y,color='red')
plt.plot(x,y,color='blue')

#plotting polynomial linear regression
plt.scatter(x,y,color='red')
plt.plot(x,reg2.predict(poly_reg.fit_transform(x)),color='blue')







