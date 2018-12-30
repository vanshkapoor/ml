#SVR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
y.reshape(-1,1)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)


#svr fitting
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')

