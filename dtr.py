#decision tree regression

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
y.reshape(-1,1)

"""
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y)
"""

#svr fitting
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(x,y)

"""
#simple visualisation
plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
"""

#higher resolution vidualisation
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color='blue')



