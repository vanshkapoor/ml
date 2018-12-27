""" linear regression """

from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

x = np.array([[1,3],[2,4],[5,6],[7,8],[10,9]])
y = np.array([[4,6],[2,6],[9,7],[9,12],[1,4]])

regressor = LinearRegression()
regressor.fit(x,y)

y_predict = regressor.predict([[1,1]])
y_predict

plt.scatter(x,y,color='red')
plt.plot(x,regressor.predict(x),color='blue')
#plt.scatter(y_predict,color='yellow')
plt.show()




