import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("insurance_data.csv")
data

plt.scatter(data.age, data.bought_insurance,marker='+',color="red")
x_train, x_test, y_train, y_test = train_test_split(data[['age']], data.bought_insurance, test_size = 1/3, random_state = 0)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(x_train, y_train)

model.predict(x_test)

#plt.scatter(data.age, data.bought_insurance,marker='+',color="red")
plt.plot(x_test,model.predict(x_test))
