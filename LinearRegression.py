import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:,:-1].values
Y = data.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
y_predict = regressor.predict(X_test)
plt.scatter(X_test,Y_test,color = 'red')
plt.plot(X_test,regressor.predict(X_test),color = 'blue')
plt.show()