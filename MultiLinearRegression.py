import pandas as pd
data = pd.read_csv('50_Startups.csv')
X = data.iloc[:,:4].values
Y = data.iloc[:,4].values
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le = LabelEncoder()
X[:,3] = le.fit_transform(X[:,3])
ohe = OneHotEncoder(categorical_features = [3])
X = ohe.fit_transform(X).toarray()
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred = regressor.predict(X_test)
