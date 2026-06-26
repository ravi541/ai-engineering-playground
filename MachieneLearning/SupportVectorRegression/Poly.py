import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Raviteja\Downloads\emp_sal.csv")

X = dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

# linear regression visualization
plt.scatter(X, y, color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('Linear Regression graph')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()


linear_model_prediction=lin_reg.predict([[6.5]])
print(linear_model_prediction)

# Polynomial Model

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=6) #change degree to get more accurate data

X_poly = poly_reg.fit_transform(X)

poly_reg.fit(X_poly,y)

lin_reg2= LinearRegression()
lin_reg2.fit(X_poly, y)


plt.scatter(X, y, color='red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color='blue')
plt.title('accept or reject Polynomial Regression graph')
plt.xlabel('Postion level')
plt.ylabel('Salary')
plt.show()

poly_model_pred = lin_reg2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred)


# Support vector regression
# SVR
#Continue same program on SVR

from sklearn.svm import SVR
svr_reg= SVR(kernel='poly',degree=4,gamma='auto') # types of kernal rbf,poly,linear,sigmoid
svr_reg.fit(X, y)


svr_pred = svr_reg.predict([[6.5]])
print(svr_pred)










































