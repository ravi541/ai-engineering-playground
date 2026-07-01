import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

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
poly_reg = PolynomialFeatures(degree=5) #change degree to get more accurate data

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
print(svr_pred) # 175.7 score

# K- Nearesr Neighbour Regression

from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor()
knn_reg.fit(X,y)

knn_pred = knn_reg.predict([[6.5]])
print(knn_pred) # 168.00

# DecisionTreeAlgorithm

from sklearn.tree import DecisionTreeRegressor
dt_reg=DecisionTreeRegressor(criterion='poisson',splitter="best", max_depth=3)
dt_reg.fit(X,y)

dt_pred=dt_reg.predict([[6.5]])
print(dt_pred) #accuracy is 175k sal for 6.5 years exp person  

# Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=30,criterion='poisson',random_state=0,min_samples_split=3)
rf_reg.fit(X,y)

rf_pred=rf_reg.predict([[6.5]])
print(rf_pred)

#joblib for .pkl file to dump and load the model
import joblib

# Save the best model
joblib.dump(rf_reg, "salary_model.pkl")

print("Model saved successfully!")






































