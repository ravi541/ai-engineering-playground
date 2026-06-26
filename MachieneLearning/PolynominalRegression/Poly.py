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