import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset=pd.read_csv(r"C:\Users\Raviteja\AVSCODE\ai-engineering-playground\MachieneLearning\MultiLinearRegression\Investment.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

x=pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

c= regressor.intercept_
m=regressor.coef_

x=np.append(arr=np.full((50,1),42467).astype(int), values=x,axis=1)