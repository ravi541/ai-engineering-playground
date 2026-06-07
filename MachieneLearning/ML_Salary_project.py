# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 11:52:22 2026

@author: Raviteja
"""

# Open in spyder

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv(r"C:\Users\Raviteja\Downloads\Salary_Data.csv")

x=dataset.iloc[:,:-1] # take all rows and take all columns except last
y=dataset.iloc[:,-1] # take all rows and take last column


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0,train_size=0.8)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

comparision  = pd.DataFrame({'Actual':y_test,'Prediction':y_pred})
print(comparision)


plt.scatter(x_test, y_test, color = 'Red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary of employee based on experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.show()
