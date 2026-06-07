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

from sklearn.metrics import r2_score

#r2 = r2_score(y_test, y_pred)
#print(r2)


m_slope=regressor.coef_
print(m_slope)

c_intercept= regressor.intercept_
print(c_intercept)

# Predict 12 years peson sal

y_12yr_exp = m_slope * 12 + c_intercept
print(y_12yr_exp)

# Predict 20 years peson sal
y_20yr_exp = m_slope * 20 + c_intercept
print(y_20yr_exp)

# bias
bias = regressor.score(x_train, y_train)
print('Bias score ',bias)

#variance
variance  = regressor.score(x_test, y_test)
print('variance score',variance)

# Stats concepts in ML - simple linear regression
#mean
dataset.mean()

dataset['Salary'].mean()
dataset['YearsExperience'].mean()

#median
dataset.median()
dataset['Salary'].median()
dataset['YearsExperience'].median()

#mode
dataset.mode()


#variance
dataset.var()
dataset['Salary'].var()
dataset['YearsExperience'].var()

#Standard deviation
dataset.std()
dataset['Salary'].std()
dataset['YearsExperience'].std()


#Coefficient of variation(cv)

from scipy.stats import variation

variation(dataset.values)

variation(dataset['Salary'])

#Correlation

dataset.corr() #correlation of entire dataframe

dataset['Salary'].corr(dataset["YearsExperience"]) 

#skewness

dataset.skew()

dataset['Salary'].skew()

# Standard Error

dataset.sem()

dataset['Salary'].sem()

# Z score
# for calculating z scire we have to import scipy 
import scipy.stats as stats

dataset.apply(stats.zscore)

stats.zscore(dataset['Salary'])

# Sum of sqaures of regression(SSR)

# ANOVA

y_mean = np.mean(y)
SSR = np.sum((y_pred-y_mean)**2)
print("SSR",SSR)


y=y[0:6]
SSE=np.sum((y-y_pred)**2)
print("SSE",SSE)

mean_total = np.mean(dataset.values)
#here dataset.to_numpy() will convert pandas dataframe
SST = np.sum((dataset.values-mean_total)**2)
print("SST",SST)


r_square = 1 - (SSR/SST)
print("R_square",r_square)



# save the trained model to disk

import pickle
filename='linear_regression_model.pkl'
with open(filename,'wb') as file:
    pickle.dump(regressor,file)
print("Model has been pickled and saved as linear_regression_model.pkl")




















