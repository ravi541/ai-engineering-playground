import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset=pd.read_csv(r"C:\Users\Raviteja\AVSCODE\ai-engineering-playground\MachieneLearning\MultiLinearRegression\Investment.csv")

x=dataset.iloc[:,:-1]
y=dataset.iloc[:,4]

#get dummies
x=pd.get_dummies(x,dtype=int)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predict
y_pred = regressor.predict(X_test)

c= regressor.intercept_
m=regressor.coef_
print(m)
x=np.append(arr=np.full((50,1),42467).astype(int), values=x,axis=1)


#Which attribute need to among 6 attriubute

import statsmodels.api as sm
X_opt=x[:,[0,1,2,3,4,5]]#ignoring 6 because if 4 r 5 any one is 1 then it is 0

#OrdinayLeastSquares
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#we have x4 as highest p value so ignoring it this is backward elimination

import statsmodels.api as sm
X_opt=x[:,[0,1,2,3,5]]#ignoring 6 because if 4 r 5 any one is 1 then it is 0

#OrdinayLeastSquares
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()



import statsmodels.api as sm
X_opt=x[:,[0,1,2,3]]#ignoring 6 because if 4 r 5 any one is 1 then it is 0

#OrdinayLeastSquares
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt=x[:,[0,1,2,3]]#ignoring 6 because if 4 r 5 any one is 1 then it is 0

#OrdinayLeastSquare
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()



import statsmodels.api as sm
X_opt=x[:,[0,1,3]]#ignoring 6 because if 4 r 5 any one is 1 then it is 0

#OrdinayLeastSquares
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()


import statsmodels.api as sm
X_opt=x[:,[0,1]]#ignoring 6 because if 4 r 5 any one is 1 then it is 0
#OrdinayLeastSquares
regressor_OLS= sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

# Why Are We Removing Highest P-Value?

# Goal:

# Keep only important features

# Remove useless features
# What is P-Value?

# P-value tells:

# How important is a feature in predicting Y?
# Rule

# Usually:

# P-value < 0.05
# Important Feature ✅

# P-value > 0.05
# Not Important 
# This is backward elimination process





















