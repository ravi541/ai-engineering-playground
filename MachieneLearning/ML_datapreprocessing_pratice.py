# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 15:01:49 2026

@author: Raviteja
"""

import numpy as np #array

import matplotlib.pyplot as plt

import pandas as pd

dataset = pd.read_csv(r'C:\Users\Raviteja\Downloads\14th - ML\14th - ML\5. Data preprocessing\Data.csv')

X = dataset.iloc[:, :-1].values

y=dataset.iloc[:,3].values


from sklearn.impute import SimpleImputer # Spyder 4


imputer = SimpleImputer()

imputer = imputer.fit(X[:,1:3])

X[:,1:3]=imputer.transform(X[:,1:3])


# change to median strategy


imputer = SimpleImputer(strategy='median')

imputer = imputer.fit(X[:,1:3])

X[:,1:3]=imputer.transform(X[:,1:3])

# Change to mode strategy


imputer = SimpleImputer(strategy='most_frequent')
imputer = imputer.fit(X[:,1:3])

X[:,1:3]=imputer.transform(X[:,1:3])

from sklearn.preprocessing import LabelEncoder

labelencoder_x= LabelEncoder()

labelencoder_x.fit_transform(X[:,0])

X[:,0]=labelencoder_x.fit_transform(X[:,0])


labelencoder_y =LabelEncoder()

y = labelencoder_y.fit_transform(y)


from sklearn.model_selection import train_test_split


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,train_size=0.7,random_state=0)























