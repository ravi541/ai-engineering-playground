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