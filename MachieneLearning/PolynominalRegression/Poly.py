import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Raviteja\Downloads\emp_sal.csv")

X = dataset.iloc[:,1:2].values
y= dataset.iloc[:,2].values