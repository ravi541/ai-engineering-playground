import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv(r"C:\Users\Raviteja\Downloads\student_info.csv")
print(df)

df
df.info()
df.head()
df.tail()
df.describe()

plt.scatter(x=df.study_hours, y=df.student_marks)
plt.xlabel("Students study hours")
plt.ylabel("Students marks")
plt.title("Scatter plot of students study hours vs students marks")
plt.show()

df.isnull().sum()

df.mean()

df2=df.fillna(df.mean()) # filling the missing values in df2


df2.isnull().sum()



df2.head()
df2.info()

#Split dataset

x = df2.drop("student_marks",axis="columns")
y = df2.drop("study_hours",axis="columns")

print("Shape of X = ",x.shape)
print("Shape of Y = ",y.shape)


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

print("Shape of X_train = ", X_train.shape)
print("Shape of X_test = ", X_test.shape)
print("Shape of y_train = ", y_train.shape)
print("Shape of y_test = ", y_test.shape)


# Select a model and train it

from sklearn.linear_model import LinearRegression
lr=LinearRegression()

print(lr)

lr.fit(X_train, y_train)

lr.coef_
lr.intercept_

#test for 10 study
m =3.93
c=50.44
y = m*12 +c
y



y_pred = lr.predict(X_test)
y_pred


# comparision table

pd.DataFrame(np.c_[X_test,y_test,y_pred],columns=["study_hours","student_marks_original","student_marks_predicted"])

# Fine tune your model
