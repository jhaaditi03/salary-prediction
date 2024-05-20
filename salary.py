import pandas as pd

#from google.colab import drive
#drive.mount('/content/drive')

import numpy as np

# if file not found
!file Salary.csv


# reading csv file
data=pd.read_csv("/content/sample_data/Salary.csv")

data.head()

data.columns

data.describe()

# if data is null or not
data.isnull()

data.isnull().any()

data.isnull().sum()

# map of salary and year of experience

import matplotlib.pyplot as plt

import seaborn as sns

plt.plot(data['Salary'],data['YearsExperience'])

data

# using sklearn

from sklearn.model_selection import train_test_split
x = data.drop('Salary',axis = 1)

x

y=data['Salary']

y.head()

#training dataset
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

from sklearn.linear_model import LinearRegression

L=LinearRegression()

L.fit(xtrain,ytrain)

y_pred=L.predict(xtest)

print(L.score(xtest, ytest))
