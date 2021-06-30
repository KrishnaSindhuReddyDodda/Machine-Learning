import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_boston
dataset=load_boston()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['MEDV']=dataset.target
plt.plot(df['RM'],df['MEDV'],'bo')
x=df[['LSTAT' ,'RM']]
y=df['MEDV']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
model.coef_
model.intercept_
model.score(x_test,y_test)
from sklearn.datasets import load_breast_cancer
dataset1=load_breast_cancer()
df1=pd.DataFrame(dataset1.data,columns=dataset1.feature_names)
df1