import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_breast_cancer 
dataset=load_breast_cancer()                                       #assigning breast_cancer data to dataset
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['MED']=dataset.target
plt.plot(df['mean perimeter'] ,df['MED'],'bo')
x=df[['mean texture' , 'mean perimeter']]                          #assigning "mean texture" and "mean perimeter" values to "x" variable
y=df['MED'] #assigning "MED" value to "y" variable
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
model.coef_                                                        #gives coefficient  values
model.intercept_                                                   #gives intercept value
model.score(x,y)                                                   #gives "r^2 value" used to find accuracy