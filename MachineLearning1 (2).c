import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_breast_cancer
dataset=load_breast_cancer()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df
df['MED']=dataset.target
plt.plot(df['mean perimeter'] ,df['MED'],'bo')
x=df[['mean texture' , 'mean perimeter']]
y=df['MED']
#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=2020)
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x_train,y_train)
model.coef_
model.intercept_
model.score(x_test,y_test)