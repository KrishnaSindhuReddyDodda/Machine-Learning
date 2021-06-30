import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.datasets import load_breast_cancer               #LOADS BREAST_CANCER DATASET FROM EXISTING ONE'S
dataset=load_breast_cancer()                                  #ASSGINING BREAST_CANCER DATA TO DATASET
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['MED']=dataset.target
x=df[['mean texture' , 'mean perimeter']]                     #ASSIGNING "MEAN_TEXTURE" AND "MEAN_PERIMETER" TO "X" VARIABLE
y=df['MED'] #"ASSIGNING "MED" TO "Y"VARIABLE
plt.plot(x,y,'bo') 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
model=LinearRegression()
model.fit(x,y)
y_pred=model.predict(x)
polynomial_features=PolynomialFeatures(degree=3)
x_poly=polynomial_features.fit_transform(x)
print(x_poly)
model2=LinearRegression()
model2.fit(x_poly,y)
y_poly_pred=model2.predict(x_poly)
polynomial_feature1=PolynomialFeatures(degree=3)
x_poly1=polynomial_feature1.fit_transform(x)
print(x_poly1)
model3=LinearRegression()
model3.fit(x_poly1,y)
y_poly_pred1=model3.predict(x_poly1)
plt.plot(x,y_poly_pred1,'bo')
print(model3.coef_)
print(model3.intercept_)
print(model.score(x,y)) 
print(model2.score(x_poly,y)) 
print(model3.score(x_poly1,y)) 
