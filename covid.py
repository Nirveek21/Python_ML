import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('Book1.csv')
x=df[['Total confirmed cases of COVID-19 per million people']]
y=df['Total confirmed deaths due to COVID-19 per million people']
#print(x)
#from sklearn.model_selection import train_test_split
#x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25, random_state=42)
#print(x,y)
from sklearn import metrics
from sklearn.linear_model import LinearRegression

regressor=LinearRegression()   #create model
regressor.fit(x,y)  #train model

#y_prediction=regressor.predict(x_test)  #predict y
#error=np.sqrt(metrics.mean_squared_error(y_test,y_prediction))
death=regressor.predict([[20]])
#accuracy=(regressor.score(x_test,y_prediction))

#print("Error : " ,error)
print(death)
#print("Accuracy : ", accuracy)