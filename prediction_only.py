import pandas as pd
from sklearn.linear_model import LinearRegression
df=pd.read_csv('Book1.csv')
x=df[['Total confirmed cases of COVID-19 per million people']]
y=df['Total confirmed deaths due to COVID-19 per million people']
regressor=LinearRegression()   #create model
regressor.fit(x,y)  #train model
death=regressor.predict([[100]])
print(death)
