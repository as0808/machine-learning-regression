import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sal=pd.read_csv("/Position_Salaries.csv")
sal

x=sal.iloc[:,1:2].values
x

y = sal.iloc[:,-1].values
y

x=pd.DataFrane(x)
x

y=pd.DataFrame(y)
y

from sklearn.svm import SVR
model=SVR()

model.fit(x,y)

model.predict(x)

model.score(x, y)

from sklearn.ensemble import RandomForestRegressor
model1=RandomForestRegressor()

model1.fit(x,y)

model1.predict(x)

model1.score(x,y)
