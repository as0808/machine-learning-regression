import pandas as pd
import numpy as np

start=pd.read_csv("/50_Startups.csv")
start

x=start.iloc[:,0:3].values
x

y=start.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =1)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()

model.fit(x_train,y_train)

model.predict(x_test)

model.score(x_test,y_test)
