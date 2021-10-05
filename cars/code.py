import pandas as pd
import numpy as np

cars=pd.read_csv("/cars.csv")
cars

x=cars.iloc[:,2:4].values
x

y=cars.iloc[:,-1].values
y

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =2)

from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor()

model1.fit(x_train,y_train)

model1.predict(x_test)

model1.score(x_test,y_test)
