import pandas as pd
import numpy as np

cars=pd.read_csv("/cars.csv")
cars

X1=cars.iloc[:,2:4].values
X1

y1=cars.iloc[:,-1].values
y1

from sklearn.model_selection import train_test_split
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size = 0.2, random_state =2)

from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor()

model1.fit(X1_train,y1_train)

model1.predict(X1_test)

model1.score(X1_test,y1_test)
