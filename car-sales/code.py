import pandas as pd
import numpy as np

cars=pd.read_csv("/car_sales_extended.csv")
cars

x=cars.drop("Price", axis =1)
x

y=cars["Price"]
y

x=pd.DataFrame(x)
x

y=pd.DataFrame(y)
y

x=pd.get_dummies(x)
x

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x, y, test_size = 0.2, random_state =1)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()

model.fit(x_train, y_train)

model.predict(x_test)

model.score(x_test, y_test)
