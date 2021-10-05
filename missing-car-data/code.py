import pandas as pd
import numpy as np

car=pd.read_csv("/car.csv")
car

car["Make"].fillna("missing", inplace= True)
car["Colour"].fillna("missing", inplace = True)
car["Odometer (KM)"].fillna(car["Odometer (KM)"].mean(), inplace = True)
car["Doors"].fillna(car["Doors"].mean(), inplace = True)

car.isna().sum()

car.dropna(inplace=True)
car.isna().sum()

x=car.drop("Price", axis =1)
x

y=car["Price"]
y

x=pd.DataFrame(x)
x

y = pd.DataFrame(y)
y

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =18)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()

model.fit(x,y)

model.predict(x)

model.score(x,y)
