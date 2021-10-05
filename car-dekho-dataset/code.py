import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

car=pd.read_csv("/car_dekho_dataset.csv")
car

x=car.drop("Selling_Price", axis =1)
x

y=car["Selling_Price"]
y

x=pd.DataFrame(x)
x

y=pd.DataFrame(y)
y

x=pd.get_dummies(x)
x

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state =1)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor()

model.fit(x_train,y_train)

model.predict(x_test)

model.score(x_test,y_test)

#Relation
x=car.iloc[:,2:3]
a=car.iloc[:,1:2]
b=car.iloc[:,3:4]
c=car.iloc[:,4:5]

plt.scatter(x, a)
plt.show()

plt.scatter(x, b)
plt.show()

plt.scatter(x, c)
plt.show()
