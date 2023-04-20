import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from matplotlib import pyplot as plot

#reader csv
csv = pd.read_csv("house/housing.csv")

#data separation
x, y = csv['RM'].values.reshape(-1, 1), csv['MEDV'].values.reshape(-1, 1) 
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state= 42)

#regression preparation 
regression = LinearRegression()

regression.fit(x_train,y_train)

#calc = lambda a,medv,b: a*medv+b #predictive calc
#print(calc(regression.coef_,2.0,regression.intercept_))

# print(regression.coef_)
# print(regression.intercept_)

y_predict = regression.predict(x_test)

modif = pd.DataFrame({'before': y_test.squeeze() ,"predict":y_predict.squeeze()})
print(modif)

mae = mean_absolute_error(y_test, y_predict)
mse = mean_squared_error(y_test, y_predict)
rmse = np.sqrt(mse)

print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')


#data to graphics 
csv.plot.scatter(x='RM', y='MEDV',title='relation between RM and MEDV' )
plot.plot(x_test, y_predict, color="red", linewidth=2)


plot.show()