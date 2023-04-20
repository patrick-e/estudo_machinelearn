from matplotlib import pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import  mean_squared_error, r2_score

diab_x, diab_y = datasets.load_diabetes(return_X_y=True)

diab_x = diab_x[:,np.newaxis,2]

diab_x_train = diab_x[:-20]
diab_x_test = diab_x[-20:]

diab_y_train = diab_y[:-20]
diab_y_test = diab_y[-20:]

regrecreat = linear_model.LinearRegression()

regrecreat.fit(diab_x_train,diab_y_train)

diab_y_pred = regrecreat.predict(diab_x_test)

print("Coefficients: \n", regrecreat.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diab_y_test, diab_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diab_y_test, diab_y_pred))

plt.scatter(x=diab_x_test, y=diab_y_test, color="black")
plt.np
plt.plot(diab_x_test, diab_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.title('regressao linear de diabetes usando doc do sklearn')

plt.show()