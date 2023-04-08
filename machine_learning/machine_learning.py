import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

pd.set_option("display.float_format", lambda x: "%.2f" % x)

# Simple Linear Regression with OLS Using Scikit_Learn #

df = pd.read_csv("datasets/kaggle/advertising.csv")

X = df[["TV"]]
y = df[["sales"]]

# Model
reg_model = LinearRegression().fit(X, y)

# y_hat = b + w * x

# sabit (b - bias)
reg_model.intercept_[0]

# TV'nin katsayısı (w1)
reg_model.coef_[0][0]

# Prediction
# 150 birimlik TV harcaması satış beklentisi
reg_model.intercept_[0] + reg_model.coef_[0][0] * 150
df.describe().T

# Visualiation
g = sns.regplot(x = X, y = y, scatter_kws = {"color": "b", "s": 9}, ci = False, color = "r")

g.set_title(f"Moden Denklemi: Sales = {round(reg_model.intercept_[0], 2)}+TV*{round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Satış Sayısı")
g.set_xlabel("TV Harcamaları")
plt.xlim(-10, 310)
plt.ylim(bottom = 0)
plt.show()

# Prediction Success
# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)
y.mean(), y.std()

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-Square
reg_model.score(X, y)

# Multiple Linear Regression #
df = pd.read_csv("datasets/kaggle/advertising.csv")
X = df.drop("sales", axis = 1)
y = df[["sales"]]

# Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state = 1)

reg_model = LinearRegression().fit(X_train, y_train)

# sabit (b - bias)
reg_model.intercept_

# coefficients (w - weights)
reg_model.coef_

# Prediction
# Sales = 2.9 + TV[30] * .04 + Radio[10] * .17 + Newspaper[40] * .002
observations = [[30], [10], [40]]
observations = pd.DataFrame(observations).T
reg_model.predict(observations)

# Prediction Success
# RMSE Train
y_pred = reg_model.predict(X_train)
np.sqrt(mean_squared_error(y_train, y_pred))

# R-Square Train
reg_model.score(X_train, y_train)

# RMSE Test
y_pred = reg_model.predict(X_test)
np.sqrt(mean_squared_error(y_test, y_pred))

# If test error is less than train error, it means good

# R-Square Test
reg_model.score(X_test, y_test)

# 10 Katlı Cross Validation - CV RMSE
np.mean(np.sqrt(-cross_val_score(reg_model, X, y, cv = 10, scoring = "neg_mean_squared_error")))

# Simple Linear Regression with Gradient Descent from Stratch
