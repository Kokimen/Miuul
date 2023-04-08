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
# Cost function MSE
def cost(Y, b, w, X):
    m = len(Y)
    sse = 0

    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        sse += (y_hat - y) ** 2

    mse = sse / m
    return mse


# update_weights
def update_weights(Y, b, w, X, learning_rate):
    m = len(Y)
    b_deriv_sum = 0
    w_deriv_sum = 0
    for i in range(0, m):
        y_hat = b + w * X[i]
        y = Y[i]
        b_deriv_sum += (y_hat - y)
        w_deriv_sum += (y_hat - y) * X[i]
    new_b = b - (learning_rate * 1 / m * b_deriv_sum)
    new_w = w - (learning_rate * 1 / m * w_deriv_sum)
    return new_b, new_w


# train fonksiyonu
def train(Y, initial_b, initial_w, X, learning_rate, num_iters):
    print("Starting gradient descent at b = {0}, w = {1}, mse = {2}".format(initial_b, initial_w,
                                                                            cost(Y, initial_b, initial_w, X)))

    b = initial_b
    w = initial_w
    cost_history = []

    for i in range(num_iters):
        b, w = update_weights(Y, b, w, X, learning_rate)
        mse = cost(Y, b, w, X)
        cost_history.append(mse)

        if i % 100 == 0:
            print("iter={:d}    b={:.2f}    w={:.4f}    mse={:.4}".format(i, b, w, mse))

    print("After {0} iterations b = {1}, w = {2}, mse = {3}".format(num_iters, b, w, cost(Y, b, w, X)))
    return cost_history, b, w


df = pd.read_csv("datasets/kaggle/advertising.csv")

X = df["radio"]
Y = df["sales"]

# Hyperparameters
learning_rate = .001
initial_b = .001
initial_w = .001
num_iters = 10000

cost_history, b, w = train(Y, initial_b, initial_w, X, learning_rate, num_iters)
