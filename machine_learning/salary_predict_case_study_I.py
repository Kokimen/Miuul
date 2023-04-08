import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

from machine_learning.machine_learning import reg_model


def pd_options():
    pd.set_option("display.max_columns", 35)
    pd.set_option("display.max_rows", 35)
    pd.set_option('display.width', 300)
    pd.set_option("display.expand_frame_repr", True)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

df = pd.read_excel("datasets/miuul/linear_reg_salary_predict.xlsx")

df = df.set_axis(["Experience", "Salary"], axis = "columns")
# Linear Regression
# bias = 275, weight = 90 (y=b+wx)
X = df[["Experience"]]
y = df[["Salary"]]

# Visualization
plt.figure(figsize = (12, 6))
sns.pairplot(df, x_vars = ['Experience'], y_vars = ['Salary'], size = 7, kind = 'scatter')
plt.xlabel('Years')
plt.ylabel('Salary')
plt.title('Salary Prediction')
plt.show()

reg_model = LinearRegression().fit(X, y)

# sabit (b - bias)
reg_model.intercept_[0]

# TV'nin katsayısı (w1)
reg_model.coef_[0][0]

# Prediction
reg_model.intercept_ + reg_model.coef_[0]

# Prediction Success
# MSE
y_pred = reg_model.predict(X)
mean_squared_error(y, y_pred)

# RMSE
np.sqrt(mean_squared_error(y, y_pred))

# MAE
mean_absolute_error(y, y_pred)

# R-Square
reg_model.score(X, y)

# Visualization
g = sns.regplot(x = X, y = y, scatter_kws = {"color": "r", "s": 15}, ci = False, color = "black")

g.set_title(f"Salary = {round(reg_model.intercept_[0], 2)} + Experience * {round(reg_model.coef_[0][0], 2)}")
g.set_ylabel("Salary")
g.set_xlabel("Experience")
plt.xlim(0, 12)
plt.ylim(bottom = 0)
plt.show()
