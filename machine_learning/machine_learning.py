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

######################################################
# Diabetes Prediction with Logistic Regression
######################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing
# 3. Model & Prediction
# 4. Model Evaluation
# 5. Model Validation: Holdout
# 6. Model Validation: 10-Fold Cross Validation
# 7. Prediction for A New Observation


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, plot_roc_curve
from sklearn.model_selection import train_test_split, cross_validate


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

######################################################
# Exploratory Data Analysis
######################################################

df = pd.read_csv("datasets/diabetes.csv")

##########################
# Target'ın Analizi
##########################

df["Outcome"].value_counts()

sns.countplot(x = "Outcome", data = df)
plt.show()

100 * df["Outcome"].value_counts() / len(df)

##########################
# Feature'ların Analizi
##########################

df.head()

df["BloodPressure"].hist(bins = 20)
plt.xlabel("BloodPressure")
plt.show()


def plot_numerical_col(dataframe, numerical_col):
    dataframe[numerical_col].hist(bins = 20)
    plt.xlabel(numerical_col)
    plt.show(block = True)


for col in df.columns:
    plot_numerical_col(df, col)

cols = [col for col in df.columns if "Outcome" not in col]

# for col in cols:
#     plot_numerical_col(df, col)

df.describe().T

##########################
# Target vs Features
##########################

df.groupby("Outcome").agg({"Pregnancies": "mean"})


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end = "\n\n\n")


for col in cols:
    target_summary_with_num(df, "Outcome", col)

######################################################
# Data Preprocessing (Veri Ön İşleme)
######################################################
df.shape
df.head()

df.isnull().sum()

df.describe().T

for col in cols:
    print(col, check_outlier(df, col))

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

######################################################
# Model & Prediction
######################################################

y = df["Outcome"]

X = df.drop(["Outcome"], axis = 1)

log_model = LogisticRegression().fit(X, y)

log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)

y_pred[0:10]

y[0:10]

######################################################
# Model & Prediction
######################################################

y = df["Outcome"]

X = df.drop(["Outcome"], axis = 1)

log_model = LogisticRegression().fit(X, y)
log_model.intercept_
log_model.coef_

y_pred = log_model.predict(X)
y_pred[0:10]

y[0:10]


######################################################
# Model Evaluation
######################################################

def plot_confusion_matrix(y, y_pred):
    acc = round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot = True, fmt = ".0f")
    plt.xlabel('y_pred')
    plt.ylabel('y')
    plt.title('Accuracy Score: {0}'.format(acc), size = 10)
    plt.show()


plot_confusion_matrix(y, y_pred)

print(classification_report(y, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# ROC AUC
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.83939


######################################################
# Model Validation: Holdout
######################################################

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size = 0.20, random_state = 17)

log_model = LogisticRegression().fit(X_train, y_train)

y_pred = log_model.predict(X_test)
y_prob = log_model.predict_proba(X_test)[:, 1]

print(classification_report(y_test, y_pred))

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63

plot_roc_curve(log_model, X_test, y_test)
plt.title('ROC Curve')
plt.plot([0, 1], [0, 1], 'r--')
plt.show()

# AUC
roc_auc_score(y_test, y_prob)

######################################################
# Model Validation: 10-Fold Cross Validation
######################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

log_model = LogisticRegression().fit(X, y)

cv_results = cross_validate(log_model,
                            X, y,
                            cv = 5,
                            scoring = ["accuracy", "precision", "recall", "f1", "roc_auc"])

# Accuracy: 0.78
# Precision: 0.74
# Recall: 0.58
# F1-score: 0.65

# Accuracy: 0.77
# Precision: 0.79
# Recall: 0.53
# F1-score: 0.63


cv_results['test_accuracy'].mean()
# Accuracy: 0.7721

cv_results['test_precision'].mean()
# Precision: 0.7192

cv_results['test_recall'].mean()
# Recall: 0.5747

cv_results['test_f1'].mean()
# F1-score: 0.6371

cv_results['test_roc_auc'].mean()
# AUC: 0.8327

######################################################
# Prediction for A New Observation
######################################################

X.columns

random_user = X.sample(1, random_state = 45)
log_model.predict(random_user)

################################################
# KNN
################################################

# 1. Exploratory Data Analysis
# 2. Data Preprocessing & Feature Engineering
# 3. Modeling & Prediction
# 4. Model Evaluation
# 5. Hyperparameter Optimization
# 6. Final Model

import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

################################################
# 1. Exploratory Data Analysis
################################################

df = pd.read_csv("datasets/diabetes.csv")

df.head()
df.shape
df.describe().T
df["Outcome"].value_counts()

################################################
# 2. Data Preprocessing & Feature Engineering
################################################

y = df["Outcome"]
X = df.drop(["Outcome"], axis = 1)

X_scaled = StandardScaler().fit_transform(X)

X = pd.DataFrame(X_scaled, columns = X.columns)

################################################
# 3. Modeling & Prediction
################################################

knn_model = KNeighborsClassifier().fit(X, y)

random_user = X.sample(1, random_state = 45)

knn_model.predict(random_user)

################################################
# 4. Model Evaluation
################################################

# Confusion matrix için y_pred:
y_pred = knn_model.predict(X)

# AUC için y_prob:
y_prob = knn_model.predict_proba(X)[:, 1]

print(classification_report(y, y_pred))
# acc 0.83
# f1 0.74
# AUC
roc_auc_score(y, y_prob)
# 0.90

cv_results = cross_validate(knn_model, X, y, cv = 5, scoring = ["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

# 0.73
# 0.59
# 0.78

# 1. Örnek boyutu arttıralabilir.
# 2. Veri ön işleme
# 3. Özellik mühendisliği
# 4. İlgili algoritma için optimizasyonlar yapılabilir.

knn_model.get_params()

################################################
# 5. Hyperparameter Optimization
################################################

knn_model = KNeighborsClassifier()
knn_model.get_params()

knn_params = {"n_neighbors": range(2, 50)}

knn_gs_best = GridSearchCV(knn_model,
                           knn_params,
                           cv = 5,
                           n_jobs = -1,
                           verbose = 1).fit(X, y)

knn_gs_best.best_params_

################################################
# 6. Final Model
################################################

knn_final = knn_model.set_params(**knn_gs_best.best_params_).fit(X, y)

cv_results = cross_validate(knn_final,
                            X,
                            y,
                            cv = 5,
                            scoring = ["accuracy", "f1", "roc_auc"])

cv_results['test_accuracy'].mean()
cv_results['test_f1'].mean()
cv_results['test_roc_auc'].mean()

random_user = X.sample(1)

knn_final.predict(random_user)
