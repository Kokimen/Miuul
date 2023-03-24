# Feature Engineering
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import math
from datetime import date
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


def pd_options():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 300)
    pd.set_option("display.expand_frame_repr", True)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()


def load_application_train():
    data = pd.read_csv("datasets/miuul/application_train.csv")
    return data


df = load_application_train()
df.head()


def load_titanic():
    data = pd.read_csv("datasets/kaggle/titanic.csv")
    return data


df = load_titanic()
df.head()

# 1. Catching Outliers
sns.boxplot(x = df["Age"])
plt.show()

q1 = df.Age.quantile(.25)
q3 = df.Age.quantile(.75)
iqr = q3 - q1
up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

# Show Outliers Values
df[(df["Age"] < low) | (df["Age"] > up)]

# Find Is There Any Outlier
df[(df["Age"] < low) | (df["Age"] > up)].any(axis = None)


# Function the Outliers Checker and Replacer
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(.25)
    quartile3 = dataframe[variable].quantile(.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe[(dataframe[variable] > up_limit), variable] = up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis = None):
        return True
    else:
        return False


dff = load_application_train()


def grab_col_names(dataframe, categoric_threshold=10, cardinal_threshold=20):
    # categoric_cols, categoric_but_cardinal
    categoric_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    numeric_but_categoric = [col for col in dataframe.columns if dataframe[col].nunique() < categoric_threshold and
                             dataframe[col].dtypes != "O"]
    categoric_but_cardinal = [col for col in dataframe.columns if dataframe[col].nunique() > cardinal_threshold and
                              dataframe[col].dtypes == "O"]
    categoric_cols = categoric_cols + numeric_but_categoric
    categoric_cols = [col for col in categoric_cols if col not in categoric_but_cardinal]

    # numeric_cols
    numeric_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    numeric_cols = [col for col in numeric_cols if col not in numeric_but_categoric]

    print(f"observations: {dataframe.shape[0]}")
    print(f"variables: {dataframe.shape[1]}")
    print(f"categoric_cols: {len(categoric_cols)}")
    print(f"numeric_cols: {len(numeric_cols)}")
    print(f"categoric_but_cardinal: {len(categoric_but_cardinal)}")
    print(f"numeric_but_categoric: {len(numeric_but_categoric)}")
    return categoric_cols, numeric_cols, categoric_but_cardinal


categoric_cols, numeric_cols, categoric_but_cardinal = grab_col_names(df)

numeric_cols = [col for col in numeric_cols if col not in "PassengerId"]

for col in numeric_cols:
    print(col, check_outlier(df, col))


# Reach the Outliers
def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


grab_outliers(df, "Age", True)

# Solve the Outliers Problem
low, up = outlier_thresholds(df, "Fare")
df[~((df.Fare < low) | (df.Fare > up))].shape


def remove_outliers(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


for col in numeric_cols:
    new_df = remove_outliers(df, col)

# Re-Assignment with Thresholds
low, up = outlier_thresholds(df, "Fare")

df[(df.Fare < low) | (df.Fare > up)]["Fare"]

# This is make re-assignment
df.loc[(df.Fare > up), "Fare"] = up

df.loc[(df.Fare < low), "Fare"] = low


def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in numeric_cols:
    print(col, replace_with_threshold(df, col))

# Summary
df = load_titanic()
outlier_thresholds(df, "Age")
check_outlier(df, "Age")
grab_outliers(df, "Age", index = True)
remove_outliers(df, "Age")
replace_with_threshold(df, "Age")
