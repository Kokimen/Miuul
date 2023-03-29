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

# Local Outlier Factor
df = sns.load_dataset("diamonds")
df = df.select_dtypes(include = ["float64", "int64"])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low, up = outlier_thresholds(df, "depth")

df[((df.depth < low) | (df.depth > up))].shape

clf = LocalOutlierFactor(n_neighbors = 20)
clf.fit_predict(df)
df_scores = clf.negative_outlier_factor_

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked = True, xlim = [0, 20], style = ".-")
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th].drop(axis = 0, labels = df[df_scores < th].index)

# Missing Values
# Catch the Missing Values
df = load_titanic()
# missing values numbers by columns
df.isnull().sum()
# total missing values numbers
df.isnull().sum().sum()
# show the at least one missing value
df[df.isnull().any(axis = 1)]
# show the no missing value
df[df.notnull().all(axis = 1)]
# show the missing values percentage
(df.isnull().sum() / df.shape[0] * 100)
# show the columns which contains missing values
na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]


# Check the missing values specs
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ["n_miss", "ratio"])
    print(missing_df, end = "\n")
    if na_name:
        return na_columns


missing_values_table(df)

# Solve the Missing Values Problem
# Remove
df.dropna()
# Fill with basic assignments
df.Age.fillna(df.Age.mean())
df.Age.fillna(df.Age.median())
df.Age.fillna(0)
# Only fill numerical columns
df = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis = 0)
# Only fill categoric columns
df.Embarked.fillna(df.Embarked.mode()[0])
df.Embarked.fillna("missing")
df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis = 0)
# Assignment with categorical breakdown
df.groupby("Sex")["Age"].mean()
df.Age.fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.loc[(df.Age.isnull()) & (df.Sex == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df.Age.isnull()) & (df.Sex == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

# This part is about machnine learning
# Predictive Assignment
categoric_cols, numeric_cols, categoric_but_cardinal = grab_col_names(df)
numeric_cols = [col for col in numeric_cols if col not in "PassengerId"]
dff = pd.get_dummies(df[categoric_cols + numeric_cols], drop_first = True)

# Standardization of Variables
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff), columns = dff.columns)

# Application of KNN
from sklearn.impute import KNNImputer

imputer = KNNImputer(n_neighbors = 5)
dff = pd.DataFrame(imputer.fit_transform(dff), columns = dff.columns)

# Data rollback to make sense of it
dff = pd.DataFrame(scaler.inverse_transform(dff), columns = dff.columns)

df["age_imputed_knn"] = dff[["Age"]]
df.loc[df.Age.isnull(), ["Age", "age_imputed_knn"]]

# Advance Analysis of Missing Values
msno.bar(df)
plt.show()

msno.matrix(df)
plt.show()

msno.heatmap(df)
plt.show()

# The Relationship of Missing Values with the Dependent Variable (Bonus)
na_columns = missing_values_table(df, True)


def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + "_NA_FLAG"] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end = "\n\n\n")


missing_vs_target(df, "Survived", na_columns)

# Label Encoding & Binary Encoding
df = load_titanic()
df.Sex.head()

le = LabelEncoder()
le.fit_transform(df.Sex)[0:5]
le.inverse_transform([0, 1])  # --> değerlerin karşılığını içinde tutar.


def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

df = load_application_train()

binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    label_encoder(df, col)

# One Hot Encoding
df.Embarked.value_counts()
# this is for see the nan values
# pd.get_dummies(df, columns=["Embarked"], dummy_na = True).head()
pd.get_dummies(df, columns = ["Sex", "Embarked"], drop_first = True).head()


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe


categoric_cols, numeric_cols, categoric_but_cardinal = grab_col_names(df)

ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

one_hot_encoder(df, ohe_cols).head()

# Rare Encoding
# Analysis of scarcity and abundance of variables
df = load_application_train()

categoric_cols, numeric_cols, categoric_but_cardinal = grab_col_names(df)


def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show()


for col in categoric_cols:
    cat_summary(df, col)

# Analysis of the relationship between rare categories and dependent variable
df.groupby("NAME_INCOME_TYPE")["TARGET"].mean()


def rare_analyser(dataframe, target, categoric_cols):
    for col in categoric_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")


rare_analyser(df, "TARGET", categoric_cols)


# Rare Enconder Function
def rare_encoder(dataframe, rare_percentage):
    temp_df = dataframe.copy()
    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == "O"
                    and (temp_df[col].value_counts() / len(temp_df) < rare_percentage).any(axis = None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_percentage].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), "Rare", temp_df[var])

    return temp_df


rare_encoder(df, .01)

# Feature Scaling
# StandardScaler: Klasik standartlaştırma. Ortalamayı çıkar, standart sapmaya böl. z = (x - u) / s
ss = StandardScaler()
df["Age_standard_scaler"] = ss.fit_transform(df[["Age"]])

# RobustScaler: Medyanı çıkar, iqr'a böl. (Outliersa karşı dayanıklı)
rs = RobustScaler()
df["Age_robuts_scaler"] = rs.fit_transform(df[["Age"]])

# Minimum Maximum Scaler
mms = MinMaxScaler()

df["Age_min_max_scaler"] = mms.fit_transform(df[["Age"]])
df.describe().T

age_cols = [col for col in df.columns if "Age" in col]


def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [.05, .1]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins = 20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()


for col in age_cols:
    num_summary(df, col, plot = True)

# Numeric to Categorical
# Binning
df["Age_qcut"] = pd.qcut(df.Age, 5)

# Feature Extraction
# Binary features: flag, bool, true-false
df["NEW_CABIN_BOOL"] = df.Cabin.notnull().astype("int")
df.groupby("NEW_CABIN_BOOL").agg({"Survived": "mean"})

# Relationship check between two variables, Proportion Test
from statsmodels.stats.proportion import proportions_ztest

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].sum(),
                                               df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].sum()],
                                      nobs = [df.loc[df["NEW_CABIN_BOOL"] == 1, "Survived"].shape[0],
                                              df.loc[df["NEW_CABIN_BOOL"] == 0, "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

df.loc[((df.SibSp + df.Parch) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df.SibSp + df.Parch) == 0), "NEW_IS_ALONE"] = "YES"
df.groupby("NEW_IS_ALONE").agg({"Survived": "mean"})

test_stat, pvalue = proportions_ztest(count = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].sum(),
                                               df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].sum()],
                                      nobs = [df.loc[df["NEW_IS_ALONE"] == "YES", "Survived"].shape[0],
                                              df.loc[df["NEW_IS_ALONE"] == "NO", "Survived"].shape[0]])

print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Text Features
# Letter Count
df["NEW_NAME_COUNT"] = df.Name.str.len()

# Word Count
df["NEW_NAME_WORD_COUNT"] = df.Name.apply(lambda x: len(str(x).split(" ")))

# Catch the Special Structures
df["NEW_NAME_DR"] = df.Name.apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
df.groupby("NEW_NAME_DR").agg({"Survived": ["mean", "count"]})

# Regex Features
df["NEW_TITLE"] = df.Name.str.extract(" ([A-Za-z]+).", expand = False)
df[["NEW_TITLE", "Survived", "Age"]].groupby(["NEW_TITLE"]).agg({"Survived": "mean", "Age": ["count", "mean"]})

# Date Features
df = pd.read_csv("datasets/miuul/course_reviews.csv")
df["Timestamp"] = pd.to_datetime(df.Timestamp, format = "%Y-%m-%d")
df["year"] = df.Timestamp.dt.year
df["month"] = df.Timestamp.dt.month
df["year_diff"] = date.today().year - df.Timestamp.dt.year  # --> try as df.year
df["month_diff"] = (date.today().year - df.Timestamp.dt.year) * 12 + date.today().month - df.Timestamp.dt.month
df["day_name"] = df.Timestamp.dt.day_name()

# Feature Interactions
df["NEW_AGE_PCLASS"] = df.Age * df.Pclass
df["NEW_FAMILY_SIZE"] = df.SibSp + df.Parch + 1
df.loc[(df.Sex == "male") & (df.Age <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df.Sex == "female") & (df.Age <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.groupby("NEW_SEX_CAT")["Survived"].mean()

# Feature Engineering & Data Preprocessing for Titanic
df.columns = [col.upper() for col in df.columns]
# 1. Feature Engineering
# cabin bool
df["NEW_CABIN_BOOL"] = df.CABIN.notnull().astype("int")
# name count
df["NEW_NAME_COUNT"] = df.NAME.str.len()
# name word count
df["NEW_NAME_WORD_COUNT"] = df.NAME.apply(lambda x: len(str(x).split(" ")))
# name dr
df["NEW_NAME_DR"] = df.NAME.apply(lambda x: len([x for x in x.split() if x.startswith("Dr")]))
# name title
df["NEW_TITLE"] = df.NAME.str.extract(" ([A-Za-z]+).", expand = False)
# family size
df["NEW_FAMILY_SIZE"] = df.SIBSP + df.PARCH + 1
# age pclass
df["NEW_AGE_PCLASS"] = df.AGE * df.PCLASS
# is alone
df.loc[((df.SIBSP + df.PARCH) > 0), "NEW_IS_ALONE"] = "NO"
df.loc[((df.SIBSP + df.PARCH) == 0), "NEW_IS_ALONE"] = "YES"
# age level
df.loc[(df.AGE < 18), "NEW_AGE_CAT"] = "young"
df.loc[(df.AGE >= 18) & (df.AGE < 56), "NEW_AGE_CAT"] = "mature"
df.loc[(df.AGE >= 56), "NEW_AGE_CAT"] = "senior"

# age * sex
df.loc[(df.SEX == "male") & (df.AGE <= 21), "NEW_SEX_CAT"] = "youngmale"
df.loc[(df.SEX == "male") & (df.AGE > 21) & (df.AGE <= 50), "NEW_SEX_CAT"] = "maturemale"
df.loc[(df.SEX == "male") & (df.AGE > 50), "NEW_SEX_CAT"] = "seniormale"
df.loc[(df.SEX == "female") & (df.AGE <= 21), "NEW_SEX_CAT"] = "youngfemale"
df.loc[(df.SEX == "female") & (df.AGE > 21) & (df.AGE <= 50), "NEW_SEX_CAT"] = "maturefemale"
df.loc[(df.SEX == "female") & (df.AGE > 50), "NEW_SEX_CAT"] = "seniorfemale"

categoric_cols, numeric_cols, categoric_but_cardinal = grab_col_names(df)
numeric_cols = [col for col in numeric_cols if "PASSENGERID" not in col]

# 2. Check the Outliers
for col in numeric_cols:
    print(col, check_outlier(df, col))

for col in numeric_cols:
    replace_with_threshold(df, col)

# 3. Check the Missing Values
missing_values_table(df)
df = df.drop("CABIN", axis = 1)
df = df.drop(["TICKET", "NAME"], axis = 1)

df["AGE"] = df["AGE"].fillna(df.groupby("NEW_TITLE")["AGE"].transform("median"))

df = df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis = 0)

# 4. Label Encoding
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)

# 4.1. Rare Encoding
rare_analyser(df, "SURVIVED", categoric_cols)
df = rare_encoder(df, .01)

# 4.2. One-Hot Encoding
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols)

categoric_cols, numeric_cols, categoric_but_cardinal = grab_col_names(df)
numeric_cols = [col for col in numeric_cols if "PASSENGERID" not in col]

rare_analyser(df, "SURVIVED", categoric_cols)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < .01).any(axis = None)]

df = df.drop(useless_cols, axis = 1)

# 5. Standard Scaler (we don't need, just for information)
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# 6. Creating Model
y = df.SURVIVED  # --> dependent variable
x = df.drop(["PASSENGERID", "SURVIVED"], axis = 1)  # --> other variables are independent variables

from sklearn.ensemble import RandomForestClassifier

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .3, random_state = 17)
rf_model = RandomForestClassifier(random_state = 46).fit(x_train, y_train)
y_pred = rf_model.predict(x_test)
accuracy_score(y_pred, y_test)
