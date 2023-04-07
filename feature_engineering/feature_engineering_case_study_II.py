import itertools
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import classification_report


def pd_options():
    pd.set_option("display.max_columns", 80)
    pd.set_option("display.max_rows", 80)
    pd.set_option('display.width', 400)
    pd.set_option("display.expand_frame_repr", True)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

df = pd.read_csv("datasets/kaggle/diabetes.csv")


# Check the Data Specs, Examine the Data
def data_summary(dataframe):
    print("# # # [ D A T A F R A M E --> I N F O ] # # #")
    print(dataframe.info())
    print("# # # [ D A T A F R A M E --> M I S S I N G V A L U E S ] # # #")
    print(dataframe.isnull().sum())
    print("# # # [ D A T A F R A M E --> D U P L I C A T E D ] # # #")
    print(dataframe.duplicated().sum())
    print("# # # [ D A T A F R A M E --> D E S C R I B E ] # # #")
    print(dataframe.describe([.05, .25, .5, .75, .9, .99]).T)
    print("# # # [ D A T A F R A M E --> H E A D ] # # #")
    print(dataframe.head(10))


data_summary(df)


# Outlier Function
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(.25)
    quartile3 = dataframe[variable].quantile(.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


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


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end = "\n\n\n")


for col in numeric_cols:
    target_summary_with_num(df, "Outcome", col)


# Analysis of Outliers
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[dataframe[col_name] > up_limit].any(axis = None):
        return (f"--> Uplimit Outlier, {low_limit, up_limit}")
    elif dataframe[dataframe[col_name] < low_limit].any(axis = None):
        return (f" --> Lowlimit Outlier, {low_limit, up_limit}")
    else:
        return (f" --> No Outlier, {low_limit, up_limit}")


for col in numeric_cols:
    print(col, check_outlier(df, col))

outliers_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in outliers_columns:
    replace_with_threshold(df, col)

for col in numeric_cols:
    print(col, check_outlier(df, col))


def grab_outliers(dataframe, col_name, index=False):
    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


for col in numeric_cols:
    print(col, grab_outliers(df, col))

outliers_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in outliers_columns:
    replace_with_threshold(df, col)

# Analysis of Missing Values
zero_values = [col for col in df.columns if 0 in df[col].values]

cant_zero_columns = ["Glucose", "BloodPressure", "BMI", "Insulin", "SkinThickness"]


def replace_zero_with_nan(df, cols):
    for col in cols:
        df[col].replace(0, np.nan, inplace = True)


replace_zero_with_nan(df, cant_zero_columns)


def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in df.columns if df[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending = False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending = False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis = 1, keys = ["n_miss", "ratio"])
    print(missing_df, end = "\n")
    if na_name:
        return na_columns


missing_values_table(df)


# Fill missing values
def filling_with_mean(dataframe, column):
    dataframe[column] = dataframe[column].fillna(dataframe[column].mean())


for col in cant_zero_columns:
    filling_with_mean(df, col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end = "\n\n\n")


for col in numeric_cols:
    target_summary_with_num(df, "Outcome", col)

# Analysis of Correlation
df_corr = df.corr().abs().unstack().sort_values(kind = "quicksort", ascending = False).reset_index()
df_corr.rename(columns = {"level_0": "Dependent", "level_1": "Independent", 0: 'Correlation'}, inplace = True)
df_corr[df_corr['Dependent'] == 'Outcome']

corr_feature_helper = df_corr.sort_values(by = "Correlation", ascending = False)[9::].drop_duplicates(["Correlation"])


# Feature Engineering
def categorize_age(df):
    df["New_Age"] = df["Age"].apply(lambda x: "Young-Adults" if 21 <= x <= 39
    else "Middle-Adults" if 40 <= x <= 59
    else "Old-Adults")
    return df


categorize_age(df)


def categorize_bmi(df):
    df["New_BMI"] = df["BMI"].apply(lambda x: "Underweight" if x <= 18.5
    else "Normal" if x <= 24.9
    else "Overweight" if x <= 29.9
    else "Obesite_1" if x <= 34.9
    else "Obesite_2" if x <= 39.9
    else "Obesite_3")
    return df


categorize_bmi(df)


def categorize_bmi(df):
    bins = [0, 18.5, 24.9, 29.9, 34.9, 39.9, float('inf')]
    labels = ["Underweight", "Normal", "Overweight", "Obesite_1", "Obesite_2", "Obesite_3"]
    df["New_BMI"] = pd.cut(df["BMI"], bins = bins, labels = labels)
    return df


def categorize_glucose(df):
    df["New_Glucose"] = df["Glucose"].apply(lambda x: "Normal" if x >= 80 and x <= 100
    else "Impaired" if x >= 101 and x <= 125
    else "Diabetic")
    return df


categorize_glucose(df)

categoric_cols, numeric_cols, categoric_but_cardinal = grab_col_names(df)


# Encoding
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe


binary_cols = [col for col in df.columns if df[col].dtype not in [int, float]
               and df[col].nunique() == 2]

for col in binary_cols:
    df = label_encoder(df, col)


# Rare Enconding
def rare_analyser(dataframe, target, categoric_cols):
    for col in categoric_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end = "\n\n\n")


# One Hot Encoding
rare_analyser(df, "Outcome", categoric_cols)


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


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns = categorical_cols, drop_first = drop_first)
    return dataframe


ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]

df = one_hot_encoder(df, ohe_cols, drop_first = True)

categoric_cols, numeric_cols, categoric_but_cardinal = grab_col_names(df)

useless_cols = [col for col in df.columns if df[col].nunique() == 2 and
                (df[col].value_counts() / len(df) < .01).any(axis = None)]

# Standarzation
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Create Model
y = df["Outcome"]  # --> dependent variable
X = df.drop(["Outcome"], axis = 1)  # --> other variables are independent variables

from sklearn.ensemble import RandomForestClassifier

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3, random_state = 17)
rf_model = RandomForestClassifier(random_state = 46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
print(classification_report(y_pred, y_test))


def plot_importance(model, features, num=len(X)):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize = (10, 10))
    sns.set(font_scale = 1)
    sns.barplot(x = "Value", y = "Feature", data = feature_imp.sort_values(by = "Value",
                                                                           ascending = False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()


plot_importance(rf_model, X_train)
