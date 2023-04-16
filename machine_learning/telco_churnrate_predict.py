#######################
# TELCO CHURN PREDICTON
#######################
# Business Problem: Development of the machine learning that can predict the customer who will leave the company.

####################################
# 1.IMPORT LIBRARY, PD OPTIONS, DATA
####################################
import itertools
import math
import warnings
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
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 300)
    pd.set_option("display.expand_frame_repr", True)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.2f" % x)


pd_options()

warnings.filterwarnings("ignore", category = DeprecationWarning)
warnings.filterwarnings("ignore", category = FutureWarning)
warnings.filterwarnings("ignore", category = UserWarning)

df = pd.read_csv("datasets/kaggle/telco_customer_churn.csv")


#################################
# 2. EDA (EXPLATORY DATA ANALYSIS
#################################
# Check the Data Specs, Examine the Data
def data_summary(dataframe):
    print("# # # [ D A T A F R A M E --> I N F O ] # # #")
    print(dataframe.info())
    print("# # # [ D A T A F R A M E --> M I S S I N G V A L U E S ] # # #")
    print(dataframe.isnull().sum())
    print("# # # [ D A T A F R A M E --> D U P L I C A T E D ] # # #")
    print(dataframe.duplicated().sum())
    print("# # # [ D A T A F R A M E --> D E S C R I B E ] # # #")
    print(dataframe.describe([.05, .25, .5, .75, .9, .99]))
    print("# # # [ D A T A F R A M E --> H E A D ] # # #")
    print(dataframe.head(10))


data_summary(df)

# Should be categoric(object) column, not integer.
df["SeniorCitizen"] = df["SeniorCitizen"].astype("object")

# Some observations are empty.
df.loc[df['TotalCharges'].str.contains(' ')][["TotalCharges"]]
# Replacing empty with NaN.
df.replace(" ", pd.np.nan, inplace = True)
# Should be float64 column, not object. This contains money info.
df["TotalCharges"] = df["TotalCharges"].astype("float64")
# Filling NaN with totalcharges formula.
df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"], inplace = True)
# Some columns are useless for predict the churn rate.
df.drop("customerID", axis = 1, inplace = True)


# Catching categoric, numeric columns.
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


# observations: 7043
# variables: 21
# categoric_cols: 17
# numeric_cols: 3
# categoric_but_cardinal: 1
# numeric_but_categoric: 1

# Mean of categorics and target variable.


# Checking outliers.
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(.25)
    quartile3 = dataframe[variable].quantile(.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


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
