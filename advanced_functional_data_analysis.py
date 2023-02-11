import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()


def check_df(dataframe):
    print(dataframe.shape)
    print(dataframe.dtypes)
    print(dataframe.isnull().sum())
    print(dataframe.describe().T)


check_df(df)

df["sex"].unique()
df["sex"].nunique()

categoric_cols = [col for col in df.columns if str(df[col].dtypes) in
                  ["category", "object", "bool"]]

numeric_but_categoric = [col for col in df.columns if df[col].nunique() < 10
                         and df[col].dtypes
                         in ["int64", "float64"]]

# bir kategorik değişkenin çok fazla sınıfının olması bilgi taşımadığı anlamına gelir
categoric_but_cardinal = [col for col in df.columns if df[col].nunique() > 20
                          and str(df[col].dtypes)
                          in ["category", "object"]]

categoric_cols += numeric_but_categoric

categoric_cols = [col for col in categoric_cols if col not in categoric_but_cardinal]

df[categoric_cols].nunique()

[col for col in df.columns if col not in categoric_cols]


def categoric_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))


categoric_summary(df, "sex")

for col in categoric_cols:
    categoric_summary(df, col)