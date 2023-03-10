# RATING PRODUCTS

import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 200)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.2f" % x)

df = pd.read_csv("datasets/course_reviews.csv")
df.head(), df.shape

df["Rating"].value_counts()
df["Questions Asked"].value_counts()
df.groupby("Questions Asked").agg({"Questions Asked": "count",
                                   "Rating": "mean"})
df["Rating"].mean()

# Time-Based Weighted Average
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

current_date = pd.to_datetime("2021-02-10 0:0:0")

df["days"] = (current_date - df["Timestamp"]).dt.days

df.loc[df["days"] <= 30, "Rating"].mean()

df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean()

df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean()

df.loc[df["days"] > 180, "Rating"].mean()

df.loc[df["days"] <= 30, "Rating"].mean() * 28 / 100 + \
df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * 26 / 100 + \
df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * 24 / 100 + \
df.loc[df["days"] > 30, "Rating"].mean() * 22 / 100


def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return df.loc[df["days"] <= 30, "Rating"].mean() * w1 / 100 + \
           df.loc[(df["days"] > 30) & (df["days"] <= 90), "Rating"].mean() * w2 / 100 + \
           df.loc[(df["days"] > 90) & (df["days"] <= 180), "Rating"].mean() * w3 / 100 + \
           df.loc[df["days"] > 30, "Rating"].mean() * w4 / 100


time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)

# User-Based Weighted Average
df.groupby("Progress").agg({"Rating": "mean"})


def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return df.loc[df["Progress"] <= 10, "Rating"].mean() * w1 / 100 + \
           df.loc[(df["Progress"] > 10) & (df["Progress"] <= 90), "Rating"].mean() * w2 / 100 + \
           df.loc[(df["Progress"] > 45) & (df["Progress"] <= 180), "Rating"].mean() * w3 / 100 + \
           df.loc[df["Progress"] > 75, "Rating"].mean() * w4 / 100


user_based_weighted_average(df)


# Weighted Rating

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w / 100 \
           + user_based_weighted_average(dataframe) * user_w / 100


course_weighted_rating(df)

# SORTING PRODUCTS
df = pd.read_csv("datasets/product_sorting.csv")
df.info
