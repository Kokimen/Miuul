# RATING PRODUCTS

import math

import pandas as pd
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
    return time_based_weighted_average(dataframe) * time_w / 100 + \
           user_based_weighted_average(dataframe) * user_w / 100


course_weighted_rating(df)

# SORTING PRODUCTS
df = pd.read_csv("datasets/product_sorting.csv")
df.info

# Sorting by Rating
df.sort_values(by = "rating", ascending = False).head(28)

# Sorting by Comment or Purchase Count
df.sort_values(by = "purchase_count", ascending = False)
df.sort_values(by = "commment_count", ascending = False)

# Sorting by Rating, Comment, Purchase
df["purchase_count_scaled"] = MinMaxScaler(feature_range = (1, 5)). \
    fit(df[["purchase_count"]]).transform(df[["purchase_count"]])

df["comment_count_scaled"] = MinMaxScaler(feature_range = (1, 5)). \
    fit(df[["commment_count"]]).transform(df[["commment_count"]])


# Calculating Scores
def weighted_scoring_score(dataframe, w1=32, w2=26, w3=42):
    return (dataframe["comment_count_scaled"] * w1 +
            dataframe["purchase_count_scaled"] * w2 +
            dataframe["rating"] * w3)


df["weighted_sorting_score"] = weighted_scoring_score(df)

df.sort_values(by = "weighted_sorting_score", ascending = True).head(20)


# Bayesian Average Rating Score
# Sorting Products According to Distribution of 5-Star Rating

def bayesian_average_rating(n, confidence=0.95):  # puanların dağılımı üzerinden ortalama hesaplar. n means stars.
    if sum(n) == 0:
        return 0
    k = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    n = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (n + k)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (n + k)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (n + k + 1))
    return score


df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                                "2_point",
                                                                "3_point",
                                                                "4_point",
                                                                "5_point"]]), axis = 1)
df.sort_values(by = "bar_score", ascending = False).head(20)
df[df["course_name"].index.isin([5, 1])].sort_values(by = "bar_score", ascending = False)


# Hybrid Sorting: Bar Score + Other Factors
def hybrid_sorting_score(dataframe, bar_w=0.6, wss_w=0.4):
    bar_score = df.apply(lambda x: bayesian_average_rating(x[["1_point",
                                                              "2_point",
                                                              "3_point",
                                                              "4_point",
                                                              "5_point"]]), axis = 1)
    wss_score = weighted_scoring_score(dataframe)
    return bar_score * bar_w + wss_score * wss_w


df["hybrid_sorting_score"] = hybrid_sorting_score(df)

df.sort_values(by = "hybrid_sorting_score", ascending = False).head(20)
