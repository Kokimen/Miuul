import datetime
import math

import pandas as pd
import scipy.stats as st


def pd_options():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option('display.width', 500)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.4f" % x)


pd_options()

df = pd.read_csv("datasets/amazon_reviews.csv")
df.head(20), df.shape, df.describe(), df.info()

# Calculate Average Rating According to Currenct Comments
# Calculate Product Average Rating
df["overall"].value_counts()
df["overall"].mean()
df = df.drop("day_diff", axis = 1)

# Time-Based Weighted Average
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
current_date = pd.to_datetime(df["reviewTime"].max() + datetime.timedelta(days = 1))
df["days_diff"] = (current_date - df["reviewTime"]).dt.days

df["days_diff"].describe().T


def time_based_weighted_avg(dataframe, w1=.28, w2=.24, w3=.20, w4=.16, w5=.12):
    return df.loc[df["days_diff"] <= 90, "overall"].mean() * w1 + \
           df.loc[(df["days_diff"] > 90) & (df["days_diff"] <= 180), "overall"].mean() * w2 + \
           df.loc[(df["days_diff"] > 180) & (df["days_diff"] <= 270), "overall"].mean() * w3 + \
           df.loc[(df["days_diff"] > 270) & (df["days_diff"] <= 360), "overall"].mean() * w4 + \
           df.loc[df["days_diff"] > 360, "overall"].mean() * w5


time_based_weighted_avg(df)

# Create Vote Down Variable with Values
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


# Create and Calculate SPND, SAR and WLB Scores
def score_pos_neg_diff(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    else:
        return helpful_yes - helpful_no


def score_average_rating(helpful_yes, helpful_no):
    if helpful_yes + helpful_no == 0:
        return 0
    else:
        return helpful_yes / (helpful_yes + helpful_no)


def wilson_lower_bound(helpful_yes, helpful_no, confidence=0.95):
    n = helpful_yes + helpful_no
    if n == 0:
        return 0
    else:
        z = st.norm.ppf(1 - (1 - confidence) / 2)
        phat = 1.0 * helpful_yes / n
        return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


# score_pos_neg_diff
df["score_pos_neg_diff"] = df.apply(lambda x: score_pos_neg_diff(x["helpful_yes"], x["helpful_no"]), axis = 1)

# score_avg_rating
df["score_avg_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis = 1)

# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis = 1)

df.loc[df["total_vote"] > 0].sort_values(by = "total_vote", ascending = False).head(20)

df.loc[df["wilson_lower_bound"] > 0].sort_values(by = "wilson_lower_bound", ascending = False).head(20)
