# IMDB Scoring and Sorting
import math

import pandas as pd
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 200)
pd.set_option("display.expand_frame_repr", False)
pd.options.mode.chained_assignment = None
pd.set_option("display.float_format", lambda x: "%.2f" % x)

df = pd.read_csv("datasets/movies_metadata.csv", low_memory = False)
df = df[["title", "vote_average", "vote_count"]]

# Ranking Vote Average
df.sort_values(by = "vote_average", ascending = False).head(20)
df["vote_count"].describe([0.1, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99]).T
df[df["vote_count"] > 400].sort_values(by = "vote_average", ascending = False).head(15)

df["vote_count_score"] = MinMaxScaler(feature_range = (1, 10)) \
    .fit(df[["vote_count"]]).transform(df[["vote_count"]])

df["average_count_score"] = df["vote_average"] * df["vote_count_score"]

df.sort_values(by = "average_count_score", ascending = False).head(20)

# IMBD Weighted Rating
# weighted_rating = (v/(v+m) * r) + (m/(v+m) * c)
# r = vote_average
# v = vote_count
# m = minimum votes required to be listed in the top 250
# c = the mean vote across the whole report

# Film 1, r = 8, m = 500, v = 1000
# ilk hesaplama: (1000/ (1000+500)) * 8 = 5.33
# ikinci hesaplama: 500 / (1000+500) * 7 = 2.33
# Toplan = 7.66

m = 2500
c = df["vote_average"].mean()


def weighted_rating(r, v, m, c):
    return (v / (v + m) * r) + (m / (v + m) * c)


weighted_rating(7.4, 11444, m, c)

weighted_rating(8.5, 8358, m, c)

df["weighted_rating"] = weighted_rating(df["vote_average"],
                                        df["vote_count"], m, c)

df.sort_values(by = "weighted_rating", ascending = False).head(20)


# Bayesian Average Rating Score - BAR
def bayesian_average_rating(n, confidence=0.95):  # puanların dağılımı üzerinden ortalama hesaplar. n means stars.
    if sum(n) == 0:
        return 0
    K = len(n)
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    N = sum(n)
    first_part = 0.0
    second_part = 0.0
    for k, n_k in enumerate(n):
        first_part += (k + 1) * (n[k] + 1) / (N + K)
        second_part += (k + 1) * (k + 1) * (n[k] + 1) / (N + K)
    score = first_part - z * math.sqrt((second_part - first_part * first_part) / (N + K + 1))
    return score


bayesian_average_rating([34733, 4355, 4704, 6561, 13515, 26183, 87368, 273082, 600260, 1295351])

df = pd.read_csv("datasets/imdb_ratings.csv")

df["bar_score"] = df.apply(lambda x: bayesian_average_rating(x[["one", "two", "three", "four", "five",
                                                                "six", "seven", "eight", "nine", "ten"]]), axis = 1)
