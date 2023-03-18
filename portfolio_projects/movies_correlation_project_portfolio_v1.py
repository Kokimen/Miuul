# Import libraries
import pandas as pd


def pd_options():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 200)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

# Read the data
df = pd.read_csv("datasets/kaggle/movie_industry.csv")

# Explore the data
df.info()
df.head()
# Clean Stage #
# Check the missing values
df.isnull().sum()

# Check the duplicate values
df.duplicated().sum()

# Delete the missing values
df = df.dropna()

# Control the data types
df.dtypes

# Change the wrong data type
df['released'] = df['released'].str.split('(').str[0]
df["released"] = df["released"].astype("datetime64")
df[['gross', 'budget', "votes"]] = df[['gross', 'budget', "votes"]].astype(int)
df = df.drop("released", axis = 1)

# Check the negative values
df.loc[df["gross"] < 0], ["budget", "name"]
