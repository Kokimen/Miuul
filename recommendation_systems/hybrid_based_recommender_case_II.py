import pandas as pd


# PD settings for better visualization
def pd_options():
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 30)
    pd.set_option('display.width', 400)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

# Read the data
movie = pd.read_csv("datasets/kaggle/movie.csv")
rating = pd.read_csv("datasets/kaggle/rating.csv")
df = movie.merge(rating, how = "left", on = "movieId")


# Check the Data Specs, Examine the Data
def data_summary(dataframe):
    print("# # # [ D A T A F R A M E --> I N F O ] # # #")
    print(dataframe.info())
    print("# # # [ D A T A F R A M E --> M I S S I N G V A L U E S ] # # #")
    print(dataframe.isnull().sum())
    print("# # # [ D A T A F R A M E --> D U P L I C A T E D ] # # #")
    print(dataframe.duplicated().sum())
    print("# # # [ D A T A F R A M E --> D E S C R I B E ] # # #")
    print(dataframe.describe([.05, .25, .5, .75, .85, .9, .95, .99]))
    print("# # # [ D A T A F R A M E --> H E A D ] # # #")
    print(dataframe.head(10))


data_summary(df)

# Create User Movie DataFrame
df.head()

# comment_counts = pd.DataFrame(df["title"].value_counts())
#
# rare_movies = comment_counts[comment_counts["title"] <= 10000].index
#
# common_movies = df[~df["title"].isin(rare_movies)]
#
# user_movie_df = common_movies.pivot_table(index = ["userId"], columns = ["title"], values = "rating")
