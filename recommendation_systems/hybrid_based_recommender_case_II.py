import pandas as pd


# PD settings for better visualization
def pd_options():
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 30)
    pd.set_option('display.width', 300)
    pd.set_option("display.expand_frame_repr", True)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

# 1. Read the DataFrames
movie = pd.read_csv("datasets/kaggle/movie.csv")
rating = pd.read_csv("datasets/kaggle/rating.csv")
df = rating.merge(movie, how = "left", on = "movieId")


# 2. Check the Data Specs, Examine the Data
def data_summary(dataframe):
    print("# # # [ D A T A F R A M E --> I N F O ] # # #")
    print(dataframe.info())
    print("# # # [ D A T A F R A M E --> M I S S I N G V A L U E S ] # # #")
    print(dataframe.isnull().sum())
    print("# # # [ D A T A F R A M E --> D U P L I C A T E D ] # # #")
    print(dataframe.duplicated().sum())
    print("# # # [ D A T A F R A M E --> D E S C R I B E ] # # #")
    print(dataframe.describe([.05, .25, .5, .75, .85, .9, .95, .99]).T)
    print("# # # [ D A T A F R A M E --> H E A D ] # # #")
    print(dataframe.head(10))


data_summary(df)


# 3. Create User Movie DataFrame
def create_user_movie_df(dataframe):
    comment_counts = pd.DataFrame(dataframe["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 10000].index
    common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index = ["userId"], columns = ["title"], values = "rating")
    return user_movie_df


user_movie_df = create_user_movie_df(df)

# 4. Find the User's Watched Movies #
# 4.1.Select the random user
random_user = int(pd.Series(user_movie_df.index).sample(1).values)

# 4.2.Find the user watched movies
random_user_df = user_movie_df[user_movie_df.index == random_user]

# 4.3.Find the user rated movies
movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# 5. Reach the Other User's Watched Same Movies
movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

# 5.1.Find percentage of randomly selected user movies should be watched by other users
percentage = len(movies_watched) * .6

users_same_movies = user_movie_count[user_movie_count["movie_count"] > percentage]["userId"]

# 6. Determination of Similarity
final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]

# 6.1.Find the correlation between users/movies
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns = ["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

# 6.2.Create new users dataframe which contains high correlation
top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= .65)][
    ["user_id_2", "corr"]].reset_index(drop = True)
top_users = top_users.sort_values("corr", ascending = False)
top_users = top_users.rename(columns = {"user_id_2": "userId", "corr": "correlation"})

# 6.3.Merge top users with ratings (we should know that they are like the movie)
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how = "inner")
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# 7. Calculate Weighted Average Recommendation Score
top_users_ratings["weighted_rating"] = top_users_ratings["correlation"] * top_users_ratings["rating"]

# 7.1.Find the mean of weighted ratings
recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()

# 7.2. Select the just higher than 3.5 weighted ratings
movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3.5]. \
    sort_values("weighted_rating", ascending = False)

# 7.3.Show the movie names that recommend to user
movies_to_be_recommend_names = movies_to_be_recommend.merge(movie[["movieId", "title"]])
movies_to_be_recommend_names["title"].head()
print(f"The movies you can be like \n {movies_to_be_recommend_names.title.head()}")

# Item Based Recommendation
# 1. Read the DataFrames
movie = pd.read_csv("datasets/kaggle/movie.csv")
rating = pd.read_csv("datasets/kaggle/rating.csv")
df = movie.merge(rating, how = "left", on = "movieId")


# 2. Create User Movie DataFrame
def create_user_movie_df(dataframe):
    comment_counts = pd.DataFrame(dataframe["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 1000].index
    common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index = ["userId"], columns = ["title"], values = "rating")
    return user_movie_df


user_movie_df = create_user_movie_df(df)


# 3. Select the Random User's Latest Five-Star Movie
def random_user_movie_df(dataframe):
    random_user = int(pd.Series(user_movie_df.index).sample(1).values)
    movie_id = dataframe[(dataframe.userId == random_user) & (df["rating"] == 5)]
    movie_id_latest = movie_id[movie_id["timestamp"] == movie_id["timestamp"].max()]["movieId"].values[0]
    movie_name = dataframe.loc[dataframe["movieId"] == movie_id_latest]["title"].values[0]
    return movie_name


movie_name = random_user_movie_df(df)


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending = False)[1:6]


item_based_recommender(movie_name, user_movie_df)
