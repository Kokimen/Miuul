# 1. Data Pre-Process
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from send2trash.plat_other import uid
from surprise import Reader, Dataset


def pd_options():
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 30)
    pd.set_option('display.width', 400)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

df_ = pd.read_excel("datasets/kaggle/online_retail_II.xlsx", sheet_name = "Year 2010-2011")

df = df_.copy()


def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def retail_data_prep(dataframe):
    dataframe = dataframe.dropna()
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na = False)]
    dataframe = dataframe[(dataframe["Quantity"] > 0) | (dataframe["Price"] > 0)]
    replace_with_threshold(dataframe, "Quantity")
    replace_with_threshold(dataframe, "Price")
    return dataframe


df = retail_data_prep(df)

# 2. ARL Data Structure Making (Invoice-Product Matrix)

df_fr = df[df["Country"] == "France"]


def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(
            lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(
            lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


fr_inv_pro_df = create_invoice_product_df(df_fr, True)
fr_inv_pro_df


# Product name query from stock number (bonus)
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


check_id(df, 10120)

# Association Rules Analysis
frequent_itemsets = apriori(fr_inv_pro_df.astype("bool"),
                            min_support = .01,
                            use_colnames = True)

frequent_itemsets.sort_values("support",
                              ascending = False)

rules = association_rules(frequent_itemsets,
                          metric = "support",
                          min_threshold = .01)

rules[(rules["support"] > .05) & (rules["confidence"] > .1) & (rules["lift"] > 5)]. \
    sort_values("confidence", ascending = False)


# 4. Prepare the script
def create_invoice_product_df(dataframe, id=False):
    if id:
        return dataframe.groupby(["Invoice", "StockCode"])["Quantity"].sum().unstack().fillna(0).applymap(
            lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]
    else:
        return dataframe.groupby(["Invoice", "Description"])["Quantity"].sum().unstack().fillna(0).applymap(
            lambda x: 1 if x > 0 else 0).iloc[0:5, 0:5]


def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code][["Description"]].values[0].tolist()
    print(product_name)


def create_rules(dataframe, id=True, country="France"):
    dataframe = dataframe[dataframe["Country"] == country]
    create_invoice_product_df(dataframe, id)
    frequent_itemsets = apriori(fr_inv_pro_df.astype("bool"),
                                min_support = .01,
                                use_colnames = True)
    rules = association_rules(frequent_itemsets,
                              metric = "support",
                              min_threshold = .01)
    return rules


rules = create_rules(df)


# 5. Recommend Product
# For example id: 22492
def get_recommendation(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending = False)
    recommendation_list = []
    for i, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == product_id:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"])[0])

    return recommendation_list[0:rec_count]


get_recommendation(rules, 22492, 1)

# 1. Create TF-IDF Matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets/kaggle/movies_metadata.csv", low_memory = False)  # ignore dtype errors

df["overview"].head()
# delete not measurable words
tfidf = TfidfVectorizer(stop_words = "english")
df[df["overview"].isnull()]
df["overview"] = df["overview"].fillna(" ")
tfidf_matrix = tfidf.fit_transform(df["overview"])
tfidf.get_feature_names_out()
tfidf_matrix.toarray()

# 2. Create Cosine-Similarity Matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim.shape

# 3. Recommendation related to sameness
indices = pd.Series(df.index, index = df["title"])
# looking for just last movie
indices = indices[~indices.index.duplicated(keep = "last")]

movie_index = indices["Sherlock Holmes"]
cosine_sim[movie_index]
similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns = ["score"])
# started from one because 0 is self
movie_indices = similarity_scores.sort_values("score", ascending = False)[1:11].index
df["title"].iloc[movie_indices]


# 4. Work Script
def content_based_recommender(title, cosine_sim, dataframe):
    # creating index
    indices = pd.Series(dataframe.index, index = dataframe["title"])
    indices = indices[~indices.index.duplicated(keep = "last")]
    # catches title index
    movie_index = indices[title]
    # calculate scores related to title
    similarity_scores = pd.DataFrame(cosine_sim[movie_index], columns = ["score"])
    # show first ten movies except self
    movie_indices = similarity_scores.sort_values("score", ascending = False)[1:11].index
    return dataframe["title"].iloc[movie_indices]


content_based_recommender("The Matrix", cosine_sim, df)


def calculate_cosine_sim(dataframe):
    tfidf = TfidfVectorizer(stop_words = "english")
    dataframe["overview"] = dataframe["overview"].fillna(" ")
    tfidf_matrix = tfidf.fit_transform(dataframe["overview"])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim


cosine_sim = calculate_cosine_sim(df)

# Item-Based-Collaborative Filtering

# 1. Prepare DataFrames
movie = pd.read_csv("datasets/kaggle/movie.csv")
rating = pd.read_csv("datasets/kaggle/rating.csv")
df = movie.merge(rating, how = "left", on = "movieId")

# 2. Create User Movie DataFrame
comment_counts = pd.DataFrame(df["title"].value_counts())

rare_movies = comment_counts[comment_counts["title"] <= 10000].index

common_movies = df[~df["title"].isin(rare_movies)]

user_movie_df = common_movies.pivot_table(index = ["userId"], columns = ["title"], values = "rating")

# 3. Make The Movie Recommendations
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]
movie_name = user_movie_df[movie_name]
user_movie_df.corrwith(movie_name).sort_values(ascending = False).head(10)


def check_film(keyword, user_movie_df):
    return [col for col in user_movie_df.columns if keyword in col]


check_film("Ring", user_movie_df)


# 4. Make Script
def create_user_movie_df(dataframe):
    comment_counts = pd.DataFrame(dataframe["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 10000].index
    common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index = ["userId"], columns = ["title"], values = "rating")
    return user_movie_df


user_movie_df = create_user_movie_df(df)


def item_based_recommender(movie_name, user_movie_df):
    movie_name = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie_name).sort_values(ascending = False).head(10)


item_based_recommender(movie_name, user_movie_df)
movie_name = pd.Series(user_movie_df.columns).sample(1).values[0]

# User-Based-Collaborative Filtering #

# 1. Prepare DataFrames
movie = pd.read_csv("datasets/kaggle/movie.csv")
rating = pd.read_csv("datasets/kaggle/rating.csv")
df = movie.merge(rating, how = "left", on = "movieId")


# 2. Create User Movie DataFrame
def create_user_movie_df(dataframe):
    comment_counts = pd.DataFrame(dataframe["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 10000].index
    common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index = ["userId"], columns = ["title"], values = "rating")
    return user_movie_df


user_movie_df = create_user_movie_df(df)

# Select the random user
random_user = int(pd.Series(user_movie_df.index).sample(1, random_state = 45).values)

# 3. Find the User's Watched Movies
random_user_df = user_movie_df[user_movie_df.index == random_user]

movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

# Check above codes are true
user_movie_df.loc[user_movie_df.index == random_user,
                  user_movie_df.columns == "While You Were Sleeping (1995)"]

# 4. Reach the Other User's Watched Same Movies
movies_watched_df = user_movie_df[movies_watched]
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]

percentage = len(movies_watched) * .6

users_same_movies = user_movie_count[user_movie_count["movie_count"] > percentage]["userId"]

# 5. Determination of Similarity
final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                      random_user_df[movies_watched]])
corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
corr_df = pd.DataFrame(corr_df, columns = ["corr"])
corr_df.index.names = ["user_id_1", "user_id_2"]
corr_df = corr_df.reset_index()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= .4)][
    ["user_id_2", "corr"]].reset_index(drop = True)
top_users = top_users.sort_values("corr", ascending = False)
top_users = top_users.rename(columns = {"user_id_2": "userId", "corr": "correlation"})

top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how = "inner")
top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]

# 6. Calculate Weighted Average Recommendation Score
top_users_ratings["weighted_rating"] = top_users_ratings["correlation"] * top_users_ratings["rating"]

recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 2]. \
    sort_values("weighted_rating", ascending = False)

movies_to_be_recommend_names = movies_to_be_recommend.merge(movie[["movieId", "title"]])


# 7. Create Script
def create_user_movie_df(dataframe):
    comment_counts = pd.DataFrame(dataframe["title"].value_counts())
    rare_movies = comment_counts[comment_counts["title"] <= 10000].index
    common_movies = dataframe[~dataframe["title"].isin(rare_movies)]
    user_movie_df = common_movies.pivot_table(index = ["userId"], columns = ["title"], values = "rating")
    return user_movie_df


def user_based_recommender(random_user, user_movie_df, ratio=.6, cor_th=.4, score=2):
    random_user_df = user_movie_df[user_movie_df.index == random_user]
    movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()
    movies_watched_df = user_movie_df[movies_watched]
    user_movie_count = movies_watched_df.T.notnull().sum()
    user_movie_count = user_movie_count.reset_index()
    user_movie_count.columns = ["userId", "movie_count"]
    percentage = len(movies_watched) * ratio
    users_same_movies = user_movie_count[user_movie_count["movie_count"] > percentage]["userId"]

    final_df = pd.concat([movies_watched_df[movies_watched_df.index.isin(users_same_movies)],
                          random_user_df[movies_watched]])

    corr_df = final_df.T.corr().unstack().sort_values().drop_duplicates()
    corr_df = pd.DataFrame(corr_df, columns = ["corr"])
    corr_df.index.names = ["user_id_1", "user_id_2"]
    corr_df = corr_df.reset_index()

    top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= cor_th)][
        ["user_id_2", "corr"]].reset_index(drop = True)
    top_users = top_users.sort_values("corr", ascending = False)
    top_users = top_users.rename(columns = {"user_id_2": "userId", "corr": "correlation"})
    top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how = "inner")
    top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
    top_users_ratings["weighted_rating"] = top_users_ratings["correlation"] * top_users_ratings["rating"]

    recommendation_df = top_users_ratings.groupby("movieId").agg({"weighted_rating": "mean"}).reset_index()

    movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > score]. \
        sort_values("weighted_rating", ascending = False)
    movies_to_be_recommend_names = movies_to_be_recommend.merge(movie[["movieId", "title"]])
    return movies_to_be_recommend_names


random_user = int(pd.Series(user_movie_df.index).sample(1).values)
user_based_recommender(random_user, user_movie_df)

# Model Based Matrix Factorization #
import pandas as pd
from surprise import Reader, SVD, Dataset, accuracy
from surprise.model_selection import GridSearchCV, train_test_split, cross_validate

# 1. Prepare DataFrames
movie = pd.read_csv("datasets/kaggle/movie.csv")
rating = pd.read_csv("datasets/kaggle/rating.csv")
df = movie.merge(rating, how = "left", on = "movieId")

movie_ids = [130219, 356, 4422, 541]
movies = ["The Dark Knight (2011)",
          "Cries and Whispers (Viskningar och rop) (1972)",
          "Forrest Gump (1994)",
          "Blade Runner (1982)"]

sample_df = df[df.movieId.isin(movie_ids)]

user_movie_df = sample_df.pivot_table(index = ["userId"],
                                      columns = ["title"],
                                      values = ["rating"])
reader = Reader(rating_scale = (1, 5))

data = Dataset.load_from_df(sample_df[["userId", "movieId", "rating"]], reader)

# 2. Modelling
trainset, testset = train_test_split(data, test_size = .25)
svd_model = SVD()
svd_model.fit(trainset)
predictions = svd_model.test(testset)

accuracy.rmse(predictions)

svd_model.predict(uid = 1.0, iid = 541, verbose = True)

# 3. Model Tuning
param_grid = {"n_epochs": [5, 10, 20],
              "lr_all": [.002, .005, .007]}

gs = GridSearchCV(SVD, param_grid, measures = ["rmse", "mae"],

                  cv = 3, n_jobs = -1, joblib_verbose = True)
gs.fit(data)

gs.best_score["rmse"]
gs.best_params["rmse"]

# 4. Final Model and Prediction
svd_model = SVD(**gs.best_params["rmse"])
data = data.build_full_trainset()
svd_model.fit(data)

svd_model.predict(uid = 1.0, iid = 541, verbose = True)
