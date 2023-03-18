# 1. Data Pre-Process
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def pd_options():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 200)
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
    dataframe = create_invoice_product_df(dataframe, id)
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
