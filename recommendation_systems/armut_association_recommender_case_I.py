import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


def pd_options():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 500)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

df_ = pd.read_csv("datasets/miuul/armut_data.csv")

df = df_.copy()


def data_summary(dataframe):
    print("# # # [ D A T A F R A M E --> I N F O ] # # #")
    print(dataframe.info())
    print("# # # [ D A T A F R A M E --> M I S S I N G V A L U E S ] # # #")
    print(dataframe.isnull().sum())
    print("# # # [ D A T A F R A M E --> D E S C R I B E ] # # #")
    print(dataframe.describe().T)
    print("# # # [ D A T A F R A M E --> H E A D ] # # #")
    print(dataframe.head(10))


data_summary(df)

# Join two variables in a new column
df["Service"] = df[["ServiceId", "CategoryId"]].apply(lambda x: "_".join(x.astype(str)), axis = 1)

# Create date variable that contain month and year
df["CreateDate"] = pd.to_datetime(df.CreateDate)
df["Year_Month"] = df.CreateDate.dt.strftime("%Y-%m")

# Join two variables in a new column
df["BasketId"] = df[["UserId", "Year_Month"]].apply(lambda x: "_".join(x.astype(str)), axis = 1)

# Create ARL basket and service matrix
basket_service_df = df.groupby(["BasketId", "Service"])["CategoryId"].count().unstack().applymap(
    lambda x: 1 if x > 0 else 0)

# Create association rules
frequent_itemsets = apriori(basket_service_df.astype("bool"),
                            min_support = .01,
                            use_colnames = True)

rules = association_rules(frequent_itemsets,
                          metric = "support",
                          min_threshold = .01)

rules.sort_values("support", ascending = False)


# Create recommendations
def get_recommendation(rules_df, service, rec_count=1):
    rules_lift_df = rules_df.sort_values("lift", ascending = False)
    recommendation_list = []
    for index, services in enumerate(rules_lift_df["antecedents"]):
        for j in list(services):
            if j == service:
                recommendation_list.append(list(rules_lift_df.iloc[index]["consequents"])[0])

    return print(f"These services will be useful for you \n {recommendation_list[0:rec_count]}")


get_recommendation(rules, "2_0", 6)
