# Import necessary elements
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


# Settings for better visualization

def pd_options():
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 30)
    pd.set_option('display.width', 400)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

# 1. Read the DataFrame

df_ = pd.read_excel("datasets/kaggle/online_retail_II.xlsx", sheet_name = 2010 - 2011)
df = df_.copy()


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

# 3. Prepare the Data
# POST doesn't related to product, it is a price added later, don't need it, remove
post = df.loc[df["StockCode"] == "POST"]
df = df.drop(index = post.index)

# 3.1. Remove the blanks
df.isnull().sum()
df = df.dropna()

# 3.2. C means refund in the invoice, don't need refund, remove
df = df[~df["Invoice"].str.contains("C", na = False)]

# 3.3. Need only positive prices, remove zeros and negatives
df = df[df["Price"] > 0]

# 3.4. Check values for outlier numbers
df[["Price", "Quantity"]].describe().T


# Quantity and price max number is too high when the %75 quantiles numbers are low, let's suppress them
# 3.5. Determine threshold number for suppress it the outlier


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


replace_with_threshold(df, "Quantity")
replace_with_threshold(df, "Price")

# 4. Create the Pivot Table and Rules
# 4.1. Create the Pivot Table
# A value of 1 will be entered for those who buy the product, and 0 for those who don't
invoice_product_df = df.groupby(["Invoice", "Description"])["Quantity"].count().unstack().applymap(
    lambda x: 1 if x > 0 else 0)

# 4.2. Create the Rules
frequent_itemsets = apriori(invoice_product_df.astype("bool"),
                            min_support = .01,
                            use_colnames = True)

rules = association_rules(frequent_itemsets,
                          metric = "support",
                          min_threshold = .01)


# 6. Find Product Name with ID
def check_id(dataframe, stock_code):
    product_name = dataframe[dataframe["StockCode"] == stock_code]["Description"].values[0]
    print(product_name)


check_id(df, 21987)
check_id(df, 23235)
check_id(df, 22747)


# 7. Recommend the Product to Customers
def get_recommendation(rules_df, stock_code, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending = False)
    recommendation_list = []
    for index, product in enumerate(sorted_rules["antecedents"]):
        for j in list(product):
            if j == stock_code:
                recommendation_list.append(list(sorted_rules.iloc[index]["consequents"])[0])

    return recommendation_list[0:rec_count]


get_recommendation(rules, 23235, 1)

rules.sort_values("lift", ascending = False)
rules["antecedents"]
rules["consequents"][0]
