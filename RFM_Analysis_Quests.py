import datetime as dt
import pandas as pd


def read_dataframe():
    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_rows", None)
    pd.set_option('display.width', 200)
    # I choose how many digits to see in numbers.
    pd.set_option("display.float_format", lambda x: "%.3f" % x)

    df_ = pd.read_csv("C:/Users/Estesia/Downloads/flo_customers_analysis/flo_data_20k.csv")

    df = df_.copy()

    return df


df = read_dataframe()


def examine_dataframe(dataframe):
    print(dataframe.head(10),
          dataframe.columns,
          dataframe.describe(),
          dataframe.columns.value_counts().isnull(),
          dataframe.dtypes)


examine_dataframe(df)


def prepare_data(dataframe):
    dataframe = dataframe.dropna()

    dataframe["customer_sum_shopping"] = dataframe["order_num_total_ever_online"] + dataframe[
        'order_num_total_ever_offline']
    dataframe["customer_sum_shopping"] = dataframe["customer_sum_shopping"].astype(int)
    dataframe["customer_sum_spend"] = dataframe['customer_value_total_ever_offline'] + dataframe[
        'customer_value_total_ever_online']

    dataframe = dataframe.apply(lambda columns: pd.to_datetime(columns) if "date" in columns.name else columns)

    return dataframe


df = prepare_data(df)


def group_sort(dataframe):
    print(dataframe.groupby("order_channel").agg({"customer_sum_shopping": "sum", "customer_sum_spend": "sum"}),
          dataframe.sort_values("customer_sum_spend", ascending = False).head(10),
          dataframe.sort_values("customer_sum_shopping", ascending = False).head(10))


group_sort(df)


def calculate_rfm_metric():
    today_date = dt.datetime(2021, 6, 1)
    rfm = df.groupby("master_id").agg({"last_order_date": lambda last_order_date: (today_date - last_order_date),
                                       "customer_sum_shopping": lambda customer_sum_shopping: customer_sum_shopping,
                                       "customer_sum_spend": lambda customer_sum_spend: customer_sum_spend})

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm.loc[rfm["monetary"] > 0]

    return rfm


rfm = calculate_rfm_metric()


def calculate_rfm_score():
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels = [5, 4, 3, 2, 1])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels = [1, 2, 3, 4, 5])
    rfm["RF_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

    return rfm


calculate_rfm_score()


def create_segment():
    segment_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_lose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    rfm["segment"] = rfm["RF_SCORE"].replace(segment_map, regex = True)
    rfm[["segment", "recency", "frequency", "monetary"]]

    return rfm


create_segment()


def special_customers(csv=False):
    new_rfm_df = df.merge(rfm, on = "master_id")

    new_rfm_df_female = new_rfm_df.loc[((new_rfm_df["segment"] == "loyal_customers") |
                                        (new_rfm_df["segment"] == "champions")) &
                                       (new_rfm_df["interested_in_categories_12"].str.contains("KADIN"))]

    if csv:
        new_rfm_df_female["master_id"].to_csv("rfm_special_customers.csv")

    return new_rfm_df_female


new_rfm_df_female = special_customers()


def special_customers_two(csv=False):
    new_rfm_df = df.merge(rfm, on = "master_id")

    new_rfm_df_manchi = new_rfm_df.loc[((new_rfm_df["segment"] == "cant_lose") |
                                        (new_rfm_df["segment"] == "about_to_sleep") |
                                        (new_rfm_df["segment"] == "new_customers")) &
                                       ((new_rfm_df["interested_in_categories_12"].str.contains("ERKEK")) |
                                        (new_rfm_df["interested_in_categories_12"].str.contains("COCUK")))]

    if csv:
        new_rfm_df_manchi["master_id"].to_csv("rfm_special_customers_two.csv")

    return new_rfm_df_manchi


new_rfm_df_manchi = special_customers_two()
