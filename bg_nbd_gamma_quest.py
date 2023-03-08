import datetime as dt
import pandas as pd
# pip install lifetimes
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter

# Reading Data
df_ = pd.read_csv("C:/Users/Estesia/Downloads/flo_customers_analysis/flo_data_20k.csv")
df = df_.copy()


def main():
    # Setting options
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 200)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


# Creating Thresholds
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.25)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return round(low_limit), round(up_limit)


def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Using Thresholds
def all_cltv_progress(dataframe):
    date_columns = ["order_num_total_ever_online", "order_num_total_ever_offline",
                    "customer_value_total_ever_online", "customer_value_total_ever_offline"]

    for col in date_columns:
        replace_with_threshold(dataframe, col)

    dataframe["order_num_sum"] = dataframe["order_num_total_ever_online"] + dataframe["order_num_total_ever_offline"]
    dataframe["customer_value_sum"] = dataframe["customer_value_total_ever_online"] + dataframe[
        "customer_value_total_ever_offline"]

    df = dataframe.apply(lambda columns: pd.to_datetime(columns) if "date" in columns.name else columns)

    today_date = dt.datetime(2021, 6, 1)

    cltv_df = pd.DataFrame({"customer_id": df["master_id"],
                            "recency_cltv_weekly": (df["last_order_date"] - df["first_order_date"]).
                           astype("timedelta64[D]") / 7,
                            "T_weekly": (today_date - df["first_order_date"]).astype("timedelta64[D]") / 7,
                            "frequency": df["order_num_sum"],
                            "monetary_cltv_average": df["customer_value_sum"] / df["order_num_sum"]})

    cltv_df = cltv_df.loc[(cltv_df['frequency'] > 1)]

    bgf = BetaGeoFitter(penalizer_coef = 0.001)
    bgf.fit(cltv_df['frequency'],
            cltv_df['recency_cltv_weekly'],
            cltv_df['T_weekly'])

    cltv_df["expected_sales_3_months"] = \
        bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                                cltv_df["frequency"],
                                                                cltv_df["recency_cltv_weekly"],
                                                                cltv_df["T_weekly"])
    cltv_df["expected_sales_6_months"] = \
        bgf.conditional_expected_number_of_purchases_up_to_time(4 * 6,
                                                                cltv_df["frequency"],
                                                                cltv_df["recency_cltv_weekly"],
                                                                cltv_df["T_weekly"])
    ggf = GammaGammaFitter(penalizer_coef = 0.001)
    ggf.fit(cltv_df['frequency'],
            cltv_df['monetary_cltv_average'])
    cltv_df["exp_average_value"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                           cltv_df["monetary_cltv_average"])
    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df['frequency'],
                                       cltv_df['recency_cltv_weekly'],
                                       cltv_df['T_weekly'],
                                       cltv_df['monetary_cltv_average'],
                                       time = 6,
                                       freq = "W",
                                       discount_rate = 0.01)
    cltv_df["cltv"] = cltv

    cltv_df["cltv_segment"] = pd.qcut(cltv_df["cltv"], 4, labels = ["D", "C", "B", "A"])

    return cltv_df


cltv_df = all_cltv_progress(df)

cltv_df.head()
