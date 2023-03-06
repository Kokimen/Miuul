#PREPARING DATA#
import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 200)
pd.set_option("display.float_format", lambda x: "%.5f" % x)

df_ = pd.read_excel("C:\\Users\\Estesia\\Downloads\\crmAnalytics\\datasets\\online_retail_II.xlsx",
                    sheet_name = "Year 2009-2010")
df = df_.copy()
df.head()


def prediction(dataframe, profit=0.1):
    # if there is "C" word in invoice, it means it is refund facture. We don't need them.
    dataframe = dataframe.loc[~dataframe["Invoice"].str.contains("C", na = False)]
    dataframe.describe().T

    # there is no totalprice column in database, I need in the future this one.
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    # there was minus variables in quantities, lets remove them.
    dataframe = dataframe.loc[dataframe["Quantity"] > 0]

    # we don't need missing vallues. Drop & remove them.
    dataframe = dataframe.dropna()

    cltv_c = dataframe.groupby("Customer ID").agg({"Invoice": lambda invoice: invoice.nunique(),
                                                    "Quantity": lambda quantity: quantity.sum(),
                                                    "TotalPrice": lambda totalprice: totalprice.sum()})

    cltv_c.columns = ["total_transaction", "total_unit", "total_price"]

    # Calculate Average Order Value
    cltv_c["average_order_value"] = cltv_c["total_price"] / cltv_c["total_transaction"]

    # Purschase Frequency
    cltv_c["purchase_frequency"] = cltv_c["total_transaction"] / cltv_c.shape[0]

    # Repeat Rate & Churn Rate
    repeat_rate = cltv_c.loc[cltv_c["total_transaction"] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # Profit Margin
    cltv_c["profit_margin"] = cltv_c["total_price"] * 0.1

    cltv_c["customer_value"] = cltv_c["average_order_value"] * cltv_c["purchase_frequency"]

    cltv_c["customer_lifetime_value"] = (cltv_c["customer_value"] / churn_rate) * cltv_c["profit_margin"]

    cltv_c.sort_values(by = "customer_lifetime_value", ascending = False).head()

    # Create Segment
    cltv_c["segment"] = pd.qcut(cltv_c["customer_lifetime_value"], 4, labels = ["D", "C", "B", "A"])

    cltv_c.groupby("segment").agg({"count", "mean", "sum"})

    return cltv_c


cltv = prediction(df)