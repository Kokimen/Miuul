# Business Problem

# Step 2: Data Understanding#

import datetime as dt
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
# I choose how many digits to see in numbers.
pd.set_option("display.float_format", lambda x: "%.3f" % x)

df_ = pd.read_excel("C:\\Users\\Estesia\\Downloads\\crmAnalytics\\datasets\\online_retail_II.xlsx",
                    sheet_name = "Year 2009-2010")
df = df_.copy()
df.head()
df.shape

df["Description"].nunique()
df["Description"].value_counts().head()
df.groupby("Description").agg({"Quantity": "sum"}).head()
df.groupby("Description").agg({"Quantity": "sum"}).sort_values("Quantity", ascending = False).head()

df["Invoice"].nunique()

df["TotalPrice"] = df["Quantity"] * df["Price"]

df.groupby("Invoice").agg({"Total_Price": "sum"}).head()

# Step 3: Data Preperation#
df.isnull().sum()
df = df.dropna()
df.describe().T

df = df[~df["Invoice"].str.contains("C", na = False)]

# Calculating RFM Metrics#
# Recency, Frequency, Nonetary
today_date = dt.datetime(2010, 12, 11)
type(today_date)

rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda invoice_date: (today_date - invoice_date.max()).days,
                                     "Invoice": lambda invoice: invoice.nunique(),
                                     "TotalPrice": lambda total_price: total_price.sum()})
rfm.head()
rfm.columns = ["recency", "frequency", "monetary"]
rfm.describe().T

rfm = rfm.loc[rfm["monetary"] > 0]
rfm.shape

# Calculating RFM Scores#
# recency reverse, frequency and monetary flat.
rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels = [5, 4, 3, 2, 1])

rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels = [1, 2, 3, 4, 5])

rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])

rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

rfm.loc[rfm["RFM_SCORE"] == "55"].head()

rfm.loc[rfm["RFM_SCORE"] == "11"].head()

# Creating & Analysing RFM Segments#

seg_map = {
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

rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex = True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg({"mean", "count"})

rfm.loc[rfm["segment"] == "cant_lose"].head()
rfm.loc[rfm["segment"] == "cant_lose"].index

new_df = pd.DataFrame()
new_df["new_customer_id"] = rfm.loc[rfm["segment"] == "new_customers"].index
new_df["new_customer_id"] = new_df["new_customer_id"].astype(int)

new_df.to_csv("new_customers.csv")


# Functionalization Process#


def create_rfm(dataframe, csv=False):
    # PREPARING#
    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]
    dataframe = dataframe.dropna()
    dataframe = dataframe[~dataframe["Invoice"].str.contains("C", na = False)]

    # CALCULATING METRICS#
    today_date = dt.datetime(2011, 12, 11)
    rfm = df.groupby("Customer ID").agg({"InvoiceDate": lambda invoice_date: (today_date - invoice_date.max()).days,
                                         "Invoice": lambda invoice: invoice.nunique(),
                                         "TotalPrice": lambda total_price: total_price.sum()})

    rfm.columns = ["recency", "frequency", "monetary"]
    rfm = rfm.loc[rfm["monetary"] > 0]

    # CALCULATING RFM SCORES#
    rfm["recency_score"] = pd.qcut(rfm["recency"], 5, labels = [5, 4, 3, 2, 1])
    rfm["monetary_score"] = pd.qcut(rfm["monetary"], 5, labels = [1, 2, 3, 4, 5])
    rfm["frequency_score"] = pd.qcut(rfm["frequency"].rank(method = "first"), 5, labels = [1, 2, 3, 4, 5])

    # cltv_df scores converted to categoric.
    rfm["RFM_SCORE"] = (rfm["recency_score"].astype(str) + rfm["frequency_score"].astype(str))

    # NAMING CUSTOMER SEGMENTS#
    seg_map = {
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

    rfm["segment"] = rfm["RFM_SCORE"].replace(seg_map, regex = True)
    rfm[["segment", "recency", "frequency", "monetary"]]
    rfm.index = rfm.index.astype(int)

    if csv:
        rfm.to_csv("rfm.csv")

    return rfm


rfm_new = create_rfm(df)


