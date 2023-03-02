#Business Problem

#Step 2: Data Understanding#

import datetime as dt
import pandas as pd
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
#I choose how many digits to see in numbers.
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

df["Total_Price"] = df["Quantity"] * df["Price"]

df.groupby("Invoice").agg({"Total_Price": "sum"}).head()

#Step 3: Data Preperation#
df.isnull().sum()
df = df.dropna()
df.describe().T

df = df[~df["Invoice"].str.contains("C", na=False)]

#Calculating RFM Metrics#