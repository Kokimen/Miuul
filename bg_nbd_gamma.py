import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
# pip install lifetimes
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 20)
pd.set_option('display.width', 200)
pd.options.mode.chained_assignment = None
pd.set_option("display.float_format", lambda x: "%.1f" % x)


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


# Reading Data
df_ = pd.read_excel("C:\\Users\\Estesia\\Downloads\\crmAnalytics\\datasets\\online_retail_II.xlsx",
                    sheet_name = "Year 2010-2011")
df = df_.copy()
df.head()
df.describe().T
df.isnull().sum()

# Preparing Data

df = df.dropna()
df = df.loc[~df["Invoice"].str.contains("C", na = False)]
df = df.loc[(df["Quantity"] > 0) & (df["Price"] > 0)]

replace_with_threshold(df, "Quantity")
replace_with_threshold(df, "Price")

df["TotalPrice"] = df["Quantity"] * df["Price"]

today_date = dt.datetime(2011, 12, 11)

# Preparing Lifetime Data Structure#
# Recency: Son satın alma üzerinden geçen zaman, haftalık. (kullanıcı özelinde)
# T: Müşterinin yaşı, haftalık. (analiz tarihinden ne kadar süre önce ilk satın alma yapılmış)
# Frequency: Tekrar eden toplam satın alma sayısı (frequency > 1)
# Monetary_value: Satın alma başına ortalama kazanç.

cltv_df = df.groupby("Customer ID").agg({"InvoiceDate": [
    lambda invoicedate: (invoicedate.max() - invoicedate.min()).days,
    lambda invoicedate: (today_date - invoicedate.min()).days],
    "Invoice": lambda num: num.nunique(),
    "TotalPrice": lambda totalprice: totalprice.sum()})

cltv_df.columns = cltv_df.columns.droplevel(0)
cltv_df.columns = ["recency", "T", "frequency", "monetary"]

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

cltv_df = cltv_df.loc[cltv_df["frequency"] > 1]

cltv_df["recency"] = cltv_df["recency"] / 7

cltv_df["T"] = cltv_df["T"] / 7

# BG-NBD Expected Number of Transaction

bgf = BetaGeoFitter(penalizer_coef = 0.001)

bgf.fit(cltv_df["frequency"],
        cltv_df["recency"],
        cltv_df["T"])

# 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri?
bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending = False).head(10)

cltv_df["expected_purchase_1_week"] = bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                                                              cltv_df["frequency"],
                                                                                              cltv_df["recency"],
                                                                                              cltv_df["T"])

# 1 ay içinde en çok satın alma beklediğimiz 10 müşteri?
bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sort_values(ascending = False).head(10)

cltv_df["expected_purchase_1_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                                                               cltv_df["frequency"],
                                                                                               cltv_df["recency"],
                                                                                               cltv_df["T"])

# 1 ay içinde beklediğimiz toplam satış sayısı?
bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                        cltv_df["frequency"],
                                                        cltv_df["recency"],
                                                        cltv_df["T"]).sum()

# 3 ay içinde beklediğimiz satış sayısı?
cltv_df["expected_purchase_3_month"] = bgf.conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                                                               cltv_df["frequency"],
                                                                                               cltv_df["recency"],
                                                                                               cltv_df["T"])

# Examine of Forecast Results
plot_period_transactions(bgf)
plt.show(block = True)

# Gamma-Gamma Expected Average Profit

ggf = GammaGammaFitter(penalizer_coef = 0.01)

ggf.fit(cltv_df["frequency"],
        cltv_df["monetary"])

ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                        cltv_df["monetary"]).sort_values(ascending = False).head(10)

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                                                             cltv_df["monetary"])

cltv_df.sort_values("expected_average_profit", ascending = False).head()

# Calculating CLTV with Models

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df["frequency"],
                                   cltv_df["recency"],
                                   cltv_df["T"],
                                   cltv_df["monetary"],
                                   time = 3, freq = "W", discount_rate = 0.01)

cltv = cltv.reset_index()

cltv_final = cltv_df.merge(cltv, on = "Customer ID", how = "left")

cltv_final.sort_values(by = "clv", ascending = False).head(10)

# Segments and CLTV

cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels = ["D", "C", "B", "A"])

cltv_final.sort_values(by = "clv", ascending = False).head(50)

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})


# Function Time

def create_cltv_p(dataframe, month=3):
    dataframe = dataframe.dropna()
    dataframe = dataframe.loc[~dataframe["Invoice"].str.contains("C", na = False)]
    dataframe = dataframe.loc[(dataframe["Quantity"] > 0) & (dataframe["Price"] > 0)]

    replace_with_threshold(dataframe, "Quantity")
    replace_with_threshold(dataframe, "Price")

    dataframe["TotalPrice"] = dataframe["Quantity"] * dataframe["Price"]

    today_date = dt.datetime(2011, 12, 11)

    cltv_df = dataframe.groupby("Customer ID").agg({"InvoiceDate": [
        lambda invoicedate: (invoicedate.max() - invoicedate.min()).days,
        lambda invoicedate: (today_date - invoicedate.min()).days],
        "Invoice": lambda num: num.nunique(),
        "TotalPrice": lambda totalprice: totalprice.sum()})

    cltv_df.columns = cltv_df.columns.droplevel(0)
    cltv_df.columns = ["recency", "T", "frequency", "monetary"]

    cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]

    cltv_df = cltv_df.loc[cltv_df["frequency"] > 1]

    cltv_df["recency"] = cltv_df["recency"] / 7

    cltv_df["T"] = cltv_df["T"] / 7

    # BG-NBD Expected Number of Transaction

    bgf = BetaGeoFitter(penalizer_coef = 0.001)

    bgf.fit(cltv_df["frequency"],
            cltv_df["recency"],
            cltv_df["T"])

    # 1 hafta içinde en çok satın alma beklediğimiz 10 müşteri?
    bgf.conditional_expected_number_of_purchases_up_to_time(1,
                                                            cltv_df["frequency"],
                                                            cltv_df["recency"],
                                                            cltv_df["T"]).sort_values(ascending = False).head(10)

    cltv_df["expected_purchase_1_week"] = bgf. \
        conditional_expected_number_of_purchases_up_to_time(1,
                                                            cltv_df["frequency"],
                                                            cltv_df["recency"],
                                                            cltv_df["T"])

    # 1 ay içinde en çok satın alma beklediğimiz 10 müşteri?
    bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                            cltv_df["frequency"],
                                                            cltv_df["recency"],
                                                            cltv_df["T"]).sort_values(ascending = False).head(10)

    cltv_df["expected_purchase_1_month"] = bgf. \
        conditional_expected_number_of_purchases_up_to_time(4,
                                                            cltv_df["frequency"],
                                                            cltv_df["recency"],
                                                            cltv_df["T"])

    # 1 ay içinde beklediğimiz toplam satış sayısı?
    bgf.conditional_expected_number_of_purchases_up_to_time(4,
                                                            cltv_df["frequency"],
                                                            cltv_df["recency"],
                                                            cltv_df["T"]).sum()

    # 3 ay içinde beklediğimiz satış sayısı?
    cltv_df["expected_purchase_3_month"] = bgf. \
        conditional_expected_number_of_purchases_up_to_time(4 * 3,
                                                            cltv_df["frequency"],
                                                            cltv_df["recency"],
                                                            cltv_df["T"])

    # Examine of Forecast Results
    plot_period_transactions(bgf)
    plt.show(block = True)

    # Gamma-Gamma Expected Average Profit

    ggf = GammaGammaFitter(penalizer_coef = 0.01)

    ggf.fit(cltv_df["frequency"],
            cltv_df["monetary"])

    ggf.conditional_expected_average_profit(cltv_df["frequency"],
                                            cltv_df["monetary"]).sort_values(ascending = False).head(10)

    cltv_df["expected_average_profit"] = ggf. \
        conditional_expected_average_profit(cltv_df["frequency"],
                                            cltv_df["monetary"])

    cltv_df.sort_values("expected_average_profit", ascending = False).head()

    # Calculating CLTV with Models

    cltv = ggf.customer_lifetime_value(bgf,
                                       cltv_df["frequency"],
                                       cltv_df["recency"],
                                       cltv_df["T"],
                                       cltv_df["monetary"],
                                       time = month, freq = "W", discount_rate = 0.01)

    cltv = cltv.reset_index()

    cltv_final = cltv_df.merge(cltv, on = "Customer ID", how = "left")

    cltv_final.sort_values(by = "clv", ascending = False).head(10)

    # Segments and CLTV

    cltv_final["segment"] = pd.qcut(cltv_final["clv"], 4, labels = ["D", "C", "B", "A"])

    return cltv_final


cltv_final2 = create_cltv_p(df)
