import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind


def pd_options():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 200)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

# Creating Data
control_df = pd.read_excel("datasets/ab_testing.xlsx", sheet_name = "control")
test_df = pd.read_excel("datasets/ab_testing.xlsx", sheet_name = "test")
control_df["Bidding"] = "Maximum Bid"
test_df["Bidding"] = "Average Bid"
df = pd.concat([control_df, test_df], ignore_index = True)
df.info()
df.head()
# Hyphotesis Test
# H0: M1=M2 (iki tasarım arasında satın alma dönüşüm farklılıkları yoktur)
df["Purchase"].mean()  # -> 550.89 - 582.10

test_stat, pvalue = shapiro(df.loc[df["Bidding"] == "Maximum Bid", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value 0.58 çıktığı için H0 reddedilemez. Normal dağılım vardır.
test_stat, pvalue = shapiro(df.loc[df["Bidding"] == "Average Bid", "Purchase"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value 0.15 çıktığı için H0 reddedilemez. Normal dağılım vardır.
# Normal dağılım olduğu için Parametrik TTEST_IND kullanılacaktır.

# Varyans Homojenliği Varsayım Testi LEVENE
test_stat, pvalue = levene(df.loc[df["Bidding"] == "Maximum Bid", "Purchase"].dropna(),
                           df.loc[df["Bidding"] == "Average Bid", "Purchase"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value 0.10 çıktığı için H0 reddedilemez. Varyans homojenliği vardır.
# Varyans homojenliği olduğu için Equal_var True kullanılacaktır.

test_stat, pvalue = ttest_ind(df.loc[df["Bidding"] == "Maximum Bid", "Purchase"],
                              df.loc[df["Bidding"] == "Average Bid", "Purchase"], equal_var = True)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value 0.34 çıktığı için H0 reddedilemez.
# Dolayısıyla maximum ve average bidding yöntemleri arasında satın alma bağlamında anlamlı bir farklılık görülememiştir.

# Recommends to Customer
# There is no need to use the average bidding method, which is the product of a new investment and effort, instead of
# maximum bidding, which is currently used and has various functions. Because there was no significant difference
# between the two bidding methods. Therefore, the transformation of investments to be made on average bidding will
# not be positive for the company.

# If it is believed that the average bidding method will achieve better results, the available data should be
# expanded. 40 of each bidding method reduces the data confidence interval rate. By collecting much more data,
# the confidence interval can be improved and the test can be performed again.
