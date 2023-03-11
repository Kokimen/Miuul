import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest


def pd_options():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 200)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

# Sampling
population = np.random.randint(0, 80, 10000)

population.mean()

np.random.seed(115)

sample = np.random.choice(population, size = 100)

sample.mean()

# Descriptive Statistics
df = sns.load_dataset("tips")
df.describe().T

# Confidence Intervals
df = sns.load_dataset("tips")

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
sms.DescrStatsW(df["tip"]).tconfint_mean()

# Correlation
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip", "total_bill")
plt.show(block = True)

df["tip"].corr(df["total_bill"])

# Hypothesis Testing Case Study
df.groupby("smoker").agg({"total_bill": "mean"})

# H0: M1=M2
# H1: M1!=M2

# Normallik Varsayımı SHAPIRO
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
test_stat, pvalue = shapiro(df.loc[df["smoker"] == "Yes", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# p-value < 0.05 means H0 REDDEDİLİR.
# p-value >= 0.05 means H0 REDDEDİLEMEZ.
# Normal dağılım sağlanmadığı için non-parametric test kullanılacaktır.

# Varyans Homojenliği Varsayımı LEVENE
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen değildir.
test_stat, pvalue = levene(df.loc[df["smoker"] == "Yes", "total_bill"],
                           df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))
# p-value < 0.05 means H0 REDDEDİLİR.
# p-value >= 0.05 means H0 REDDEDİLEMEZ.
# Varyanslar homojen değildir.

# Hipotezin Uygulanması
# Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
# Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

# Varsayımlar sağlanmamasına rağmen Parametrik Test Denemesi TTEST_IND
# Hem normallik dağılımı hem de varyans sağlanırsa equal_var True olur.
# Sadece normallik dağılımı sağlanıyorsa varyans sağlanmıyorsa equal_var False olur.
test_stat, pvalue = ttest_ind(df.loc[df["smoker"] == "Yes", "total_bill"],
                              df.loc[df["smoker"] == "No", "total_bill"], equal_var = True)
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Varsayımlar sağlanmadığı için Non-Parametrik Test Denemesi MANNWHITNEYU
test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Hypotesis Testing Case Study 2
df = sns.load_dataset("titanic")

df.groupby("sex").agg({"age": "mean"})
# H0: M1=M2 (anlamlı fark yoktur)
# H1: M1!=M2 (anlamlı fark vardır)

# Normallik Varsayımı SHAPIRO
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1: Normal dağılım varsayımı sağlanmamaktadır.
test_stat, pvalue = shapiro(df.loc[df["sex"] == "female", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Varyans Homojenliği Varsayımı LEVENE
# H0: Varyanslar homojendir.
# H1: Varyanslar homojen değildir.
test_stat, pvalue = levene(df.loc[df["sex"] == "female", "age"].dropna(),
                           df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Homojenlik yok ama Varsayım var Non-Parametric Test MANNWHITNEYU ve equal_var True
test_stat, pvalue = mannwhitneyu(df.loc[df["sex"] == "female", "age"].dropna(),
                                 df.loc[df["sex"] == "male", "age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Hypothesis Testing Case Study 3
df = pd.read_csv("datasets/diabetes.csv")
df.groupby("Outcome").agg({"Age": "mean"})

# H0: M1=M2 (diyabet olan/olmayanların yaş ortalamaları arasında anlamlı fark yoktur)
# H1: M1!=M2
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Non Parametrik Test MANNWHITNEYU
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Hypothesis Testing Case Study 4
df = pd.read_csv("datasets/course_reviews.csv")
df.head()

df.loc[(df["Progress"] > 75)]["Rating"].mean()
df.loc[(df["Progress"] < 25)]["Rating"].mean()
df.loc[(df["Progress"] < 10)]["Rating"].mean()

# Normal Dağılım SHAPIRO
test_stat, pvalue = shapiro(df.loc[df["Progress"] > 75, "Rating"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Progress"] < 25, "Rating"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Non-Parametirc MANNWHITNEYU
test_stat, pvalue = mannwhitneyu(df.loc[df["Progress"] > 75, "Rating"].dropna(),
                                 df.loc[df["Progress"] < 25, "Rating"].dropna())
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Two Sampling Ratio Test
# H0: P1=P2 (yeni tasarım/eski tasarım dönüşüm oranı arasında farklılık yoktur)
success = np.array([300, 250])
observation = np.array([1000, 1100])

proportions_ztest(count = success, nobs = observation)

# Two Sampling Ratio Test Case Study 2
df = sns.load_dataset("titanic")
# H0: P1=P2 (kadın/erkek hayatta kalma oranları arasında farklılık yoktur)
female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count = [female_succ_count, male_succ_count],
                                      nobs = [df.loc[df["sex"] == "female", "survived"].shape[0],
                                              df.loc[df["sex"] == "male", "survived"].shape[0]])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# ANOVA - Analysis of Variance
df = sns.load_dataset("tips")
df.groupby("day")["total_bill"].mean()

# Normallik Varsayımı
# Varyans Homojenlik Varsayımı

# Varsayım Sağlanıyorsa One-Way-Anova
# Varsayım Sağlanmıyorsa Kruskal

# H0: Normal dağılım varsayımı sağlanmaktadır. -> RED
# Below values are <0.05, so no normalization.

for group in list(df["day"].unique()):
    pvalue = shapiro(df.loc[df["day"] == group, "total_bill"])[1]
    print(group, "p-value: %.4f" % pvalue)

# H0: Varyans homojenliği varsayımı sağlanmaktadır. -> REDDEDİLEMEZ
test_stat, pvalue = levene(df.loc[df["day"] == "Sun", "total_bill"],
                           df.loc[df["day"] == "Sat", "total_bill"],
                           df.loc[df["day"] == "Thur", "total_bill"],
                           df.loc[df["day"] == "Fri", "total_bill"])
print("Test Stat = %.4f, p-value = %.4f" % (test_stat, pvalue))

# Parametrik Anova Testi
# HO: Grup ortalamaları arasında fark yoktur. -> RED
# Below values are <0.05, so there is difference.
f_oneway(df.loc[df["day"] == "Sun", "total_bill"],
         df.loc[df["day"] == "Sat", "total_bill"],
         df.loc[df["day"] == "Thur", "total_bill"],
         df.loc[df["day"] == "Fri", "total_bill"])

# Non Parametrik Anova Testi
# HO: Grup ortalamaları arasında fark yoktur. -> RED
# Below values are <0.05, so there is difference.
kruskal(df.loc[df["day"] == "Sun", "total_bill"],
        df.loc[df["day"] == "Sat", "total_bill"],
        df.loc[df["day"] == "Thur", "total_bill"],
        df.loc[df["day"] == "Fri", "total_bill"])

# Special Information
from statsmodels.stats.multicomp import MultiComparison

comparison = MultiComparison(df["total_bill"], df["day"])
tukey = comparison.tukeyhsd(0.10)  # -> tolerans gösterme seviyemiz
print(tukey)
