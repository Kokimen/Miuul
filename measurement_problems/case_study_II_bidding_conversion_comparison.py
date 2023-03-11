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

# Creating Data
control_df = pd.read_excel("datasets/ab_testing.xlsx", sheet_name = "control")
test_df = pd.read_excel("datasets/ab_testing.xlsx", sheet_name = "test")
df = pd.concat([control_df, test_df], axis = 1)
df.info()

# Hyphotesis Test
# H0: M1=M2 (iki tasarım arasında satın alma dönüşüm farklılıkları yoktur)
control_df["Purchase"].mean()  # ->550.89
test_df["Purchase"].mean()  # ->582.10
HİPOTEZ TESTİ YAP