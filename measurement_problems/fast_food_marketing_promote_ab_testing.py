import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest


def pd_options():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 200)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()
