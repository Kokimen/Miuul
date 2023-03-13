import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.stats.api as sms
from matplotlib import pyplot as plt
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, pearsonr, spearmanr, kendalltau, \
    f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest
