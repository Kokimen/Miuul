import seaborn as sns
dfcc = sns.load_dataset("car_crashes")

# ONE#
["NUM_" + col.upper() if str(dfcc[col].dtypes) not in "object" else col.upper() for col in dfcc]

# TWO#
[col + "_FLAG" if "no" not in col else col for col in dfcc]

# THREE#
import seaborn as sns
dfcc = sns.load_dataset("car_crashes")

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in dfcc.columns if col not in og_list]

new_df = dfcc[new_cols]

new_df.head()

dfcc[[col for col in dfcc.columns if col not in og_list]].head()