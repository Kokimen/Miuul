import seaborn as sns
dfcc = sns.load_dataset("car_crashes")

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in dfcc.columns if col not in og_list]

new_df = dfcc[new_cols]

new_df.head()

dfcc[[col for col in dfcc.columns if col not in og_list]].head()