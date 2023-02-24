import seaborn as sns
df = sns.load_dataset("car_crashes")

# ONE#
["NUM_" + col.upper() if str(df[col].dtypes) not in "object" else col.upper() for col in df]

# TWO#
[col + "_FLAG" if "no" not in col else col for col in df]

# THREE#
import seaborn as sns
df = sns.load_dataset("car_crashes")

og_list = ["abbrev", "no_previous"]

new_cols = [col for col in df.columns if col not in og_list]

new_df = df[new_cols]

new_df.head()

df[[col for col in df.columns if col not in og_list]].head()
