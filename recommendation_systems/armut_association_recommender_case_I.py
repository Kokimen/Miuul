import pandas as pd


def pd_options():
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.max_rows", 20)
    pd.set_option('display.width', 500)
    pd.set_option("display.expand_frame_repr", False)
    pd.options.mode.chained_assignment = None
    pd.set_option("display.float_format", lambda x: "%.3f" % x)


pd_options()

df_ = pd.read_csv("datasets/miuul/armut_data.csv")

df = df_.copy()


def data_summary(dataframe):
    print("# # # [ D A T A F R A M E --> I N F O ] # # #")
    print(dataframe.info())
    print("# # # [ D A T A F R A M E --> M I S S I N G V A L U E S ] # # #")
    print(dataframe.isnull().sum())
    print("# # # [ D A T A F R A M E --> D E S C R I B E ] # # #")
    print(dataframe.describe().T)
    print("# # # [ D A T A F R A M E --> H E A D ] # # #")
    print(dataframe.head(10))


data_summary(df)

# Join two variables in a new column
df["Service"] = df[["ServiceId", "CategoryId"]].apply(lambda x: "_".join(x.astype(str)), axis = 1)
