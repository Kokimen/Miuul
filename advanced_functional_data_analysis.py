import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()


def check_df(dataframe):
    print(dataframe.shape)
    print(dataframe.dtypes)
    print(dataframe.isnull().sum())
    print(dataframe.describe().T)


check_df(df)

df["sex"].unique()
df["sex"].nunique()

categoric_cols = [col for col in df.columns if str(df[col].dtypes) in
                  ["category", "object", "bool"]]

numeric_but_categoric = [col for col in df.columns if df[col].nunique() < 10
                         and df[col].dtypes
                         in ["int64", "float64"]]

# bir kategorik değişkenin çok fazla sınıfının olması bilgi taşımadığı anlamına gelir
categoric_but_cardinal = [col for col in df.columns if df[col].nunique() > 20
                          and str(df[col].dtypes)
                          in ["category", "object"]]

categoric_cols += numeric_but_categoric

categoric_cols = [col for col in categoric_cols if col not in categoric_but_cardinal]

df[categoric_cols].nunique()

[col for col in df.columns if col not in categoric_cols]


def categoric_summary(dataframe, col_name):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))


categoric_summary(df, "sex")

for col in categoric_cols:
    categoric_summary(df, col)


def categoric_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x = dataframe[col_name], data = dataframe)
        plt.show(block=True)


categoric_summary(df, "sex", plot=True)

for col in categoric_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
    else:
        categoric_summary(df, col, plot = True)

df[["age", "fare"]].describe().T
numeric_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

numeric_cols = [col for col in numeric_cols if col not in categoric_cols]


def numeric_summary(dataframe, numeric_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.50, 0.80, 0.90]
    print(dataframe[numeric_col].describe(quantiles).T)

    if plot:
        dataframe[numeric_col].hist()
        plt.xlabel(numeric_col)
        plt.title(numeric_col)
        plt.show(block=True)


numeric_summary(df, "age", plot = True)

for col in numeric_cols:
    numeric_summary(df, col, plot = True)


def grab_column_names(dataframe, categoric_threshold=10, cardinal_threshold=20):
    """
    Veri setindeki kategorik, numerik ve kategorik fakar kardinal değişkenlerin isimlerini verir

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'nin ismi
    categoric_threshold: int, float
        Numerik fakat kategorik olan değişkenler için sınıf eşik değeri
    cardinal_threshold: int, float
        Kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    -------
        categoric_cols: List
            Kategorik değişken listesi
        numeric_cols: List
            Numerik değişken listesi
        categoric_but_cardinal: List
            Kategorik görünümlü kardinal değişken listesi

    """
    categoric_cols = [col for col in df.columns if str(df[col].dtypes) in
                      ["category", "object", "bool"]]

    numeric_but_categoric = [col for col in df.columns if df[col].nunique() < 10
                             and df[col].dtypes
                             in ["int64", "float64"]]

    # bir kategorik değişkenin çok fazla sınıfının olması bilgi taşımadığı anlamına gelir
    categoric_but_cardinal = [col for col in df.columns if df[col].nunique() > 20
                              and str(df[col].dtypes)
                              in ["category", "object"]]

    categoric_cols += numeric_but_categoric

    categoric_cols = [col for col in categoric_cols if col not in categoric_but_cardinal]

    numeric_cols = [col for col in df.columns if df[col].dtypes in ["int64", "float64"]]

    numeric_cols = [col for col in numeric_cols if col not in categoric_cols]

    return categoric_cols, numeric_cols, categoric_but_cardinal


categoric_cols, numeric_cols, categoric_but_cardinal = grab_column_names(df)

# Bonus, bool tipteki verileri int'e çevirmek
for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

for col in categoric_cols:
    categoric_summary(df, col, plot = True)

df.head()

categoric_summary(df, "survived", plot = True)

df.groupby("sex").agg({"survived": ["mean", "sum"]})
df.groupby("sex").agg("survived").mean()


def target_summary_with_categoric(dataframe, target, categorical_col):
    print(pd.DataFrame({target: dataframe.groupby(categorical_col)[target].mean()}), categorical_col)


target_summary_with_categoric(df, "survived", "sex")

for col in categoric_cols:
    target_summary_with_categoric(df, "survived", col)

df.groupby("survived")["age"].mean()
df.groupby("survived").agg({"age": "mean"})


def target_summary_with_numeric(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}))


for col in numeric_cols:
    target_summary_with_numeric(df, "survived", col)

df = pd.read_csv("breast_cancer.csv")
df = df.iloc[:, 1:-1]
df.head()

numeric_cols = [col for col in df.columns if df[col].dtype in [int, float]]

correlation = df[numeric_cols].corr()  # corr korelasyon fonksiyonu, -1 ile +1 arasında değer alır

sns.set(rc={"figure.figsize": (12, 12)})
sns.heatmap(correlation, cmap="RdBu")
plt.show(block=True)

correlation_matrix = df.corr().abs()

triangle_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(np.bool))

drop_list = [col for col in triangle_matrix.columns if any(triangle_matrix[col] > 0.9)]

correlation_matrix[drop_list]

df.drop(drop_list, axis=1)


def high_correlated_columns(dataframe, plot=False, correlation_threshold = .9):
    correlation = dataframe.corr()
    correlation_matrix = correlation.abs()
    triangle_matrix = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k = 1).astype(np.bool))
    drop_list = [col for col in triangle_matrix.columns if any(triangle_matrix[col] > 0.9)]
    if plot:
        sns.set(rc = {"figure.figsize": (12, 12)})
        sns.heatmap(correlation, cmap = "RdBu")
        plt.show(block = True)
    return drop_list


high_correlated_columns(df)

drop_list = high_correlated_columns(df, plot=True)

high_correlated_columns(df.drop(drop_list, axis=1), plot=True)