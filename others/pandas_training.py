import pandas as pd
import seaborn as sns
import numpy as np

s = pd.Series([1, 2, 3, 4, 5, 6])
type(s)  # pandas.cores.series.Series
s.index  # start 0, stop 6, step 1
s.ndim  # pandas serileri tek boyutludur
s.values  # pandasta values ifadesi kullanırsak numpy.ndarray'e döner
type(s.values)

df = pd.read_csv("advertising.csv")
df.head()

df = sns.load_dataset("titanic")
df.head()
df.shape  # 895 satır, 15 sütun
df.info()
df.describe().T
df.isnull().values.any()  # veri değişkenlerden arındırıldığı için numpy array döndürdü
df.isnull().sum()
df["sex"].value_counts()  # değişken seçmek istediğimizde köşeli parantez ve değişkenin ismi

df.drop(0, axis = 0).head()
delete_indexes = [col for col in df.index if col % 2 == 0]
df.drop(delete_indexes, axis = 0).head(20)
df.drop(0, axis = 0, inplace = True)  # inplace değişikliği kalıcı olarak yapar

df["age"].head()
df.age.head()  # yukarıdakinin aynısı ikisi de seçim işlemi yapar
"age" in df

df.index = df.age

df.drop("age", axis = 1, inplace = True)  # satırlarda axis 0, sütunlarda axis 1

df["age"] = df.index

df.head()

df = df.reset_index().head()

pd.set_option("display.max.columns", None)

df[["age", "alive"]]

col_names = ["age", "adult_male", "alive"]
df[col_names]

df["age2"] = df["age"] ** 2
df["age3"] = df["age"] / df["age2"]

df.drop(col_names, axis = 1).head()

df.loc[:, ~df.columns.str.contains("age")].head()  # labelda/dataframede seçim yapar

df.iloc[0, 0]
df.iloc[0: 3]  # 3'e kadar
df.loc[0: 3]  # mutlak

df.iloc[0:3, 0:3]
df.loc[0:3, "age"]

col_names = ["age", "embarked", "alive"]
df.loc[0:3, col_names]

df[df["age"] > 50]["age"].count()

df.loc[df["age"] > 50, ["class", "age"]].head()  # yaşı 50'den büyüklerin sınıf bilgisi

df_new = df.loc[(df["age"] > 50) &
                (df["sex"] == "male") &
                ((df["embark_town"] == "Cherbourg") |
                 (df["embark_town"] == "Southampton")),
                ["class", "age", "embark_town"]].head()

df_new["embark_town"].value_counts()

df.age.mean()
df.groupby("sex")["age"].mean()  # cinsiyete göre yaş ortalaması
df.groupby("sex").agg({"age": ["mean", "sum"]})  # yukarıdakinin aynısı, birden fazla agg

df.groupby("sex").agg({"age": ["mean", "sum"],
                       "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg({"age": "mean",
                                                 "survived": "mean"})

df.groupby(["sex", "embark_town", "class"]).agg(dict(age = "mean", survived = "mean", sex = "count"))

df.pivot_table("survived", "sex", "embarked")  # ön tanımlı mean alır

df.pivot_table("survived", "sex", "embarked", aggfunc = "std")  # standard sapma aldırdık

df.pivot_table("survived", "sex", ["embarked", "class"])

df["new_age"] = pd.cut(df["age"], [0, 10, 18, 25, 40, 90])  # cutlar sayısal değişkenleri kategorik değişkene çevirir

df.pivot_table("survived", "sex", ["new_age", "class"])  # kesişim, satır, sütun

pd.set_option("display.width", 100)

df["age2"] = df["age"] * 2
df["age3"] = df["age"] * 5

df.head()

for col in df.columns:
    if "age" in col:
        print(col)

for col in df.columns:
    if "age" in col:
        df[col] = df[col] / 10

df[["age", "age2", "age3"]].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: x / 10).head()

df.loc[:, df.columns.str.contains("age")].apply(lambda x: (x - x.mean()) / x.std()).head()


def standart_scaler(col_name):
    return (col_name - col_name.mean()) / col_name.std()


df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()  # yukarıdakinin aynısı

df.loc[:, df.columns.str.contains("age")] = \
    df.loc[:, df.columns.str.contains("age")].apply(standart_scaler).head()  # işlemi kaydetmek için

m = np.random.randint(1, 30, (5, 3))
df1 = pd.DataFrame(m, columns = ["var1", "var2", "var3"])
df2 = df1 + 99

pd.concat([df1, df2], ignore_index = True)  # indexleri düzeltir

df1 = pd.DataFrame({"name": ["a", "ab", "taş", "abcd"],
                    "job": ["a", "ab", "abc", "abcd"]})
df2 = pd.DataFrame({"name": ["a", "ab", "abc", "abcd"],
                    "date": [2010, 2011, 2012, 2013]})
df3 = pd.merge(df1, df2)
pd.merge(df1, df2, on = "name")

df4 = pd.DataFrame({"manager": ["a", "ab", "abc", "abcd"],
                    "job": ["a", "ab", "abc", "abcd"]})
pd.merge(df3, df4)
