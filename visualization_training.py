import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

# kategorik = sütun, countplot bar
# sayısal = histogram, boxplot

df["sex"].value_counts()  # kategorik değişkenler ile ilgili akla gelecek ilk fonksiyon, betimleyici

df["sex"].value_counts().plot(kind = "bar")
plt.show(block=True)

plt.hist(df["age"])
plt.show(block=True)

plt.boxplot(df["fare"])
plt.show(block=True)  # aykırı değerleri yakalar

home_prices = np.array([1000, 2000, 3000, 4000, 5000])
car_prices = np.array([800, 1600, 2400, 3200, 4000])
plt.subplot(1, 2, 1)  # 1 satırlık, 2 sütunluk, 1. grafik
plt.title("PRICES ONE")
plt.plot(home_prices, car_prices)

food_prices = np.array([100, 200, 300, 400, 500])
water_prices = np.array([80, 160, 240, 320, 400])
plt.subplot(1, 2, 2)  # 1 satırlık, 2 sütunluk, 2. grafik
plt.title("PRICES TWO")
plt.plot(food_prices, water_prices)
plt.show(block=True)

df = sns.load_dataset("tips")
df["sex"].value_counts()
sns.countplot(x = df["sex"])
plt.show(block=True)

sns.boxplot(x = df["total_bill"])
plt.show(block=True)

df["total_bill"].hist()
plt.show(block=True)


