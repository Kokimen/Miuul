import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
df = sns.load_dataset("titanic")
df.head()

# kategorik = sütun, countplot bar
# say?sal = histogram, boxplot

df["sex"].value_counts()  # kategorik de?i?kenler ile ilgili akla gelecek ilk fonksiyon, betimleyici

df["sex"].value_counts().plot(kind = "bar")
plt.show()