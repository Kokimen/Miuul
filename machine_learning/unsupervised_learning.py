################################
# Unsupervised Learning
################################

# pip install yellowbrick

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder

################################
# K-Means
################################

df = pd.read_csv("datasets/USArrests.csv", index_col = 0)

df.head()
df.isnull().sum()
df.info()
df.describe().T

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

kmeans = KMeans(n_clusters = 4, random_state = 17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_

################################
# Optimum Küme Sayısının Belirlenmesi
################################

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters = k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show()

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k = (2, 20))
elbow.fit(df)
elbow.show()

elbow.elbow_value_

################################
# Final Cluster'ların Oluşturulması
################################

kmeans = KMeans(n_clusters = elbow.elbow_value_).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.read_csv("datasets/USArrests.csv", index_col = 0)

df["cluster"] = clusters_kmeans

df.head()

df["cluster"] = df["cluster"] + 1

df[df["cluster"] == 5]

df.groupby("cluster").agg(["count", "mean", "median"])

df.to_csv("clusters.csv")

################################
# Hierarchical Clustering
################################

df = pd.read_csv("datasets/USArrests.csv", index_col = 0)

sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)

hc_average = linkage(df, "average")

plt.figure(figsize = (10, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           leaf_font_size = 10)
plt.show()

plt.figure(figsize = (7, 5))
plt.title("Hiyerarşik Kümeleme Dendogramı")
plt.xlabel("Gözlem Birimleri")
plt.ylabel("Uzaklıklar")
dendrogram(hc_average,
           truncate_mode = "lastp",
           p = 10,
           show_contracted = True,
           leaf_font_size = 10)
plt.show()

################################
# Kume Sayısını Belirlemek
################################


plt.figure(figsize = (7, 5))
plt.title("Dendrograms")
dend = dendrogram(hc_average)
plt.axhline(y = 0.5, color = 'r', linestyle = '--')
plt.axhline(y = 0.6, color = 'b', linestyle = '--')
plt.show()

################################
# Final Modeli Oluşturmak
################################

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters = 5, linkage = "average")

clusters = cluster.fit_predict(df)

df = pd.read_csv("datasets/USArrests.csv", index_col = 0)
df["hi_cluster_no"] = clusters

df["hi_cluster_no"] = df["hi_cluster_no"] + 1

df["kmeans_cluster_no"] = df["kmeans_cluster_no"] + 1
df["kmeans_cluster_no"] = clusters_kmeans

################################
# Principal Component Analysis
################################

df = pd.read_csv("datasets/Hitters.csv")
df.head()

num_cols = [col for col in df.columns if df[col].dtypes != "O" and "Salary" not in col]

df[num_cols].head()

df = df[num_cols]
df.dropna(inplace = True)
df.shape

df = StandardScaler().fit_transform(df)

pca = PCA()
pca_fit = pca.fit_transform(df)

pca.explained_variance_ratio_
np.cumsum(pca.explained_variance_ratio_)
