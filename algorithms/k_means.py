import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn import metrics
sns.set()

cur_dir = os.getcwd()
path = os.path.dirname(cur_dir)
path = f'{path}\\dataset\\pickle'
list_dir = os.listdir(path)
file_to_read = open(f'{path}\\{list_dir[0]}', "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()
# print(loaded_object)
dataset = loaded_object

clustering_df = pd.DataFrame(dataset["df"], columns=['Hour'])
clustering_df["Flow"] = dataset["y_dataset"]
# x_train = x_train.values.reshape(-1, 1)
# print("df :", clustering_df)


# /////////////////////////////////////////////////

kmeans = KMeans(n_clusters=2,
                init="k-means++",
                random_state=None,
                max_iter=300,
                algorithm='auto',
                n_init=1000).fit(clustering_df)

y_pred = kmeans.predict(clustering_df)
centroids = kmeans.cluster_centers_

print("kmeans.inertia    :   ", kmeans.inertia_)
print("kmeans cluster centers   :   ", kmeans.cluster_centers_)
print("k-means labels : ", kmeans.labels_)
print("y_pred : ", y_pred)
colors = ("red", "green", "blue")
d_numpy = clustering_df.to_numpy()
# print("d_numpy ", d_numpy)
plt.scatter(d_numpy[y_pred == 0, 0], d_numpy[y_pred == 0, 1], s=25, c='green', label='Cluster 1')
plt.scatter(d_numpy[y_pred == 1, 0], d_numpy[y_pred == 1, 1], s=25, c='blue', label='Cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Two assumed centroid points')
plt.title('Clusters')
plt.legend()
plt.xlabel(r'Hours')
plt.ylabel('Flow Of Water')
plt.show()
# Run the Kmeans algorithm and get the index of data points clusters
elbo = []
list_k = list(range(2, 11))

for k in list_k:
    km = KMeans(n_clusters=k)
    km.fit(clustering_df)
    elbo.append(km.inertia_)

# Plot sse against k
plt.figure(figsize=(6, 6))
plt.plot(list_k, elbo, '-o')
plt.xlabel(r'Number of clusters')
plt.ylabel('Sum of squared distance')
plt.show()
