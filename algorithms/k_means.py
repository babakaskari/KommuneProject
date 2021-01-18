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
x_train = dataset['y_dataset']

# /////////////////////////////////////////////////

# print(x_train)
kmeans = KMeans(n_clusters=2,
                init="k-means++",
                random_state=None,
                max_iter=300,
                algorithm='auto',
                n_init=1000).fit(x_train)
# y_pred = kmeans.predict(x_test)
centroids = kmeans.cluster_centers_
# print("Cluster centroids are : ", centroids)
X = np.arange(1, len(x_train) + 1)
# print("x_train shape", x_train.shape)


print("kmeans.inertia    :   ", kmeans.inertia_)
print("kmeans.cluster_centers_   :   ", kmeans.cluster_centers_)
# print(metrics.accuracy_score(y_test, y_pred))
y_kmeans = kmeans.fit_predict(x_train)

centroids = kmeans.cluster_centers_
print("k-means labels : ", kmeans.labels_)
# plt.scatter(np.arange(0, len(x_train), 1), x_train, c= kmeans.labels_.astype(float), s=50, alpha=0.5)
# plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.scatter(np.arange(0, len(x_train), 1), x_train, c=kmeans.labels_, s=50, alpha=0.5)
plt.xlabel('X Lable')
plt.ylabel('Flow')
plt.show()

