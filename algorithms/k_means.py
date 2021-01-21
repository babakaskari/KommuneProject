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
file_to_read = open(f'{path}\\{list_dir[-1]}', "rb")
loaded_object = pickle.load(file_to_read)
file_to_read.close()
# print(loaded_object)
dataset = loaded_object
# print("column names : ", dataset["df"].columns)
column1 = 'Shifted_Mean_Hour'
column2 = 'Flow'

clustering_df = pd.DataFrame(dataset["df"], columns=[f'{column1}'])
clustering_df[f'{column2}'] = dataset["y_dataset"]
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
# print("cluster dataset : ", clustering_df)
centroids = kmeans.cluster_centers_

print("kmeans.inertia    :   ", kmeans.inertia_)
print("kmeans cluster centers   :   ", kmeans.cluster_centers_)
print("k-means labels : ", kmeans.labels_)
print("y_pred : ", y_pred)
# pd.DataFrame(kmeans.labels_).to_csv("labels.csv")

colors = ("red", "green", "blue")
d_numpy = clustering_df.to_numpy()
# print("d_numpy ", d_numpy)
plt.scatter(d_numpy[y_pred == 0, 0], d_numpy[y_pred == 0, 1], s=25, c='green', label='Cluster 1')
plt.scatter(d_numpy[y_pred == 1, 0], d_numpy[y_pred == 1, 1], s=25, c='blue', label='Cluster 2')
# plt.scatter(d_numpy[y_pred == 2, 0], d_numpy[y_pred == 2, 1], s=25, c='black', label='Cluster 1')
# plt.scatter(d_numpy[y_pred == 3, 0], d_numpy[y_pred == 3, 1], s=25, c='yellow', label='Cluster 2')
plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Two assumed centroid points')
plt.title('Clusters')
plt.legend()
plt.xlabel(f'{column1}')
plt.ylabel(f'{column2}')
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
var = [23.24102771, 5.319835068]
arr = np.array(var)
print("arr : ", arr)
arr = np.reshape(arr, (1, -1))
# print("arrr after reshape : ", arr)
prediction = kmeans.predict(arr)
print("prediction : ", prediction)
pkl_filename = "k_means_model.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(kmeans, file)
if prediction[0] == 1:
    print("Thers is a leak alarm")
else:
    print("There is no leak alaram")

