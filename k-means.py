# k-means clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('Mall_Customers.csv')
x = dataset.iloc[:, [3,4]].values


#using elbow method
from sklearn.cluster import KMeans
wcss = []


#using k-means to find optimum number of clusters
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init='k-means++', n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(x)
    #within cluster sum of squares
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11), wcss)
plt.title('the elbow method')
plt.xlabel('number of clusters')
plt.ylabel('wcss')
    
#applying k-means 
kmeans = KMeans(n_clusters = 5, init='k-means++', n_init = 10, max_iter = 300, random_state = 0)
y_means = kmeans.fit_predict(x)
    
#visualising
plt.scatter(x[y_means == 0, 0], x[y_means==0, 1], c='red', label='cluster 1')
plt.scatter(x[y_means == 1, 0], x[y_means==0, 1], c='blue', label='cluster 2')
plt.scatter(x[y_means == 2, 0], x[y_means==0, 1], c='green', label='cluster 3')
plt.scatter(x[y_means == 3, 0], x[y_means==0, 1], c='orange', label='cluster 4')
plt.scatter(x[y_means == 4, 0], x[y_means==0, 1], c='yellow', label='cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100,c='black',label='centroids')
plt.legend()
plt.show()
