#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np


# # Importing all necessary libraries

# In[2]:


data=pd.read_csv("iris.csv")
data.head()


# importing the data set

# In[3]:


data_label=data['Species']
data.drop(["Id","Species"],inplace=True,axis=1)
data.head()


# removing the labels inorder to make it come under unsupervised learning

# In[4]:


data.describe()


# In[5]:


data.info()


# checking for null values to preprocess the data

# In[6]:


def get_optimal_clusters_elbow_method(x):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                        max_iter = 300, n_init = 10, random_state = 0)
        kmeans.fit(x)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The elbow method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.grid()
    plt.show()


# elbow method is used to get the accurate number of clusters
# 
# 

# In[7]:


get_optimal_clusters_elbow_method(data)


# We can see from the above elbow method that three optimal number of clusters required for our K-means clustering.

# In[8]:


model = KMeans(n_clusters = 3, init = 'k-means++',max_iter = 300, n_init = 10, random_state = 0)
y_preds = model.fit_predict(data)


# In[9]:


data=np.array(data)


# In[10]:


plt.figure(figsize=(20,8))
plt.scatter(data[y_preds == 0, 0], data[y_preds == 0, 1], 
            s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(data[y_preds == 1, 0], data[y_preds == 1, 1], 
            s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(data[y_preds == 2, 0], data[y_preds == 2, 1],
            s = 100, c = 'green', label = 'Iris-virginica')

# Plotting the centroids of the clusters
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:,1], 
            s = 100, c = 'yellow', label = 'Centroids')
plt.title('Clustering of Petals' '\n', color='black',size = 20)
plt.grid()
plt.legend()


# K-MEANS HAS CLUSTERED THE DATA INTO THREE DIFFERENT CLUSTER PERFECTLY

# In[ ]:




