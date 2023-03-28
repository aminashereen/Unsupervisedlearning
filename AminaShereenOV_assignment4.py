#!/usr/bin/env python
# coding: utf-8

# In[1]:


#case study on unsupervised learning


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


df=pd.read_csv('Wine_clust.csv')


# In[4]:


df.head()


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.describe()


# In[8]:


df.isna().sum()


# In[9]:


df.duplicated()


# In[10]:


for i in df:
    sns.boxplot(x=df[i])
    plt.show()


# In[11]:


df1=df.iloc[:,[11,12]].values


# In[12]:


type(df1)


# # kmeans

# In[13]:


from sklearn.cluster import KMeans


# In[14]:


wcss=[]
for i in range(1,13):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
    kmeans.fit(df1)
    wcss.append(kmeans.inertia_)


# In[15]:


wcss


# In[16]:


# plotting no. of clusters Vs wcss


# In[17]:


plt.plot(range(1,13),wcss)
plt.title('Elbow method')
plt.xlabel('No. of clusters')
plt.ylabel('WCSS score')
plt.show()


# In[18]:


#from elbow method, optimum no.of clusters=4


# In[19]:


#kmeans
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=42)
y_kmeans=kmeans.fit_predict(df1)


# In[20]:


y_kmeans


# In[21]:


from sklearn.metrics import silhouette_score
silhouette_sc=silhouette_score(df1,y_kmeans)
print(silhouette_sc)


# # Agglomerative Clustering

# In[22]:


import scipy.cluster.hierarchy as sch


# In[23]:


dendrogram=sch.dendrogram(sch.linkage(df,method='ward'))
plt.title('Dendrogram')
plt.xlabel('Wine')
plt.ylabel('Euclidean distance')
plt.show()


# In[24]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=2,affinity='euclidean',linkage='ward')
y_hc=hc.fit_predict(df1)
y_hc


# In[25]:


#silhouette score
from sklearn.metrics import silhouette_score
silhouette_avg=silhouette_score(df1,y_hc)
print(silhouette_avg)


# In[26]:


#scaling the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
scaled_data = sc.fit_transform(df)
scaled_data = pd.DataFrame(scaled_data,columns=df.columns)


# In[27]:


scaled_data


# # PCA

# In[28]:


from sklearn.decomposition import PCA


# In[29]:


#specifying the no.of components
pca=PCA(n_components=10)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)


# In[30]:


x_pca.shape


# In[31]:


x_pca


# In[32]:


pca.explained_variance_ratio_


# In[33]:


np.sum(pca.explained_variance_ratio_)


# # DBSCAN

# In[34]:


from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=3, min_samples=10)
y_dbscan=dbscan.fit_predict(scaled_data)


# In[35]:


y_dbscan


# In[36]:


df['DBSCANCluster']=y_dbscan
df.tail(10)


# In[37]:


df_1=df[df['DBSCANCluster']==0]
df_2=df[df['DBSCANCluster']==1]
df_3=df[df['DBSCANCluster']==-1]


# In[38]:


silhouette_sc=silhouette_score(df,y_dbscan)
print(silhouette_sc)


# In[ ]:




