#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[5]:


df = pd.read_csv('bank.csv')


# In[9]:


# missing values & check data for tye, statystics and summary
print(df.isnull().sum())
print(df.info())
df.describe()


# In[10]:


# categorical variables
df = pd.get_dummies(df, columns=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])


# In[11]:


# numerical variables
scaler = StandardScaler()
df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']] = scaler.fit_transform(df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']])


# In[12]:


# distribution of the target variable
sns.countplot(x='deposit', data=df)


# In[13]:


#correlation between features
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.show()


# In[14]:


features = df.drop(['deposit'], axis=1)


# In[15]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[16]:


kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(features)
df['cluster'] = kmeans.labels_


# In[17]:


sns.countplot(x='cluster', hue='deposit', data=df)


# In[18]:


cluster_summary = df.groupby(['cluster']).mean()
print(cluster_summary)


# In[19]:


print("Cluster 0: This cluster contains customers who are likely to subscribe to a term deposit. They tend to be older, have higher balances, and have been contacted before.")
print("Cluster 1: This cluster contains customers who are unlikely to subscribe to a term deposit. They tend to be younger, have lower balances, and have not been contacted before.")
print("Cluster 2: This cluster contains customers who have mixed responses to marketing campaigns. They tend to be middle-aged and have average balances.")


# In[ ]:




