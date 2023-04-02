import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv('bank.csv')

# Check for missing values in the DataFrame
print(df.isnull().sum())

# Print information about the DataFrame
print(df.info())

# Generate descriptive statistics for the DataFrame
df.describe()

# categorical variables
df = pd.get_dummies(df, columns=['job', 'marital', 'education',
                    'default', 'housing', 'loan', 'contact', 'month', 'poutcome'])

# numerical variables
scaler = StandardScaler()
df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']] = scaler.fit_transform(
    df[['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']])

# distribution of the target variable
sns.countplot(x='deposit', data=df)

# Correlation between features: create a heatmap to visualize the correlation between features
plt.figure(figsize=(12, 10))  # Set the size of the plot
# Create the heatmap with correlation values and annotations
sns.heatmap(df.corr(), cmap='coolwarm', annot=True)
plt.show()  # Show the plot

# Features for clustering:
features = df.drop(['deposit'], axis=1)

# Determine the optimal number of clusters
wcss = []  # Create an empty list to store WCSS values for different numbers of clusters

# Calculate WCSS for 1 to 10 clusters
for i in range(1, 11):
    # Create a KMeans object with i clusters
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(features)  # Fit the KMeans object to the data
    wcss.append(kmeans.inertia_)  # Add the WCSS value to the list

# Plot the WCSS values for different numbers of clusters
# Plot the WCSS values against the number of clusters
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')  # Add a title to the plot
plt.xlabel('Number of clusters')  # Add a label to the x-axis
plt.ylabel('WCSS')  # Add a label to the y-axis
plt.show()  # Show the plot

# Clustering using KMeans algorithm
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(features)

# Assign cluster labels to each data point in the DataFrame
df['cluster'] = kmeans.labels_

# Visualization of clusters
sns.countplot(x='cluster', hue='deposit', data=df)

# Analyze each cluster
cluster_summary = df.groupby(['cluster']).mean()

# Print the mean values of each feature for each cluster
print(cluster_summary)

# Recomendations:
print("Cluster 0: This cluster comprises of customers who are more probable to opt for a term deposit. The individuals belonging to this cluster tend to be aged, possess higher account balances, and have been previously contacted.")
print("Cluster 1: This cluster encompasses customers who are less likely to enroll for a term deposit. Typically, these individuals are younger in age, have lower account balances, and have not received any prior communication regarding the same.")
print("Cluster 2: This cluster includes customers who display mixed reactions towards marketing campaigns. These individuals are usually in their middle age and possess average account balances")
