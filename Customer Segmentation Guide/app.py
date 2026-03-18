import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("Mall_Customers.csv")

print("First 5 rows:")
print(df.head())

# Convert Gender to numeric
print(df.columns)

# Select features
X = df[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------------------
# Elbow Method
# ------------------------------
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()

# ------------------------------
# Apply KMeans
# ------------------------------
kmeans = KMeans(n_clusters=5, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\nClustered Data:")
print(df.head())

# ------------------------------
# Visualization
# ------------------------------
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=df['Annual Income (k$)'],
    y=df['Spending Score (1-100)'],
    hue=df['Cluster'],
    palette='Set1'
)
plt.title("Customer Segmentation")
plt.show()

# ------------------------------
# Cluster Analysis
# ------------------------------
print("\nCluster Summary:")
print(df.groupby('Cluster').mean())