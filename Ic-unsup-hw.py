import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# 1-3. Load + numeric + scale
df = pd.read_csv("wine_dataset.csv")
df_num = df.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).dropna()
X_scaled = StandardScaler().fit_transform(df_num)

# 4-6. Hierarchical + dendrogram + fcluster
Z = linkage(X_scaled, method="ward")
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title("Dendrogram - Choose k=3")
plt.show()

k = 3
clusters_hier = fcluster(Z, t=k, criterion="maxclust")
df_num["cluster"] = clusters_hier

# 7. Print hierarchical counts
print("Hierarchical counts:")
print(df_num["cluster"].value_counts().sort_index())
print()

# 8. KMeans same k
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)
df_num["kmeans_cluster"] = labels_kmeans

print("KMeans counts:")
print(pd.Series(labels_kmeans).value_counts().sort_index())
print("Not identical: different algorithms")

# 9. PCA variance
pca = PCA()
pca.fit(X_scaled)
explained = pca.explained_variance_ratio_
cumsum = np.cumsum(explained)

print("\nPCA variance:")
for i, (ev, cv) in enumerate(zip(explained, cumsum), 1):
    print(f"PC{i}: {ev:.4f} (cum: {cv:.4f})")

# Save
df_num.to_csv("wine_results.csv", index=False)
print("\nSaved wine_results.csv")
