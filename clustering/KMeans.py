from sklearn.cluster import KMeans
from sklearn.datasets import make_circles

X, y = make_circles(n_samples=1000, random_state=42)

print(y)

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

print("Labels:", labels)
# print("Predicts:", predicts)
print("Centers:", centers)