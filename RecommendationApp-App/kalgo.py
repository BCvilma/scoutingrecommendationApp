import numpy as np
import random

class KMeans:
    def __init__(self, n_clusters=2, max_iter=100):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None

    def fit_predict(self, X):
        # Initialize centroids randomly
        random_indices = random.sample(range(len(X)), self.n_clusters)
        self.centroids = X[random_indices]

        # Iterate until convergence or max iterations
        for _ in range(self.max_iter):
            # Assign clusters
            cluster_assignments = self.assign_clusters(X)
            old_centroids = np.copy(self.centroids)
            # Update centroids
            self.update_centroids(X, cluster_assignments)
            # Check for convergence
            if np.array_equal(old_centroids, self.centroids):
                break

        return cluster_assignments

    def assign_clusters(self, X):
        # Compute distances between each point and centroids
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)
        # Assign each point to the closest centroid
        return np.argmin(distances, axis=1)

    def update_centroids(self, X, cluster_assignments):
        # Update centroids to the mean of points in each cluster
        for i in range(self.n_clusters):
            cluster_points = X[cluster_assignments == i]
            if len(cluster_points) > 0:
                self.centroids[i] = np.mean(cluster_points, axis=0)
