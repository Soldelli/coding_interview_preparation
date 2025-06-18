import numpy as np
import matplotlib.pyplot as plt

class Kmeans():
    def __init__(self, k=3, max_iters=100, tol=1e-4):
        """
        k: Number of clusters
        max_iters: Maximum number of iterations
        tol: Convergence tolerance (change in centroids)
        """
        self.k = k
        self.max_iters = max_iters 
        self.tol = tol

    def fit(self, X):
        '''
            Run k-means clustering
        '''

        # step 1: randomly initialize the centroids
        np.random.seed(42)
        random_indices = np.random.choice(X.shape[0], self.k, replace=False)
        self.centroids = X[random_indices]


        for iter in range(self.max_iters):
            # step 2: assign each point to the nearest centroid
            distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # [num_samples, k]   np.linalg.norm = torch.norm = L2 norm
            cluster_labels = np.argmin(distances, axis=1)  # finds the nearest centroid index for each point

            # step 3: compute the new centroids as the mean of assigned data points
            new_centroids = np.array([X[cluster_labels == i].mean(axis=0) for i in range(self.k)])

            # step 4: check for convergence
            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroid = new_centroids # update centroids

        self.labels_ = cluster_labels # store final assignments


    def predict(self, X):
        '''
            Predict the nearest cluster for new data points X
        '''
        distances = np.linalg.norm(X[:, np.newaxis] - self.centroids, axis=2)  # [num_samples, k]
        return np.argmin(distances, axis=1)
    


        
if __name__ == "__main__":
    # Generate synthetic data
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)

    # Run K-Means
    kmeans = Kmeans(k=3)
    kmeans.fit(X)
    labels = kmeans.labels_

    print(kmeans.centroid)

    # # Plot results
    # plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
    # plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], c='red', marker='x', s=200, label="Centroids")
    # plt.legend()
    # plt.title("K-Means Clustering")
    # plt.show()