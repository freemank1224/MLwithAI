import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_blobs

class KMeansClustering:
    def __init__(self, n_clusters=3, max_iterations=100):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations

    def fit(self, X):
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iterations):
            # 分配点到最近的质心
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # 更新质心
            new_centroids = np.array([X[self.labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # 检查是否收敛
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids

        return self.labels, self.centroids

def animate_kmeans(X, kmeans, n_clusters):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(X[:, 0], X[:, 1], c='gray')
    centroid_scatter = ax.scatter([], [], c='red', s=200, marker='*')
    
    def init():
        ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
        ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
        return scatter, centroid_scatter

    def update(frame):
        labels, centroids = kmeans.fit(X)
        scatter.set_array(labels)
        centroid_scatter.set_offsets(centroids)
        ax.set_title(f'Iteration {frame + 1}')
        return scatter, centroid_scatter

    anim = FuncAnimation(fig, update, frames=kmeans.max_iterations, 
                         init_func=init, blit=True, repeat=False, interval=500)
    plt.colorbar(scatter)
    plt.show()

if __name__ == "__main__":
    # 生成示例数据
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

    # 创建并运行K-means聚类
    n_clusters = 4
    kmeans = KMeansClustering(n_clusters=n_clusters, max_iterations=10)
    
    # 显示动画
    animate_kmeans(X, kmeans, n_clusters)
