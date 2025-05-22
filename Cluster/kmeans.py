import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

class KMeans:
    def __init__(self, n_clusters=3, max_iter=100, random_state=None):
        """
        K-means（硬c均值）聚类算法实现
        
        参数:
            n_clusters: 簇的数量
            max_iter: 最大迭代次数
            random_state: 随机数种子，用于结果复现
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None
        self.labels_ = None
        self.inertia_ = None
        self.iterations = 0

    def fit(self, X):
        """
        拟合数据
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            self: 返回拟合后的实例
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # 初始化簇中心（随机选择样本点作为初始中心）
        idx = np.random.choice(len(X), self.n_clusters, replace=False)
        self.centroids = X[idx].copy()
        
        # 迭代过程
        old_centroids = np.zeros_like(self.centroids)
        self.labels_ = np.zeros(len(X))
        
        for i in range(self.max_iter):
            # 计算每个样本到每个簇中心的距离
            distances = self._calc_distances(X)
            
            # 分配样本到最近的簇
            self.labels_ = np.argmin(distances, axis=1)
            
            # 保存旧簇中心
            old_centroids = self.centroids.copy()
            
            # 更新簇中心
            for j in range(self.n_clusters):
                if np.sum(self.labels_ == j) > 0:  # 避免空簇
                    self.centroids[j] = np.mean(X[self.labels_ == j], axis=0)
            
            # 检查收敛性（簇中心不再变化）
            self.iterations = i + 1
            if np.all(old_centroids == self.centroids):
                break
        
        # 计算簇内误差平方和（inertia）
        self.inertia_ = self._calc_inertia(X)
        
        return self
    
    def predict(self, X):
        """
        预测新数据的簇标签
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            labels: 每个样本的簇标签
        """
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
    
    def _calc_distances(self, X):
        """
        计算每个样本到每个簇中心的欧氏距离
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            distances: 形状为 (n_samples, n_clusters) 的距离矩阵
        """
        distances = np.zeros((X.shape[0], self.n_clusters))
        
        for i, centroid in enumerate(self.centroids):
            # 欧氏距离的平方
            distances[:, i] = np.sum((X - centroid) ** 2, axis=1)
            
        return distances
    
    def _calc_inertia(self, X):
        """
        计算簇内误差平方和（惯性）
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            inertia: 簇内误差平方和
        """
        distances = self._calc_distances(X)
        return np.sum(distances[np.arange(len(X)), self.labels_])
    
    def silhouette_score(self, X):
        """
        计算聚类结果的轮廓系数
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            score: 轮廓系数，范围为[-1, 1]，越大越好
        """
        if len(np.unique(self.labels_)) < 2:
            return -1  # 只有一个簇时无法计算
        return silhouette_score(X, self.labels_)
    
    def davies_bouldin_score(self, X):
        """
        计算聚类结果的Davies-Bouldin指数
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            score: Davies-Bouldin指数，越小越好
        """
        if len(np.unique(self.labels_)) < 2:
            return float('inf')  # 只有一个簇时无法计算
        return davies_bouldin_score(X, self.labels_)

    def plot_clusters_2d(self, X, features=[0, 1], title="K-means Clustering", save_path=None):
        """
        绘制二维聚类结果图
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
            features: 要绘制的两个特征的索引
            title: 图表标题
            save_path: 保存图片的路径，如果为None则不保存
        """
        plt.figure(figsize=(10, 7))
        
        # 绘制数据点
        for i in range(self.n_clusters):
            plt.scatter(
                X[self.labels_ == i, features[0]], 
                X[self.labels_ == i, features[1]], 
                s=50, label=f'Cluster {i+1}'
            )
        
        # 绘制簇中心
        plt.scatter(
            self.centroids[:, features[0]], 
            self.centroids[:, features[1]], 
            s=200, marker='*', c='red', label='Centroids'
        )
        
        plt.title(title)
        plt.xlabel(f'Feature {features[0]+1}')
        plt.ylabel(f'Feature {features[1]+1}')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()