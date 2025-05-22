import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score

class FCM:
    def __init__(self, n_clusters=3, max_iter=100, m=2, error=1e-5, random_state=None):
        """
        FCM（模糊c均值）聚类算法实现
        
        参数:
            n_clusters: 簇的数量
            max_iter: 最大迭代次数
            m: 模糊指数，控制聚类的模糊度，通常取值为2
            error: 收敛阈值
            random_state: 随机数种子，用于结果复现
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.m = m
        self.error = error
        self.random_state = random_state
        self.centroids = None
        self.membership = None  # 隶属度矩阵
        self.labels_ = None     # 硬聚类标签
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
        
        n_samples = X.shape[0]
        
        # 初始化隶属度矩阵（随机生成，满足每个样本对所有簇的隶属度之和为1）
        self.membership = np.random.rand(n_samples, self.n_clusters)
        self.membership = self.membership / np.sum(self.membership, axis=1, keepdims=True)
        
        # 迭代优化
        for i in range(self.max_iter):
            # 更新簇中心
            self._update_centroids(X)
            
            # 保存旧的隶属度矩阵
            old_membership = self.membership.copy()
            
            # 更新隶属度矩阵
            self._update_membership(X)
            
            # 检查收敛性
            self.iterations = i + 1
            if np.linalg.norm(self.membership - old_membership) < self.error:
                break
        
        # 计算硬聚类标签（取隶属度最大的簇）
        self.labels_ = np.argmax(self.membership, axis=1)
        
        # 计算簇内误差平方和（inertia）
        self.inertia_ = self._calc_inertia(X)
        
        return self
    
    def _update_centroids(self, X):
        """
        更新簇中心
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        """
        # 计算每个样本对每个簇的贡献（隶属度的m次方）
        weights = self.membership ** self.m
        
        # 计算簇中心（加权平均）
        self.centroids = np.dot(weights.T, X) / np.sum(weights, axis=0, keepdims=True).T
    
    def _update_membership(self, X):
        """
        更新隶属度矩阵
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        """
        for i in range(X.shape[0]):
            distances = np.sum((X[i] - self.centroids) ** 2, axis=1)
            
            # 处理零距离的特殊情况
            if np.any(distances == 0):
                self.membership[i] = 0
                self.membership[i, distances == 0] = 1 / np.sum(distances == 0)
            else:
                # 计算隶属度
                for j in range(self.n_clusters):
                    sum_term = 0
                    for k in range(self.n_clusters):
                        sum_term += (distances[j] / distances[k]) ** (1 / (self.m - 1))
                    self.membership[i, j] = 1 / sum_term
    
    def predict(self, X):
        """
        预测新数据的簇标签
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            labels: 每个样本的簇标签（硬聚类结果）
        """
        # 计算距离矩阵
        n_samples = X.shape[0]
        membership = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            distances = np.sum((X[i] - self.centroids) ** 2, axis=1)
            
            if np.any(distances == 0):
                membership[i] = 0
                membership[i, distances == 0] = 1
            else:
                sum_dist = np.sum([(distances[i] / distances[j]) ** (1 / (self.m - 1)) 
                                  for j in range(self.n_clusters)])
                
                for j in range(self.n_clusters):
                    membership[i, j] = 1 / sum_dist
        
        return np.argmax(membership, axis=1)
    
    def predict_membership(self, X):
        """
        预测新数据的隶属度矩阵
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            membership: 隶属度矩阵
        """
        n_samples = X.shape[0]
        membership = np.zeros((n_samples, self.n_clusters))
        
        for i in range(n_samples):
            distances = np.sum((X[i] - self.centroids) ** 2, axis=1)
            
            if np.any(distances == 0):
                membership[i] = 0
                membership[i, distances == 0] = 1 / np.sum(distances == 0)
            else:
                for j in range(self.n_clusters):
                    sum_term = 0
                    for k in range(self.n_clusters):
                        sum_term += (distances[j] / distances[k]) ** (1 / (self.m - 1))
                    membership[i, j] = 1 / sum_term
        
        return membership
    
    def _calc_inertia(self, X):
        """
        计算簇内误差平方和（惯性）- 加权版本
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            inertia: 簇内误差平方和
        """
        inertia = 0
        for i in range(X.shape[0]):
            for j in range(self.n_clusters):
                # 欧氏距离的平方 * 隶属度的m次方
                inertia += (self.membership[i, j] ** self.m) * np.sum((X[i] - self.centroids[j]) ** 2)
        
        return inertia
    
    def silhouette_score(self, X):
        """
        计算聚类结果的轮廓系数（基于硬聚类结果）
        
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
        计算聚类结果的Davies-Bouldin指数（基于硬聚类结果）
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
        
        返回:
            score: Davies-Bouldin指数，越小越好
        """
        if len(np.unique(self.labels_)) < 2:
            return float('inf')  # 只有一个簇时无法计算
        return davies_bouldin_score(X, self.labels_)
    
    def plot_clusters_2d(self, X, features=[0, 1], title="FCM Clustering", save_path=None):
        """
        绘制二维聚类结果图
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
            features: 要绘制的两个特征的索引
            title: 图表标题
            save_path: 保存图片的路径，如果为None则不保存
        """
        plt.figure(figsize=(10, 7))
        
        # 绘制数据点，颜色根据隶属度混合
        x = X[:, features[0]]
        y = X[:, features[1]]
        
        # 使用硬聚类标签绘制
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
        
    def plot_membership_2d(self, X, features=[0, 1], title="FCM Membership", save_path=None):
        """
        绘制隶属度的二维可视化图
        
        参数:
            X: 形状为 (n_samples, n_features) 的数据
            features: 要绘制的两个特征的索引
            title: 图表标题
            save_path: 保存图片的路径，如果为None则不保存
        """
        fig, axes = plt.subplots(1, self.n_clusters, figsize=(5*self.n_clusters, 4))
        
        if self.n_clusters == 1:
            axes = [axes]
            
        for i in range(self.n_clusters):
            scatter = axes[i].scatter(
                X[:, features[0]], 
                X[:, features[1]], 
                c=self.membership[:, i],
                cmap='viridis', 
                s=50, 
                alpha=0.8
            )
            
            # 绘制簇中心
            axes[i].scatter(
                self.centroids[i, features[0]], 
                self.centroids[i, features[1]], 
                s=200, marker='*', c='red'
            )
            
            axes[i].set_title(f'Cluster {i+1} Membership')
            axes[i].set_xlabel(f'Feature {features[0]+1}')
            axes[i].set_ylabel(f'Feature {features[1]+1}')
            axes[i].grid(True, linestyle='--', alpha=0.7)
            
            fig.colorbar(scatter, ax=axes[i])
        
        plt.suptitle(title)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()