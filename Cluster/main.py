import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import os

# 导入自定义聚类算法
from kmeans import KMeans
from fcm import FCM

# 创建结果目录
os.makedirs('results', exist_ok=True)

# 加载数据集
def load_data(filepath):
    """
    加载数据集并进行预处理
    
    参数:
        filepath: 数据集文件路径
    
    返回:
        X: 特征数据
        y: 标签
        X_scaled: 标准化后的特征数据
    """
    # 手动读取文件以处理不规则格式
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            # 跳过注释行
            if line.strip().startswith('//'):
                continue
            # 使用空白字符分割
            values = [float(val.strip()) for val in line.strip().split() if val.strip()]
            if len(values) == 8:  # 确保有8个值（7个特征+1个标签）
                data.append(values)
    
    # 转换为numpy数组
    data = np.array(data)
    
    # 分离特征和标签
    X = data[:, :-1]  # 所有特征
    y = data[:, -1].astype(int)  # 标签
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X, y, X_scaled

# 可视化原始数据
def visualize_original_data(X, y, save_path=None):
    """
    使用PCA可视化原始数据
    
    参数:
        X: 特征数据
        y: 标签
        save_path: 保存图片的路径，如果为None则不保存
    """
    # 使用PCA降维到2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    
    # 绘制数据点，按真实标签着色
    plt.figure(figsize=(10, 7))
    for i in np.unique(y):
        plt.scatter(
            X_pca[y == i, 0], 
            X_pca[y == i, 1], 
            s=50, label=f'Class {i}'
        )
    
    plt.title('PCA Visualization of Seeds Dataset')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # 返回PCA结果供后续使用
    return X_pca, pca.explained_variance_ratio_

# 评估聚类性能
def evaluate_clustering(X, y, kmeans_labels, fcm_labels):
    """
    评估聚类性能
    
    参数:
        X: 特征数据
        y: 真实标签
        kmeans_labels: KMeans聚类标签
        fcm_labels: FCM聚类标签
    
    返回:
        results: 包含评估指标的字典
    """
    results = {}
    
    # 调整标签以匹配真实标签（因为聚类标签是任意的）
    # 这里简单起见，我们直接比较聚类结果与真实标签
    
    # KMeans评估
    results['kmeans_ari'] = adjusted_rand_score(y, kmeans_labels)
    results['kmeans_nmi'] = normalized_mutual_info_score(y, kmeans_labels)
    
    # FCM评估
    results['fcm_ari'] = adjusted_rand_score(y, fcm_labels)
    results['fcm_nmi'] = normalized_mutual_info_score(y, fcm_labels)
    
    return results

# 主函数
def main():
    # 设置随机种子以便结果可复现
    np.random.seed(42)
    
    # 加载数据
    data_path = 'data/seeds_dataset.txt'
    X, y, X_scaled = load_data(data_path)
    
    print(f"数据集形状: {X.shape}")
    print(f"类别数量: {len(np.unique(y))}")
    
    # 可视化原始数据
    X_pca, explained_variance = visualize_original_data(X_scaled, y, save_path='results/original_data_pca.png')
    print(f"PCA解释方差比例: {explained_variance}")
    
    # 硬C均值聚类 (KMeans)
    n_clusters = len(np.unique(y))  # 使用与真实类别数相同的簇数
    kmeans = KMeans(n_clusters=n_clusters, max_iter=100, random_state=42)
    kmeans.fit(X_scaled)
    
    print("\n硬C均值聚类 (KMeans) 结果:")
    print(f"迭代次数: {kmeans.iterations}")
    print(f"簇内误差平方和: {kmeans.inertia_:.4f}")
    print(f"轮廓系数: {kmeans.silhouette_score(X_scaled):.4f}")
    print(f"Davies-Bouldin指数: {kmeans.davies_bouldin_score(X_scaled):.4f}")
    
    # 使用原始特征绘制KMeans聚类结果
    kmeans.plot_clusters_2d(X_scaled, features=[0, 1], 
                           title="KMeans Clustering (Features 1 & 2)", 
                           save_path='results/kmeans_features_1_2.png')
    
    kmeans.plot_clusters_2d(X_scaled, features=[2, 3], 
                           title="KMeans Clustering (Features 3 & 4)", 
                           save_path='results/kmeans_features_3_4.png')
    
    # 使用PCA降维结果绘制KMeans聚类
    kmeans_pca = KMeans(n_clusters=n_clusters, max_iter=100, random_state=42)
    kmeans_pca.fit(X_pca)
    kmeans_pca.plot_clusters_2d(X_pca, features=[0, 1], 
                               title="KMeans Clustering (PCA)", 
                               save_path='results/kmeans_pca.png')
    
    # 模糊C均值聚类 (FCM)
    fcm = FCM(n_clusters=n_clusters, max_iter=100, m=2, error=1e-5, random_state=42)
    fcm.fit(X_scaled)
    
    print("\n模糊C均值聚类 (FCM) 结果:")
    print(f"迭代次数: {fcm.iterations}")
    print(f"簇内误差平方和: {fcm.inertia_:.4f}")
    print(f"轮廓系数: {fcm.silhouette_score(X_scaled):.4f}")
    print(f"Davies-Bouldin指数: {fcm.davies_bouldin_score(X_scaled):.4f}")
    
    # 使用原始特征绘制FCM聚类结果
    fcm.plot_clusters_2d(X_scaled, features=[0, 1], 
                        title="FCM Clustering (Features 1 & 2)", 
                        save_path='results/fcm_features_1_2.png')
    
    fcm.plot_clusters_2d(X_scaled, features=[2, 3], 
                        title="FCM Clustering (Features 3 & 4)", 
                        save_path='results/fcm_features_3_4.png')
    
    # 绘制FCM隶属度
    fcm.plot_membership_2d(X_scaled, features=[0, 1], 
                          title="FCM Membership (Features 1 & 2)", 
                          save_path='results/fcm_membership_1_2.png')
    
    # 使用PCA降维结果绘制FCM聚类
    fcm_pca = FCM(n_clusters=n_clusters, max_iter=100, m=2, error=1e-5, random_state=42)
    fcm_pca.fit(X_pca)
    fcm_pca.plot_clusters_2d(X_pca, features=[0, 1], 
                            title="FCM Clustering (PCA)", 
                            save_path='results/fcm_pca.png')
    
    # 评估聚类性能
    results = evaluate_clustering(X_scaled, y, kmeans.labels_, fcm.labels_)
    
    print("\n聚类性能评估:")
    print(f"KMeans - 调整兰德指数: {results['kmeans_ari']:.4f}")
    print(f"KMeans - 标准化互信息: {results['kmeans_nmi']:.4f}")
    print(f"FCM - 调整兰德指数: {results['fcm_ari']:.4f}")
    print(f"FCM - 标准化互信息: {results['fcm_nmi']:.4f}")
    
    # 比较两种算法
    plt.figure(figsize=(12, 6))
    
    metrics = ['Adjusted Rand Index', 'Normalized Mutual Info']
    kmeans_scores = [results['kmeans_ari'], results['kmeans_nmi']]
    fcm_scores = [results['fcm_ari'], results['fcm_nmi']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, kmeans_scores, width, label='KMeans')
    plt.bar(x + width/2, fcm_scores, width, label='FCM')
    
    plt.xlabel('Evaluation Metrics')
    plt.ylabel('Score')
    plt.title('KMeans vs FCM Clustering Performance')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 保存结果数据
    results_df = pd.DataFrame({
        'Algorithm': ['KMeans', 'FCM'],
        'Iterations': [kmeans.iterations, fcm.iterations],
        'Inertia': [kmeans.inertia_, fcm.inertia_],
        'Silhouette Score': [kmeans.silhouette_score(X_scaled), fcm.silhouette_score(X_scaled)],
        'Davies-Bouldin Index': [kmeans.davies_bouldin_score(X_scaled), fcm.davies_bouldin_score(X_scaled)],
        'Adjusted Rand Index': [results['kmeans_ari'], results['fcm_ari']],
        'Normalized Mutual Info': [results['kmeans_nmi'], results['fcm_nmi']]
    })
    
    results_df.to_csv('results/clustering_results.csv', index=False)
    print("\n结果已保存到 'results/clustering_results.csv'")

if __name__ == "__main__":
    main()