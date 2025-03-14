import pytest
import numpy as np

@pytest.fixture
def random_seed():
    """设置随机种子以确保测试的可重复性"""
    np.random.seed(42)
    return 42

@pytest.fixture
def small_dataset():
    """创建小型测试数据集"""
    x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    return x, y

@pytest.fixture
def binary_classification_data():
    """创建二分类测试数据"""
    n_samples = 100
    n_features = 2
    
    # 生成两个类别的数据
    x1 = np.random.normal(0, 1, (n_samples//2, n_features))
    x2 = np.random.normal(2, 1, (n_samples//2, n_features))
    x = np.vstack([x1, x2])
    y = np.array([0] * (n_samples//2) + [1] * (n_samples//2))
    
    return x, y

@pytest.fixture
def multiclass_classification_data():
    """创建多分类测试数据"""
    n_samples = 150
    n_features = 2
    
    # 生成三个类别的数据
    x1 = np.random.normal(0, 1, (n_samples//3, n_features))
    x2 = np.random.normal(2, 1, (n_samples//3, n_features))
    x3 = np.random.normal(-2, 1, (n_samples//3, n_features))
    x = np.vstack([x1, x2, x3])
    y = np.array([0] * (n_samples//3) + [1] * (n_samples//3) + [2] * (n_samples//3))
    
    return x, y

@pytest.fixture
def regression_data():
    """创建回归测试数据"""
    n_samples = 100
    n_features = 2
    
    # 生成带有噪声的线性数据
    x = np.random.normal(0, 1, (n_samples, n_features))
    y = np.sum(x, axis=1) + np.random.normal(0, 0.1, n_samples)
    
    return x, y 