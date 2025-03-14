import numpy as np
import pytest
from SimpleNN.layer import Dense

def test_dense_initialization():
    """测试Dense层的初始化"""
    dense = Dense(10, 5)
    assert dense.W.shape == (10, 5)
    assert dense.b.shape == (1, 5)
    assert np.allclose(dense.b, np.zeros((1, 5)))

def test_dense_forward():
    """测试Dense层的前向传播"""
    dense = Dense(3, 2)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    output = dense.forward(x)
    assert output.shape == (2, 2)
    assert np.allclose(output, x @ dense.W + dense.b)

def test_dense_backward():
    """测试Dense层的反向传播"""
    dense = Dense(3, 2)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    grad = np.array([[1, 2], [3, 4]])
    
    # 前向传播
    dense.forward(x)
    
    # 反向传播
    dx = dense.backward(grad)
    
    # 检查梯度形状
    assert dense.dW.shape == (3, 2)
    assert dense.db.shape == (1, 2)
    assert dx.shape == (2, 3)
    
    # 检查梯度计算是否正确
    expected_dW = x.T @ grad
    expected_db = np.sum(grad, axis=0, keepdims=True)
    expected_dx = grad @ dense.W.T
    
    assert np.allclose(dense.dW, expected_dW)
    assert np.allclose(dense.db, expected_db)
    assert np.allclose(dx, expected_dx)

def test_dense_update():
    """测试Dense层的参数更新"""
    dense = Dense(3, 2)
    x = np.array([[1, 2, 3], [4, 5, 6]])
    grad = np.array([[1, 2], [3, 4]])
    
    # 前向和反向传播
    dense.forward(x)
    dense.backward(grad)
    
    # 保存原始参数
    original_W = dense.W.copy()
    original_b = dense.b.copy()
    
    # 模拟优化器
    class MockOptimizer:
        def update(self, params, grads):
            return [p - 0.01 * g for p, g in zip(params, grads)]
    
    # 更新参数
    dense.update(MockOptimizer())
    
    # 检查参数是否更新
    assert not np.allclose(dense.W, original_W)
    assert not np.allclose(dense.b, original_b)
    assert np.allclose(dense.W, original_W - 0.01 * dense.dW)
    assert np.allclose(dense.b, original_b - 0.01 * dense.db) 