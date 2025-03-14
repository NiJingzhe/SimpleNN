import numpy as np
import pytest
from SimpleNN.loss import MSE, SoftmaxCrossEntropy, BinaryCrossEntropy

def test_mse():
    """测试MSE损失函数"""
    mse = MSE()
    y_pred = np.array([[1, 2, 3], [4, 5, 6]])
    y_true = np.array([[1, 2, 3], [4, 5, 6]])
    
    # 计算损失
    loss = mse.forward(y_pred, y_true)
    expected_loss = 0.0
    assert np.allclose(loss, expected_loss)
    
    # 计算梯度
    grad = mse.backward()
    expected_grad = 2 * (y_pred - y_true) / y_pred.size
    assert np.allclose(grad, expected_grad)

def test_softmax_cross_entropy():
    """测试Softmax交叉熵损失函数"""
    sce = SoftmaxCrossEntropy()
    y_pred = np.array([[1, 2, 3], [4, 5, 6]])
    y_true = np.array([0, 2])  # 类别索引
    
    # 计算损失
    loss = sce.forward(y_pred, y_true)
    exp_x = np.exp(y_pred - np.max(y_pred, axis=1, keepdims=True))
    probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    expected_loss = -np.mean(np.log(probs[range(2), y_true]))
    assert np.allclose(loss, expected_loss)
    
    # 计算梯度
    grad = sce.backward()
    expected_grad = probs.copy()
    expected_grad[range(2), y_true] -= 1
    expected_grad /= y_pred.shape[0]
    assert np.allclose(grad, expected_grad)

def test_binary_cross_entropy():
    """测试二元交叉熵损失函数"""
    bce = BinaryCrossEntropy()
    y_pred = np.array([[0.1, 0.9], [0.8, 0.2]])
    y_true = np.array([[0, 1], [1, 0]])
    
    # 计算损失
    loss = bce.forward(y_pred, y_true)
    expected_loss = -np.mean(y_true * np.log(y_pred + 1e-7) + 
                           (1 - y_true) * np.log(1 - y_pred + 1e-7))
    assert np.allclose(loss, expected_loss)
    
    # 计算梯度
    grad = bce.backward()
    expected_grad = (y_pred - y_true) / (y_pred * (1 - y_pred) + 1e-7)
    expected_grad /= y_pred.shape[0]
    assert np.allclose(grad, expected_grad)

def test_loss_numerical_stability():
    """测试损失函数的数值稳定性"""
    # 测试MSE
    mse = MSE()
    y_pred = np.array([[1e6, 1e6], [1e6, 1e6]])
    y_true = np.array([[1e6, 1e6], [1e6, 1e6]])
    loss = mse.forward(y_pred, y_true)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    
    # 测试Softmax交叉熵
    sce = SoftmaxCrossEntropy()
    y_pred = np.array([[1e6, 1e6, 1e6], [1e6, 1e6, 1e6]])
    y_true = np.array([0, 1])
    loss = sce.forward(y_pred, y_true)
    assert not np.isnan(loss)
    assert not np.isinf(loss)
    
    # 测试二元交叉熵
    bce = BinaryCrossEntropy()
    y_pred = np.array([[1e-7, 1-1e-7], [1e-7, 1-1e-7]])
    y_true = np.array([[0, 1], [0, 1]])
    loss = bce.forward(y_pred, y_true)
    assert not np.isnan(loss)
    assert not np.isinf(loss) 