import numpy as np
import pytest
from SimpleNN.F import ReLU, Sigmoid, Tanh, Softmax

def test_relu():
    """测试ReLU激活函数"""
    relu = ReLU()
    x = np.array([[-1, 0, 1], [2, -2, 3]])
    
    # 前向传播
    output = relu.forward(x)
    expected = np.array([[0, 0, 1], [2, 0, 3]])
    assert np.allclose(output, expected)
    
    # 反向传播
    grad = np.array([[1, 2, 3], [4, 5, 6]])
    dx = relu.backward(grad)
    expected_grad = np.array([[0, 0, 3], [4, 0, 6]])
    assert np.allclose(dx, expected_grad)

def test_sigmoid():
    """测试Sigmoid激活函数"""
    sigmoid = Sigmoid()
    x = np.array([[-1, 0, 1], [2, -2, 3]])
    
    # 前向传播
    output = sigmoid.forward(x)
    expected = 1 / (1 + np.exp(-x))
    assert np.allclose(output, expected)
    
    # 反向传播
    grad = np.array([[1, 2, 3], [4, 5, 6]])
    dx = sigmoid.backward(grad)
    expected_grad = grad * output * (1 - output)
    assert np.allclose(dx, expected_grad)

def test_tanh():
    """测试Tanh激活函数"""
    tanh = Tanh()
    x = np.array([[-1, 0, 1], [2, -2, 3]])
    
    # 前向传播
    output = tanh.forward(x)
    expected = np.tanh(x)
    assert np.allclose(output, expected)
    
    # 反向传播
    grad = np.array([[1, 2, 3], [4, 5, 6]])
    dx = tanh.backward(grad)
    expected_grad = grad * (1 - output ** 2)
    assert np.allclose(dx, expected_grad)

def test_softmax():
    """测试Softmax激活函数"""
    softmax = Softmax()
    x = np.array([[1, 2, 3], [4, 5, 6]])
    
    # 前向传播
    output = softmax.forward(x)
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    expected = exp_x / np.sum(exp_x, axis=1, keepdims=True)
    assert np.allclose(output, expected)
    
    # 反向传播
    grad = np.array([[1, 2, 3], [4, 5, 6]])
    dx = softmax.backward(grad)
    expected_grad = output * (grad - np.sum(grad * output, axis=1, keepdims=True))
    assert np.allclose(dx, expected_grad)

def test_softmax_numerical_stability():
    """测试Softmax的数值稳定性"""
    softmax = Softmax()
    x = np.array([[1000, 1001, 1002], [1003, 1004, 1005]])
    
    # 前向传播应该不会出现数值溢出
    output = softmax.forward(x)
    assert not np.any(np.isnan(output))
    assert not np.any(np.isinf(output))
    assert np.allclose(np.sum(output, axis=1), 1.0) 