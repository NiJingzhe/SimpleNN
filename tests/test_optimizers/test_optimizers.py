import numpy as np
import pytest
from SimpleNN.optimizer import SGD, Adam

def test_sgd():
    """测试SGD优化器"""
    sgd = SGD(lr=0.01)
    params = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
    grads = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.5, 0.6])]
    
    # 更新参数
    new_params = sgd.update(params, grads)
    
    # 验证参数更新
    expected_params = [
        np.array([[1 - 0.01 * 0.1, 2 - 0.01 * 0.2], [3 - 0.01 * 0.3, 4 - 0.01 * 0.4]]),
        np.array([5 - 0.01 * 0.5, 6 - 0.01 * 0.6])
    ]
    for p, ep in zip(new_params, expected_params):
        assert np.allclose(p, ep)

def test_adam():
    """测试Adam优化器"""
    adam = Adam(lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)
    params = [np.array([[1, 2], [3, 4]]), np.array([5, 6])]
    grads = [np.array([[0.1, 0.2], [0.3, 0.4]]), np.array([0.5, 0.6])]
    
    # 第一次更新
    new_params = adam.update(params, grads)
    
    # 验证参数更新
    for p, g in zip(new_params, grads):
        assert p.shape == g.shape
    
    # 第二次更新
    new_grads = [np.array([[0.2, 0.3], [0.4, 0.5]]), np.array([0.6, 0.7])]
    new_params = adam.update(new_params, new_grads)
    
    # 验证参数更新
    for p, g in zip(new_params, new_grads):
        assert p.shape == g.shape

def test_optimizer_learning_rate():
    """测试优化器的学习率"""
    # 测试SGD
    sgd = SGD(lr=0.1)
    params = [np.array([[1, 2], [3, 4]])]
    grads = [np.array([[1, 1], [1, 1]])]
    
    new_params = sgd.update(params, grads)
    assert np.allclose(new_params[0], np.array([[0.9, 1.9], [2.9, 3.9]]))
    
    # 测试Adam
    adam = Adam(lr=0.1)
    new_params = adam.update(params, grads)
    assert not np.allclose(new_params[0], params[0])
