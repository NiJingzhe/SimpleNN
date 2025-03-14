import numpy as np
import pytest
from SimpleNN.metric import Accuracy, F1Score

def test_accuracy():
    """测试准确率指标"""
    accuracy = Accuracy()
    
    # 测试二分类
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[1, 0], [0.8, 0.9], [0.7, 0.3], [0.2, 0.8]])  # 直接使用类别索引
    
    assert accuracy(y_pred, y_true) == pytest.approx(1.0, abs=1e-5)
        
    y_pred = np.array([[0, 1], [0, 1], [1, 0], [0, 1]])
    assert accuracy(y_pred, y_true) == pytest.approx(0.75, abs=1e-5)
    
    # 测试多分类
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert accuracy(y_pred, y_true) == pytest.approx(1.0, abs=1e-5)
    
    y_pred = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert accuracy(y_pred, y_true) == pytest.approx(0.75, abs=1e-5)

def test_f1_score():
    """测试F1分数指标"""
    f1 = F1Score()
    
    # 测试二分类
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[1, 0], [0.8, 0.9], [0.7, 0.3], [0.2, 0.8]])
    assert f1(y_pred, y_true) == pytest.approx(1.0, abs=1e-5)
    
    y_pred = np.array([[0, 1], [0, 1], [1, 0], [0, 1]])  # 1, 1, 0, 1
    
    y_pred_classes = np.argmax(y_pred, axis=1)
    # 计算TP, FP, FN
    TP = np.sum((y_pred_classes == 1) & (y_true == 1))
    FP = np.sum((y_pred_classes == 1) & (y_true == 0))
    FN = np.sum((y_pred_classes == 0) & (y_true == 1))
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    expected_f1 = 2 * (precision * recall) / (precision + recall)
    assert f1(y_pred, y_true) == pytest.approx(expected_f1, abs=1e-5)
    
    # 测试多分类
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert f1(y_pred, y_true) == pytest.approx(1.0, abs=1e-5)
    
    y_pred = np.array([[0, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    assert f1(y_pred, y_true) == pytest.approx(2/3, abs=1e-5)

def test_metrics_edge_cases():
    """测试评估指标的边界情况"""
    accuracy = Accuracy()
    f1 = F1Score()
    
    # 测试空预测
    y_true = np.array([])
    y_pred = np.array([]).reshape(0, 2)  # 确保y_pred是二维的
    assert np.isnan(accuracy(y_pred, y_true))
    assert np.isnan(f1(y_pred, y_true))
    
    # 测试样本数量不一致raise  error
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[1, 0], [0, 1]])
    with pytest.raises(ValueError):
        accuracy(y_pred, y_true)
    with pytest.raises(ValueError):
        f1(y_pred, y_true)
   
    
    # 测试全错预测
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([[0, 1], [1, 0], [0, 1], [1, 0]])
    assert accuracy(y_pred, y_true) == pytest.approx(0.0, abs=1e-5)
    assert f1(y_pred, y_true) == pytest.approx(0.0, abs=1e-5) 