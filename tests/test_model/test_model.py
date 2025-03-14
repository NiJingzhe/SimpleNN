import numpy as np
import pytest
from SimpleNN.model import Model
from SimpleNN.layer import Dense
from SimpleNN.functional import ReLU, Sigmoid, Softmax
from SimpleNN.loss import MSE, SoftmaxCrossEntropy
from SimpleNN.optimizer import SGD
from SimpleNN.metric import Accuracy


def create_test_model():
    """创建测试用的模型"""
    model = Model()
    model.add(Dense(2, 4))
    model.add(ReLU())
    model.add(Dense(4, 2))
    model.add(Softmax())
    model.compile(
        optimizer=SGD(lr=0.01), loss=SoftmaxCrossEntropy(), metrics=[Accuracy()]
    )
    return model


def test_model_initialization():
    """测试模型初始化"""
    model = create_test_model()
    assert len(model.layers) == 4
    assert model.optimizer is not None
    assert model.loss_fn is not None
    assert len(model.metrics) == 1


def test_model_forward():
    """测试模型前向传播"""
    model = create_test_model()
    x = np.array([[1, 2], [3, 4]])

    # 前向传播
    output = model.predict(x)

    # 验证输出形状
    assert output.shape == (2, 2)
    # 验证输出是有效的概率分布
    assert np.allclose(np.sum(output, axis=1), 1.0)
    assert np.all(output >= 0) and np.all(output <= 1)



def test_model_update():
    """测试模型参数更新"""
    model = create_test_model()
    x = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    # 保存原始参数
    original_params = []
    for layer in model.layers:
        if hasattr(layer, "W"):
            original_params.append(layer.W.copy())
        if hasattr(layer, "b"):
            original_params.append(layer.b.copy())

    # 训练一个批次
    model._train_step(x, y)

    # 验证参数已更新
    new_params = []
    for layer in model.layers:
        if hasattr(layer, "W"):
            new_params.append(layer.W)
        if hasattr(layer, "b"):
            new_params.append(layer.b)

    for orig, new in zip(original_params, new_params):
        assert not np.allclose(orig, new)


def test_model_evaluate():
    """测试模型评估"""
    model = create_test_model()
    x = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    # 评估模型
    mertric_or_loss = model.evaluate(x, y)
    
    if isinstance(mertric_or_loss, dict):
        loss = mertric_or_loss['loss']
        metrics = mertric_or_loss
    else:
        loss = mertric_or_loss
        metrics = None


    # 验证损失和指标
    assert isinstance(loss, float)
    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert 0 <= metrics["accuracy"] <= 1


def test_model_fit():
    """测试模型训练"""
    model = create_test_model()
    x = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])

    # 训练模型
    history = model.fit(x, y, epochs=2, batch_size=2, validation_data=(x, y))

    # 验证训练历史
    assert isinstance(history, dict)
    assert "loss" in history
    assert "val_loss" in history
    assert "accuracy" in history
    assert "val_accuracy" in history
    assert len(history["loss"]) == 2
    assert len(history["val_loss"]) == 2
    assert len(history["accuracy"]) == 2
    assert len(history["val_accuracy"]) == 2


def test_model_predict():
    """测试模型预测"""
    model = create_test_model()
    x = np.array([[1, 2], [3, 4]])

    # 预测
    predictions = model.predict(x)

    # 验证预测结果
    assert predictions.shape == (2, 2)
    assert np.allclose(np.sum(predictions, axis=1), 1.0)
    assert np.all(predictions >= 0) and np.all(predictions <= 1)
