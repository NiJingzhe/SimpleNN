"""
SimpleNN - 一个基于NumPy的简单静态计算图神经网络框架
"""

# 导入层模块
from .layer import Layer, Dense, BatchNorm

# 导入激活函数模块
from .functional import ReLU, Sigmoid, Tanh, Softmax, Dropout

# 导入损失函数模块
from .loss import (
    Loss, 
    MSE, 
    SoftmaxCrossEntropy, 
    CrossEntropy, 
    BinaryCrossEntropy,
    L1Loss,
    HuberLoss
)

# 导入优化器模块
from .optimizer import (
    Optimizer,
    SGD,
    MomentumSGD,
    Adam,
    RMSprop
)

# 导入指标模块
from .metric import (
    Metric,
    Accuracy,
    F1Score,
)

# 导入模型模块
from .model import Model

# 版本信息
__version__ = '0.1.0'
