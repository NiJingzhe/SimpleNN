import numpy as np
from typing import List, Callable, Optional, Union, Any, cast, Dict, Set
from .optimizer import Optimizer
from .parameter import Parameter

class Layer:
    """神经网络层的基类，定义了所有层必须实现的接口"""
    
    def __init__(self):
        """初始化层"""
        self._parameters: Dict[str, Parameter] = {}
    
    def register_parameter(self, name: str, param: Parameter) -> None:
        """注册参数
        
        Args:
            name: 参数名称
            param: 参数对象
        """
        self._parameters[name] = param
    
    def parameters(self) -> Dict[str, Parameter]:
        """获取层的所有参数
        
        Returns:
            参数字典
        """
        return self._parameters
    
    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
        raise NotImplementedError
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        raise NotImplementedError
        
    def update(self, optimizer: Optional[Optimizer] = None) -> None:
        """参数更新"""
        pass  # 在新设计中，参数更新由优化器统一管理

class Dense(Layer):
    """全连接层"""
    
    def __init__(self, input_dim: int, output_dim: int):
        """初始化全连接层
        
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
        """
        super().__init__()
        # He初始化
        W_data = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        b_data = np.zeros((1, output_dim))
        
        # 创建参数并注册
        self.register_parameter('W', Parameter(W_data))
        self.register_parameter('b', Parameter(b_data))
        
        self.x: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
        self.x = x
        W = self._parameters['W'].data
        b = self._parameters['b'].data
        return x @ W + b
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播
        
        数学推导:
        前向传播: y = x @ W + b
        
        对W的梯度: 
        dL/dW = dL/dy * dy/dW = x^T * dL/dy
        其中dL/dy就是输入的grad参数
        
        对b的梯度:
        dL/db = dL/dy * dy/db = dL/dy * 1 = sum(dL/dy, axis=0)
        
        对x的梯度:
        dL/dx = dL/dy * dy/dx = dL/dy * W^T
        """
        if self.x is None:
            raise ValueError("无法对没有输入的层进行反向传播")
            
        W = self._parameters['W'].data
        
        # 计算梯度
        dW = self.x.T @ grad
        db = np.sum(grad, axis=0, keepdims=True)
        
        # 设置梯度
        self._parameters['W'].grad = dW
        self._parameters['b'].grad = db
        
        return grad @ W.T

class BatchNorm(Layer):
    """批量归一化层"""
    
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.9):
        """初始化批量归一化层
        
        Args:
            dim: 特征维度
            eps: 数值稳定性参数
            momentum: 动量参数
        """
        super().__init__()
        
        # 创建参数并注册
        self.register_parameter('gamma', Parameter(np.ones((1, dim))))
        self.register_parameter('beta', Parameter(np.zeros((1, dim))))
        
        self.eps = eps
        self.momentum = momentum
        
        # 运行时参数（不参与梯度计算）
        self.running_mean = np.zeros((1, dim))
        self.running_var = np.ones((1, dim))
        
        # 反向传播所需缓存
        self.x: Optional[np.ndarray] = None
        self.x_norm: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
        gamma = self._parameters['gamma'].data
        beta = self._parameters['beta'].data
        
        if training:
            self.x = x
            self.mu = np.mean(x, axis=0, keepdims=True)
            self.var = np.var(x, axis=0, keepdims=True)
            
            # 使用断言保证类型检查器理解这些变量不为None
            assert self.running_mean is not None, "running_mean 不能为None"
            assert self.running_var is not None, "running_var 不能为None"
            assert self.mu is not None, "mu 不能为None"
            assert self.var is not None, "var 不能为None"
            
            momentum = float(self.momentum)
            self.running_mean = momentum * self.running_mean + (1.0 - momentum) * self.mu
            self.running_var = momentum * self.running_var + (1.0 - momentum) * self.var
            
            # 归一化
            self.x_norm = (x - self.mu) / np.sqrt(self.var + self.eps)
            return gamma * self.x_norm + beta
        else:
            # 测试阶段使用运行时统计量
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return gamma * x_norm + beta
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        if self.x is None or self.mu is None or self.var is None or self.x_norm is None:
            raise ValueError("请先调用forward方法初始化内部状态")
            
        gamma = self._parameters['gamma'].data
        
        x = self.x
        mu = self.mu
        var = self.var
        x_norm = self.x_norm
        eps = self.eps
        
        N = x.shape[0]
        
        # 计算gamma和beta的梯度
        dgamma = np.sum(grad * x_norm, axis=0, keepdims=True)
        dbeta = np.sum(grad, axis=0, keepdims=True)
        
        # 设置梯度
        self._parameters['gamma'].grad = dgamma
        self._parameters['beta'].grad = dbeta
        
        # 计算x_norm的梯度
        dx_norm = grad * gamma
        
        # 计算方差的梯度
        dvar = np.sum(dx_norm * (x - mu) * -0.5 * np.power(var + eps, -1.5), axis=0, keepdims=True)
        
        # 计算均值的梯度
        dmu = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mu), axis=0, keepdims=True)
        
        # 计算输入x的梯度
        dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mu) / N + dmu / N
        
        return dx
