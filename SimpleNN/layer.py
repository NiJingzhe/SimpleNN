import numpy as np
from typing import List, Callable, Optional, Union, Any, cast
from .optimizer import Optimizer

class Layer:
    """神经网络层的基类，定义了所有层必须实现的接口"""
    
    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
        raise NotImplementedError
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        raise NotImplementedError
        
    def update(self, optimizer: Optional[Optimizer] = None) -> None:
        """参数更新"""
        pass

class Dense(Layer):
    """全连接层"""
    
    def __init__(self, input_dim: int, output_dim: int):
        # He初始化
        self.W: np.ndarray = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        self.b: np.ndarray = np.zeros((1, output_dim))
        self.x: Optional[np.ndarray] = None
        self.dW: Optional[np.ndarray] = None
        self.db: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
        self.x = x
        return x @ self.W + self.b
        
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
            
        self.dW = self.x.T @ grad   
        # x has shape (batch_size, input_dim), grad has shape (batch_size, output_dim)
        # x.T @ grad has shape (input_dim, output_dim), match the shape of W
        self.db = np.sum(grad, axis=0, keepdims=True)
        return grad @ self.W.T
        
    def update(self, optimizer: Optional[Optimizer] = None) -> None:
        """使用优化器更新参数"""
        # 确保梯度已经计算
        if self.dW is None or self.db is None or optimizer is None:
            return
            
        # 分别更新权重和偏置
        params = [self.W, self.b]
        grads = [self.dW, self.db]
        updated_params = optimizer.update(params, grads)
        
        # 更新参数
        self.W = updated_params[0]
        self.b = updated_params[1]

class BatchNorm(Layer):
    """批量归一化层"""
    
    def __init__(self, dim: int, eps: float = 1e-5, momentum: float = 0.9):
        self.gamma: np.ndarray = np.ones((1, dim))  # 缩放参数
        self.beta: np.ndarray = np.zeros((1, dim))  # 平移参数
        self.eps: float = eps  # 数值稳定性参数
        self.momentum: float = momentum  # 动量参数
        
        # 运行时参数
        self.running_mean: np.ndarray = np.zeros((1, dim))
        self.running_var: np.ndarray = np.ones((1, dim))
        
        # 反向传播所需缓存
        self.x_norm: Optional[np.ndarray] = None
        self.x: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.dgamma: Optional[np.ndarray] = None
        self.dbeta: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
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
            return self.gamma * self.x_norm + self.beta
        else:
            # 测试阶段使用运行时统计量
            x_norm = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
            return self.gamma * x_norm + self.beta
    
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播
        
        数学推导:
        前向传播: 
        x_norm = (x - μ) / sqrt(σ² + ε)
        y = γ * x_norm + β
        
        对γ的梯度:
        dL/dγ = dL/dy * dy/dγ = dL/dy * x_norm = sum(dL/dy * x_norm, axis=0)
        
        对β的梯度:
        dL/dβ = dL/dy * dy/dβ = dL/dy * 1 = sum(dL/dy, axis=0)
        
        对x_norm的梯度:
        dL/dx_norm = dL/dy * dy/dx_norm = dL/dy * γ
        
        对方差σ²的梯度:
        dL/dσ² = dL/dx_norm * dx_norm/dσ² = sum(dL/dx_norm * (x - μ) * (-0.5) * (σ² + ε)^(-1.5), axis=0)
        
        对均值μ的梯度:
        dL/dμ = dL/dx_norm * dx_norm/dμ + dL/dσ² * dσ²/dμ
              = sum(dL/dx_norm * (-1/sqrt(σ² + ε)), axis=0) + dL/dσ² * sum(-2(x - μ), axis=0)/N
        
        对输入x的梯度:
        dL/dx = dL/dx_norm * dx_norm/dx + dL/dσ² * dσ²/dx + dL/dμ * dμ/dx
              = dL/dx_norm / sqrt(σ² + ε) + dL/dσ² * 2(x - μ)/N + dL/dμ * 1/N
        """
        if self.x is None or self.mu is None or self.var is None or self.x_norm is None:
            raise ValueError("请先调用forward方法初始化内部状态")
            
        # 确保所有变量都不为None
        x = self.x
        mu = self.mu
        var = self.var
        x_norm = self.x_norm
        eps = self.eps
        gamma = self.gamma
        
        N = x.shape[0]
        
        # 计算gamma和beta的梯度
        self.dgamma = np.sum(grad * x_norm, axis=0, keepdims=True)
        self.dbeta = np.sum(grad, axis=0, keepdims=True)
        
        # 计算x_norm的梯度
        dx_norm = grad * gamma
        
        # 计算方差的梯度
        dvar = np.sum(dx_norm * (x - mu) * -0.5 * np.power(var + eps, -1.5), axis=0, keepdims=True)
        
        # 计算均值的梯度
        dmu = np.sum(dx_norm * -1 / np.sqrt(var + eps), axis=0, keepdims=True) + dvar * np.mean(-2 * (x - mu), axis=0, keepdims=True)
        
        # 计算输入x的梯度
        dx = dx_norm / np.sqrt(var + eps) + dvar * 2 * (x - mu) / N + dmu / N
        
        return dx
    
    def update(self, optimizer: Optional[Optimizer] = None) -> None:
        """使用优化器更新参数"""
        # 确保梯度已经计算
        if self.dgamma is None or self.dbeta is None or optimizer is None:
            return
            
        # 分别更新gamma和beta
        params = [self.gamma, self.beta]
        grads = [self.dgamma, self.dbeta]
        updated_params = optimizer.update(params, grads)
        
        # 更新参数
        self.gamma = updated_params[0]
        self.beta = updated_params[1]
