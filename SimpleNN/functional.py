import numpy as np
from .layer import Layer

class ReLU(Layer):
    """ReLU激活函数层"""
    
    def __init__(self):
        self.mask = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.mask = (x > 0)
        return x * self.mask
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        return grad * self.mask

class Sigmoid(Layer):
    """Sigmoid激活函数层"""
    
    def __init__(self):
        self.out = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.out = 1 / (1 + np.exp(-x))
        return self.out
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        return grad * self.out * (1 - self.out)

class Tanh(Layer):
    """Tanh激活函数层"""
    
    def __init__(self):
        self.out = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.out = np.tanh(x)
        return self.out
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        return grad * (1 - self.out**2)

class Softmax(Layer):
    """Softmax激活函数层"""
    
    def __init__(self):
        self.probs = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 数值稳定性处理：减去最大值
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.probs = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.probs
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        # 使用更稳定的梯度计算方式
        return self.probs * (grad - np.sum(grad * self.probs, axis=1, keepdims=True))

class Dropout(Layer):
    """Dropout正则化层"""
    
    def __init__(self, keep_prob=0.5):
        self.keep_prob = keep_prob
        self.mask = None
        
    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
        if training:
            self.mask = np.random.binomial(1, self.keep_prob, size=x.shape) / self.keep_prob
            return x * self.mask
        else:
            return x
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        return grad * self.mask
