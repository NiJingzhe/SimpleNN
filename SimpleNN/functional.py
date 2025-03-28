import numpy as np
from typing import Optional
from .layer import Layer

class ReLU(Layer):
    """ReLU 
   
    :math:`ReLU(x) = max(0, x)`
    
    And the gradient should be represented as:
    
    :math:`ReLU'(x) = 1 if x > 0 else 0`
   
    
    """
    def __init__(self):
        """初始化ReLU层"""
        super().__init__()
        self._name = "ReLU"
        self.x: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.x = x
        return np.maximum(0, x)
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        if self.x is None:
            raise ValueError("请先调用forward方法初始化内部状态")
        return grad * (self.x > 0)

class Sigmoid(Layer):
    """Sigmoid激活函数层

    :math:`Sigmoid(x) = 1 / (1 + e^{-x})`
    
    And the gradient should be represented as:
    
    :math:`Sigmoid'(x) = Sigmoid(x) * (1 - Sigmoid(x))`
    """
    
    def __init__(self):
        """初始化Sigmoid层"""
        super().__init__()
        self._name = "Sigmoid"
        self.out: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.out = 1 / (1 + np.exp(-x))
        return self.out
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        if self.out is None:
            raise ValueError("请先调用forward方法初始化内部状态")
        return grad * self.out * (1 - self.out)

class Tanh(Layer):
    """Tanh激活函数层"""
    
    def __init__(self):
        """初始化Tanh层"""
        super().__init__()
        self._name = "Tanh"
        self.out: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        self.out = np.tanh(x)
        return self.out
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        if self.out is None:
            raise ValueError("请先调用forward方法初始化内部状态")
        return grad * (1 - self.out**2)

class Softmax(Layer):
    """Softmax激活函数层"""
    
    def __init__(self, axis=-1):
        """初始化Softmax层
        
        Args:
            axis: 对哪个轴计算Softmax
        """
        super().__init__()
        self._name = "Softmax"
        self.axis = axis
        self.out: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray) -> np.ndarray:
        """前向传播"""
        # 数值稳定性处理
        shifted_x = x - np.max(x, axis=self.axis, keepdims=True)
        exp_x = np.exp(shifted_x)
        self.out = exp_x / np.sum(exp_x, axis=self.axis, keepdims=True)
        if self.out is None:
            raise ValueError("Softmax: 输出为None")
        return self.out
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        if self.out is None:
            raise ValueError("请先调用forward方法初始化内部状态")
            
        # Softmax的梯度计算比较复杂，这里简化处理
        # 当与交叉熵损失一起使用时，梯度简化为 dL/dx = softmax(x) - y
        # 这种情况下，通常会将Softmax集成到损失函数中
        return grad * self.out * (1 - self.out)  # 这是一个近似

class Dropout(Layer):
    """Dropout层，用于防止过拟合"""
    
    def __init__(self, rate=0.5):
        """初始化Dropout层
        
        Args:
            rate: 丢弃率，0到1之间
        """
        super().__init__()
        self._name = "Dropout"
        self.rate = rate
        self.mask: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """前向传播"""
        if training:
            self.mask = np.random.binomial(1, 1 - self.rate, size=x.shape) / (1 - self.rate)
            return x * self.mask
        else:
            return x
            
    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        if self.mask is None:
            raise ValueError("请先在训练模式下调用forward方法初始化内部状态")
        return grad * self.mask
