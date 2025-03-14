import numpy as np
from typing import List

class Optimizer:
    """优化器基类"""
    
    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """更新参数"""
        raise NotImplementedError

class SGD(Optimizer):
    """随机梯度下降优化器"""
    
    def __init__(self, lr=0.01):
        self.lr = lr
        
    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """使用梯度更新参数"""
        return [p - self.lr * g for p, g in zip(params, grads)]

class MomentumSGD(Optimizer):
    """带动量的SGD优化器"""
    
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocities = {}
        self.param_shapes = {}
        
    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """使用动量更新参数"""
        updated_params = []
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # 使用参数形状作为键
            param_shape = p.shape
            param_key = f"param_{i}_{param_shape}"
            
            # 如果这个参数之前没有见过，初始化它的速度
            if param_key not in self.velocities:
                self.velocities[param_key] = np.zeros_like(p)
                
            # 更新速度和参数
            self.velocities[param_key] = self.momentum * self.velocities[param_key] - self.lr * g
            updated_params.append(p + self.velocities[param_key])
            
        return updated_params

class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}  # 一阶动量
        self.v = {}  # 二阶动量
        self.t = 0   # 时间步
        
    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """使用Adam算法更新参数"""
        self.t += 1
        updated_params = []
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # 使用参数形状和索引作为键
            param_shape = p.shape
            param_key = f"param_{i}_{param_shape}"
            
            # 如果这个参数之前没有见过，初始化它的动量
            if param_key not in self.m:
                self.m[param_key] = np.zeros_like(p)
                self.v[param_key] = np.zeros_like(p)
            
            # 更新一阶动量
            self.m[param_key] = self.beta1 * self.m[param_key] + (1 - self.beta1) * g
            # 更新二阶动量
            self.v[param_key] = self.beta2 * self.v[param_key] + (1 - self.beta2) * (g**2)
            
            # 偏差修正
            m_hat = self.m[param_key] / (1 - self.beta1**self.t)
            v_hat = self.v[param_key] / (1 - self.beta2**self.t)
            
            # 更新参数
            updated_params.append(p - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon))
            
        return updated_params

class RMSprop(Optimizer):
    """RMSprop优化器"""
    
    def __init__(self, lr=0.01, decay_rate=0.99, epsilon=1e-8):
        self.lr = lr
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = {}
        
    def update(self, params: List[np.ndarray], grads: List[np.ndarray]) -> List[np.ndarray]:
        """使用RMSprop算法更新参数"""
        updated_params = []
        
        for i, (p, g) in enumerate(zip(params, grads)):
            # 使用参数形状和索引作为键
            param_shape = p.shape
            param_key = f"param_{i}_{param_shape}"
            
            # 如果这个参数之前没有见过，初始化它的缓存
            if param_key not in self.cache:
                self.cache[param_key] = np.zeros_like(p)
                
            # 更新缓存
            self.cache[param_key] = self.decay_rate * self.cache[param_key] + (1 - self.decay_rate) * g**2
            # 更新参数
            updated_params.append(p - self.lr * g / (np.sqrt(self.cache[param_key]) + self.epsilon))
            
        return updated_params
