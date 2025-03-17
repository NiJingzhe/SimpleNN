import numpy as np
from typing import List, Dict, Any, Optional
from .parameter import Parameter

class Optimizer:
    """优化器基类"""
    
    def __init__(self, lr: float):
        """初始化优化器
        
        Args:
            lr: 学习率
        """
        self.lr = lr
        self.param_states: Dict[int, Dict[str, Any]] = {}  # 存储参数状态
    
    def update_parameter(self, param: Parameter) -> None:
        """更新单个参数
        
        Args:
            param: 参数对象
        """
        raise NotImplementedError
    
    def update(self, parameters: Dict[str, Parameter]) -> None:
        """更新所有参数
        
        Args:
            parameters: 参数字典
        """
        for name, param in parameters.items():
            if param.grad is not None:
                self.update_parameter(param)
    
    def zero_grad(self, parameters: Dict[str, Parameter]) -> None:
        """清空所有参数的梯度
        
        Args:
            parameters: 参数字典
        """
        for param in parameters.values():
            param.zero_grad()

class SGD(Optimizer):
    """随机梯度下降优化器"""
    
    def __init__(self, lr=0.01):
        """初始化SGD优化器
        
        Args:
            lr: 学习率
        """
        super().__init__(lr=lr)
        
    def update_parameter(self, param: Parameter) -> None:
        """使用梯度更新参数
        
        Args:
            param: 参数对象
        """
        if param.grad is None:
            return
        
        # 梯度下降更新
        param.data = param.data - self.lr * param.grad

class MomentumSGD(Optimizer):
    """带动量的SGD优化器"""
    
    def __init__(self, lr=0.1, momentum=0.9):
        """初始化带动量的SGD优化器
        
        Args:
            lr: 学习率
            momentum: 动量因子
        """
        super().__init__(lr=lr)
        self.momentum = momentum
        
    def update_parameter(self, param: Parameter) -> None:
        """使用动量更新参数
        
        Args:
            param: 参数对象
        """
        if param.grad is None:
            return
            
        param_id = param.id
        
        # 如果参数没有状态，则初始化
        if param_id not in self.param_states:
            self.param_states[param_id] = {'velocity': np.zeros_like(param.data)}
            
        # 获取参数状态
        state = self.param_states[param_id]
        
        # 更新速度和参数
        state['velocity'] = self.momentum * state['velocity'] - self.lr * param.grad
        param.data = param.data + state['velocity']

class Adam(Optimizer):
    """Adam优化器"""
    
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        """初始化Adam优化器
        
        Args:
            lr: 学习率
            beta1: 一阶动量系数
            beta2: 二阶动量系数
            epsilon: 数值稳定性参数
        """
        super().__init__(lr=lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0  # 时间步
        
    def update(self, parameters: Dict[str, Parameter]) -> None:
        """更新所有参数
        
        Args:
            parameters: 参数字典
        """
        self.t += 1  # 更新时间步
        super().update(parameters)
    
    def update_parameter(self, param: Parameter) -> None:
        """使用Adam算法更新参数
        
        Args:
            param: 参数对象
        """
        if param.grad is None:
            return
            
        param_id = param.id
        
        # 如果参数没有状态，则初始化
        if param_id not in self.param_states:
            self.param_states[param_id] = {
                'm': np.zeros_like(param.data),  # 一阶动量
                'v': np.zeros_like(param.data)   # 二阶动量
            }
            
        # 获取参数状态
        state = self.param_states[param_id]
        
        # 更新一阶动量
        state['m'] = self.beta1 * state['m'] + (1 - self.beta1) * param.grad
        # 更新二阶动量
        state['v'] = self.beta2 * state['v'] + (1 - self.beta2) * (param.grad**2)
        
        # 偏差修正
        m_hat = state['m'] / (1 - self.beta1**self.t)
        v_hat = state['v'] / (1 - self.beta2**self.t)
        
        # 更新参数
        param.data = param.data - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

class RMSprop(Optimizer):
    """RMSprop优化器"""
    
    def __init__(self, lr=0.01, decay_rate=0.99, epsilon=1e-8):
        """初始化RMSprop优化器
        
        Args:
            lr: 学习率
            decay_rate: 衰减率
            epsilon: 数值稳定性参数
        """
        super().__init__(lr=lr)
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        
    def update_parameter(self, param: Parameter) -> None:
        """使用RMSprop算法更新参数
        
        Args:
            param: 参数对象
        """
        if param.grad is None:
            return
            
        param_id = param.id
        
        # 如果参数没有状态，则初始化
        if param_id not in self.param_states:
            self.param_states[param_id] = {'cache': np.zeros_like(param.data)}
            
        # 获取参数状态
        state = self.param_states[param_id]
        
        # 更新缓存
        state['cache'] = self.decay_rate * state['cache'] + (1 - self.decay_rate) * param.grad**2
        # 更新参数
        param.data = param.data - self.lr * param.grad / (np.sqrt(state['cache']) + self.epsilon)
