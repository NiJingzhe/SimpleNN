import numpy as np
from typing import Optional, Dict, Any

class Parameter:
    """参数类，用于包装网络参数及其梯度"""
    
    def __init__(self, data: np.ndarray):
        """初始化参数
        
        Args:
            data: 参数数据
        """
        self.data = data
        self.grad: Optional[np.ndarray] = None
        self.id = id(self)  # 唯一标识符
    
    def zero_grad(self) -> None:
        """清空梯度"""
        self.grad = None
    
    @property
    def shape(self) -> tuple:
        """获取参数形状"""
        return self.data.shape 