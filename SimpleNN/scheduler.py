import numpy as np
from typing import Dict, Any, Optional
from SimpleNN import Optimizer
# 实现学习率调度器

class Scheduler:
    
    def update_optimizer(self, optimizer: Optimizer, train_history: Dict[str, Any]) -> None:
        raise NotImplementedError


class LinearDecayScheduler(Scheduler):
    
    """ 
    线性递减学习率调度器
    """
    
    def __init__(self, final_lr: float, total_steps: int):
        """
        Args:
            final_lr: 最终学习率
            total_steps: 总步数
        """
        self.final_lr = final_lr
        self.total_steps = total_steps
        self.step = 0
        self.initial_lr: Optional[float] = None
        
    def update_optimizer(self, optimizer: Optimizer, train_history: Dict[str, Any]) -> None:
        """更新优化器的学习率
        
        Args:
            optimizer: 优化器
            train_history: 训练历史
        """
        # 第一次调用时保存初始学习率
        if self.initial_lr is None:
            self.initial_lr = optimizer.lr
            
        if self.step >= self.total_steps:
            print("学习率调度器已达到最大步数，停止更新学习率。")
            return 
            
        # 计算当前学习率
        alpha = float(self.step / self.total_steps)
        new_lr = self.initial_lr * (1 - alpha) + self.final_lr * alpha
        optimizer.lr = new_lr
        self.step += 1




