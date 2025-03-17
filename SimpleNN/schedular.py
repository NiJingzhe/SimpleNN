import numpy as np
from typing import Dict, Any
from SimpleNN import Optimizer
# 实现学习率调度器

class Schedular:
    
    def update_optimizer(self, optimizer: Optimizer, train_history: Dict[str, Any]) -> None:
        raise NotImplementedError


class ExponentDecaySchedular(Schedular):
    
    """ 
    指数衰减学习率调度器
    """
    
    def __init__(self, gamma: np.float32):
        
        self.gamma = gamma
        self.step = 0
        
    def update_optimizer(self, optimizer: Optimizer, train_history: Dict[str, Any]) -> None:
        
        current_lr = optimizer.lr
        
        # calc new lr
        new_lr = current_lr * np.exp(self.step)
        
        # set new lr for optimizer
        optimizer.lr = new_lr
        
        # update step
        self.step += 1
        