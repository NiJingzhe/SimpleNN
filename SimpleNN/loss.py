from typing import Optional
import numpy as np

class Loss:
    """损失函数基类"""
    
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算损失值"""
        raise NotImplementedError
        
    def backward(self) -> np.ndarray:
        """计算损失对预测值的梯度"""
        raise NotImplementedError

class MSE(Loss):
    """均方误差损失"""
    
    def __init__(self):
        self.y_pred: Optional[np.ndarray] = None
        self.y_true: Optional[np.ndarray] = None
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算MSE损失"""
        self.y_pred = y_pred
        self.y_true = y_true
        mse_value = np.mean(np.square(y_pred - y_true))
        return float(mse_value)
        
    def backward(self) -> np.ndarray:
        """计算MSE损失的梯度"""
        if self.y_pred is None or self.y_true is None:
            raise ValueError("y_pred 和 y_true 不能为None")
        return 2 * (self.y_pred - self.y_true) / self.y_pred.shape[0]

class SoftmaxCrossEntropy(Loss):
    """Softmax交叉熵损失，结合了Softmax激活和交叉熵损失"""
    
    def __init__(self):
        self.y_true: Optional[np.ndarray] = None
        self.probs: Optional[np.ndarray] = None
        
    def forward(self, x: np.ndarray, y_true: np.ndarray) -> float:
        """计算Softmax交叉熵损失
        
        Args:
            x: 模型输出的logits，形状为(batch_size, num_classes)
            y_true: 真实标签，形状为(batch_size,)，包含类别索引
            
        Returns:
            损失值
        """
        # 保存真实标签
        self.y_true = y_true
        
        # 计算Softmax概率
        shifted_logits = x - np.max(x, axis=1, keepdims=True)
        exp_logits = np.exp(shifted_logits)
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # 计算交叉熵损失
        batch_size = x.shape[0]
        
        if self.probs is None or self.y_true is None:
            raise ValueError("probs 和 y_true 不能为None")
        
        log_likelihoods = -np.log(self.probs[np.arange(batch_size), self.y_true] + 1e-8)
        return float(np.mean(log_likelihoods))
        
    def backward(self) -> np.ndarray:
        """计算Softmax交叉熵损失的梯度"""
        if self.probs is None or self.y_true is None:
            raise ValueError("probs 和 y_true 不能为None")
        
        batch_size = self.probs.shape[0]
        grad = self.probs.copy()
        grad[np.arange(batch_size), self.y_true] -= 1
        return grad / batch_size

class CrossEntropy(Loss):
    """交叉熵损失（不包含Softmax）"""
    
    def __init__(self):
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算交叉熵损失
        
        Args:
            y_pred: 预测概率，形状为(batch_size, num_classes)
            y_true: 真实标签，可以是one-hot编码或类别索引
            
        Returns:
            损失值
        """
        self.y_pred = y_pred
        
        # 处理y_true为类别索引的情况
        if y_true.ndim == 1:
            batch_size = y_true.shape[0]
            self.y_true = np.zeros_like(y_pred)
            self.y_true[np.arange(batch_size), y_true] = 1
        else:
            self.y_true = y_true
            
        # 数值稳定性处理
        y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
        
        # 计算交叉熵
        loss_value = -np.mean(np.sum(self.y_true * np.log(y_pred_clipped), axis=1))
        return float(loss_value)
        
    def backward(self) -> np.ndarray:
        """计算交叉熵损失的梯度"""
        if self.y_pred is None or self.y_true is None:
            raise ValueError("y_pred 和 y_true 不能为None")
        y_pred_clipped = np.clip(self.y_pred, 1e-8, 1 - 1e-8)
        return -self.y_true / y_pred_clipped / self.y_pred.shape[0]

class BinaryCrossEntropy(Loss):
    """二元交叉熵损失"""
    
    def __init__(self):
        self.y_true: Optional[np.ndarray] = None
        self.y_pred: Optional[np.ndarray] = None
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算二元交叉熵损失"""
        self.y_pred = y_pred
        self.y_true = y_true
        
        # 数值稳定性处理
        y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
        
        # 计算二元交叉熵
        loss_value = -np.mean(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        return float(loss_value)
        
    def backward(self) -> np.ndarray:
        """计算二元交叉熵损失的梯度"""
        if self.y_pred is None or self.y_true is None:
            raise ValueError("y_pred 和 y_true 不能为None")
        y_pred_clipped = np.clip(self.y_pred, 1e-8, 1 - 1e-8)
        return -(self.y_true / y_pred_clipped - (1 - self.y_true) / (1 - y_pred_clipped)) / self.y_pred.shape[0]

class L1Loss(Loss):
    """L1损失（平均绝对误差）"""
    
    def __init__(self):
        self.y_pred: Optional[np.ndarray] = None
        self.y_true: Optional[np.ndarray] = None
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算L1损失"""
        self.y_pred = y_pred
        self.y_true = y_true
        loss_value = np.mean(np.abs(y_pred - y_true))
        return float(loss_value)
        
    def backward(self) -> np.ndarray:
        """计算L1损失的梯度"""
        if self.y_pred is None or self.y_true is None:
            raise ValueError("y_pred 和 y_true 不能为None")
        return np.sign(self.y_pred - self.y_true) / self.y_pred.shape[0]

class HuberLoss(Loss):
    """Huber损失（结合了MSE和L1的优点）"""
    
    def __init__(self, delta=1.0):
        self.delta = delta
        self.y_pred: Optional[np.ndarray] = None
        self.y_true: Optional[np.ndarray] = None
        
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算Huber损失"""
        self.y_pred = y_pred
        self.y_true = y_true
        
        error = np.abs(y_pred - y_true)
        quadratic = np.minimum(error, self.delta)
        linear = error - quadratic
        
        loss_value = np.mean(0.5 * np.square(quadratic) + self.delta * linear)
        return float(loss_value)
        
    def backward(self) -> np.ndarray:
        """计算Huber损失的梯度"""
        if self.y_pred is None or self.y_true is None:
            raise ValueError("y_pred 和 y_true 不能为None")
        error = self.y_pred - self.y_true
        grad = np.zeros_like(error)
        
        # 对于|error| <= delta的情况，梯度为error
        mask_quadratic = np.abs(error) <= self.delta
        grad[mask_quadratic] = error[mask_quadratic]
        
        # 对于|error| > delta的情况，梯度为delta * sign(error)
        mask_linear = ~mask_quadratic
        grad[mask_linear] = self.delta * np.sign(error[mask_linear])
        
        return grad / self.y_pred.shape[0]
