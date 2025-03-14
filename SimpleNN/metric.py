import numpy as np

class Metric:
    """指标基类，用于定义评估指标"""
    
    def __init__(self, name: str):
        """初始化指标
        
        Args:
            name: 指标名称
        """
        self.name = name
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算指标值
        
        Args:
            y_pred: 预测值，具有shape (batch_size, class_num)
            y_true: 真实值，具有shape (batch_size,)
            
        Returns:
            指标值
        """
        raise NotImplementedError("子类必须实现__call__方法")

class Accuracy(Metric):
    """准确率指标"""
    
    def __init__(self):
        """初始化准确率指标"""
        super().__init__("accuracy")
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算准确率
        
        Args:
            y_pred: 预测值，具有shape (batch_size, class_num)
            y_true: 真实值，具有shape (batch_size,)
            
        Returns:
            准确率
        """
        # y_pred 是 logits 或者 概率 或者 one hot，y_true 只能是 类别标签
        
        # 首先检查 y_pred.shape[0] == y_true.shape[0]
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("y_pred 和 y_true 的样本数量不一致")
        
        
        # 然后 y_pred 必然是 batch size, class_num的 shape
        # y_true 必然是 batch size的 shape
        
        
        if len(y_pred.shape) != 2:
            raise ValueError("y_pred 必须具有两个维度， batch size, class_num")
        
        if len(y_true.shape) != 1:
            raise ValueError("y_true 必须具有一个维度， batch size")
        
        # 检查是否空预测
        if y_pred.shape[0] == 0 or y_true.shape[0] == 0:
            return np.nan
        
        # 计算准确率
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = y_true
        return np.mean(y_pred_classes == y_true_classes)
            

class F1Score(Metric):
    """F1分数指标"""
    
    def __init__(self):
        """初始化F1分数指标"""
        super().__init__("f1_score")
    
    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """计算F1分数
        
        Args:
            y_pred: 预测值，具有shape (batch_size, class_num)
            y_true: 真实值，具有shape (batch_size,)
            
        Returns:
            F1分数
        """
        # y_pred 是 logits 或者 概率 或者 one hot，y_true 只能是 类别标签
        
        # 首先检查 y_pred.shape[0] == y_true.shape[0]
        if y_pred.shape[0] != y_true.shape[0]:
            raise ValueError("y_pred 和 y_true 的样本数量不一致")
        
        # 然后 y_pred 必然是 batch size, class_num的 shape
        # y_true 必然是 batch size的 shape
        if len(y_pred.shape) != 2:
            raise ValueError("y_pred 必须具有两个维度， batch size, class_num")
        
        if len(y_true.shape) != 1:
            raise ValueError("y_true 必须具有一个维度， batch size")
        
        # 检查 y_pred 和 y_true 是否为空
        if y_pred.shape[0] == 0 or y_true.shape[0] == 0:
            return np.nan
        
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # 计算TP, FP, FN
        true_positives = np.sum((y_pred_classes == 1) & (y_true == 1))
        false_positives = np.sum((y_pred_classes == 1) & (y_true == 0))
        false_negatives = np.sum((y_pred_classes == 0) & (y_true == 1))
        
        # 计算精确率和召回率
        precision = true_positives / (true_positives + false_positives + 1e-10) 
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        
        # 计算F1分数
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        
        return f1
