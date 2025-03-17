import numpy as np
from typing import List, Dict, Union, Callable, Tuple, Optional, Any, cast, TypeVar

from .scheduler import Scheduler
from .layer import Layer, Dense, BatchNorm
from .loss import Loss
from .optimizer import Optimizer
from .metric import Metric, Accuracy
import time

# 定义类型变量以便后续使用
LayerWithWeights = TypeVar("LayerWithWeights", bound=Layer)
LayerWithBatchNorm = TypeVar("LayerWithBatchNorm", bound=Layer)


class Model:
    """神经网络模型"""

    def __init__(self, layers: List[Layer] | None = None):
        """初始化模型

        Args:
            layers: 网络层列表，可选，也可以通过add方法添加
        """
        self.layers = layers if layers is not None else []
        self.loss_fn: Optional[Loss] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[Scheduler] = None
        self.history: Dict[str, List[float]] = {}  # 空字典，将在fit方法中初始化
        self.metrics: List[Metric] = []  # 用于存储要计算的指标

    def add(self, layer: Layer):
        """添加一个层到模型

        Args:
            layer: 要添加的层
        """
        if self.layers is None:
            self.layers = []
        self.layers.append(layer)

    def _forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """内部前向传播方法

        Args:
            x: 输入数据
            training: 是否处于训练模式

        Returns:
            模型输出
        """
        for layer in self.layers:
            # 处理有training参数的层（如BatchNorm、Dropout）
            if (
                hasattr(layer, "forward")
                and "training" in layer.forward.__code__.co_varnames
            ):
                x = layer.forward(x, training=training)
            else:
                x = layer.forward(x)
        return x

    def _backward(self, grad: np.ndarray):
        """内部反向传播方法

        Args:
            grad: 损失对输出的梯度
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def _update(self):
        """内部参数更新方法"""
        # 收集所有层的参数
        all_parameters = {}
        for layer in self.layers:
            if hasattr(layer, 'parameters'):
                layer_params = layer.parameters()
                # 更新全局参数字典，避免名称冲突
                for name, param in layer_params.items():
                    param_key = f"{layer.__class__.__name__}_{id(layer)}_{name}"
                    all_parameters[param_key] = param
        
        # 更新学习率调度器
        if self.scheduler is not None and self.optimizer is not None:
            self.scheduler.update_optimizer(self.optimizer, self.history)
            
        # 使用优化器更新所有参数
        if self.optimizer is not None:
            self.optimizer.update(all_parameters)
            self.optimizer.zero_grad(all_parameters)
            
    def _train_step(self, x: np.ndarray, y: np.ndarray) -> float:
        """内部训练步骤方法
        
        Args:
            x: 输入数据
            y: 目标值
            
        Returns:
            损失值
        """
        try:
            # 前向传播
            outputs = self._forward(x, training=True)
            
            # 确保y的维度与outputs匹配
            # 对于二分类问题，如果outputs是(batch_size, 1)，但y是(batch_size,)，需要调整维度
            if len(outputs.shape) > 1 and outputs.shape[1] == 1 and len(y.shape) == 1:
                y = y.reshape(-1, 1)
            
            # 确保loss_fn不为None
            if self.loss_fn is None:
                raise ValueError("模型尚未编译，请先调用compile方法")
                
            # 计算损失
            loss = self.loss_fn.forward(outputs, y)
            
            # 反向传播
            grad = self.loss_fn.backward()
            self._backward(grad)
            
            # 更新参数
            self._update()
            
            return loss
        except Exception as e:
            print(f"Error in _train_step: {e}")
            # 打印输入和输出的形状，帮助调试
            print(f"Input shape: {x.shape}, Output shape: {outputs.shape if 'outputs' in locals() else 'N/A'}, Target shape: {y.shape}")
            raise

    def _compute_metrics(
        self, y_pred: np.ndarray, y_true: np.ndarray
    ) -> Dict[str, float]:
        """计算评估指标

        Args:
            y_pred: 预测值
            y_true: 真实值

        Returns:
            包含各指标值的字典
        """
        metrics_values = {}

        # 计算每个指标的值
        for metric in self.metrics:
            metrics_values[metric.name] = metric(y_pred, y_true)

        return metrics_values

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> Union[float, Dict[str, float]]:
        """评估模型

        Args:
            x: 输入数据
            y: 目标值

        Returns:
            如果没有指定metrics，则返回损失值；否则返回包含损失和其他指标的字典
        """
        if self.loss_fn is None:
            raise ValueError("模型尚未编译，请先调用compile方法")

        # 前向传播（测试模式）
        outputs = self._forward(x, training=False)

        # 确保y的维度与outputs匹配
        if len(outputs.shape) > 1 and outputs.shape[1] == 1 and len(y.shape) == 1:
            y = y.reshape(-1, 1)

        # 计算损失
        loss = self.loss_fn.forward(outputs, y)

        # 如果没有指定metrics，直接返回损失值
        if not self.metrics:
            return loss

        # 计算其他指标
        metrics_values = self._compute_metrics(outputs, y)
        metrics_values["loss"] = loss

        return metrics_values

    def predict(self, x: np.ndarray) -> np.ndarray:
        """使用模型进行预测

        Args:
            x: 输入数据

        Returns:
            预测结果
        """
        return self._forward(x, training=False)

    def fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        epochs: int = 10,
        validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        shuffle: bool = True,
        verbose: int = 1,
        callbacks: Optional[List[Callable[[int, Dict[str, List[float]]], None]]] = None,
    ) -> Dict[str, List[float]]:
        """训练模型

        Args:
            x: 训练数据
            y: 训练标签
            batch_size: 批量大小
            epochs: 训练轮数
            validation_data: 验证数据，格式为(x_val, y_val)
            shuffle: 是否在每个epoch前打乱数据
            verbose: 日志显示级别，0为不显示，1为显示进度条，2为每个batch显示
            callbacks: 回调函数列表

        Returns:
            训练历史记录，包含loss、val_loss和其他指标
        """
        if self.loss_fn is None or self.optimizer is None:
            raise ValueError("模型尚未编译，请先调用compile方法")

        num_samples = x.shape[0]
        iterations_per_epoch = int(np.ceil(num_samples / batch_size))

        # 初始化历史记录
        self.history = {"loss": []}

        # 如果有验证数据，添加val_loss
        if validation_data is not None:
            self.history["val_loss"] = []

        # 添加其他指标的历史记录
        if self.metrics:
            for metric in self.metrics:
                self.history[metric.name] = []
                if validation_data is not None:
                    self.history[f"val_{metric.name}"] = []

        # 初始化回调函数
        if callbacks is None:
            callbacks = []

        # 开始训练
        for epoch in range(epochs):
            start_time = time.time()
            epoch_loss = 0.0
            epoch_metrics: Dict[str, float] = {
                metric.name: 0.0 for metric in self.metrics
            }

            # 打乱数据
            if shuffle:
                indices = np.random.permutation(num_samples)
                x_shuffled = x[indices]
                y_shuffled = y[indices]
            else:
                x_shuffled = x
                y_shuffled = y

            # 批量训练
            for i in range(iterations_per_epoch):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, num_samples)

                batch_x = x_shuffled[start_idx:end_idx]
                batch_y = y_shuffled[start_idx:end_idx]

                try:
                    # 执行一步训练
                    batch_loss = self._train_step(batch_x, batch_y)
                    epoch_loss += batch_loss * (end_idx - start_idx)
                    
                    # 使用调度器更新学习率
                    if self.scheduler is not None and self.optimizer is not None:
                        self.scheduler.update_optimizer(self.optimizer, self.history)

                    # 计算其他指标
                    if self.metrics:
                        batch_pred = self._forward(batch_x, training=False)
                        # 确保batch_y的维度与batch_pred匹配
                        if (
                            len(batch_pred.shape) > 1
                            and batch_pred.shape[1] == 1
                            and len(batch_y.shape) == 1
                        ):
                            batch_y_reshaped = batch_y.reshape(-1, 1)
                        else:
                            batch_y_reshaped = batch_y
                        batch_metrics = self._compute_metrics(
                            batch_pred, batch_y_reshaped
                        )
                        for metric_name, value in batch_metrics.items():
                            epoch_metrics[metric_name] += value * (end_idx - start_idx)

                    # 显示进度
                    if verbose == 2:
                        print(
                            f"Epoch {epoch+1}/{epochs} - Batch {i+1}/{iterations_per_epoch} - Loss: {batch_loss:.4f}"
                        )
                    elif (
                        verbose == 1
                        and (i + 1) % max(1, iterations_per_epoch // 10) == 0
                    ):
                        progress = (i + 1) / iterations_per_epoch
                        bar_length = 30
                        bar = (
                            "=" * int(bar_length * progress)
                            + ">"
                            + " " * (bar_length - int(bar_length * progress) - 1)
                        )
                        print(
                            f"\rEpoch {epoch+1}/{epochs} - [{bar}] {progress*100:.1f}% - Loss: {batch_loss:.4f}",
                            end="",
                        )
                except Exception as e:
                    print(
                        f"\nError during training at epoch {epoch+1}, batch {i+1}: {e}"
                    )
                    print(f"Batch shapes - X: {batch_x.shape}, y: {batch_y.shape}")
                    raise

            # 计算平均损失和指标
            epoch_loss /= num_samples
            self.history["loss"].append(epoch_loss)

            for metric in self.metrics:
                epoch_metrics[metric.name] /= num_samples
                self.history[metric.name].append(epoch_metrics[metric.name])

            # 验证
            val_metrics: Dict[str, float] = {}
            val_loss = 0.0
            if validation_data is not None:
                x_val, y_val = validation_data

                if self.metrics:
                    # 如果有指定metrics，计算所有指标
                    val_results = self.evaluate(x_val, y_val)
                    if isinstance(val_results, dict):
                        val_loss = val_results["loss"]
                        for metric_name, value in val_results.items():
                            if metric_name != "loss":
                                val_metrics[metric_name] = value
                                self.history[f"val_{metric_name}"].append(value)
                    else:
                        val_loss = val_results
                else:
                    # 否则只计算损失
                    val_loss = cast(float, self.evaluate(x_val, y_val))

                self.history["val_loss"].append(val_loss)

            # 显示epoch结果
            if verbose > 0:
                epoch_time = time.time() - start_time
                metrics_str = ""
                for metric_name, value in epoch_metrics.items():
                    metrics_str += f" - {metric_name}: {value:.4f}"

                val_str = (
                    f" - val_loss: {val_loss:.4f}"
                    if validation_data is not None
                    else ""
                )
                for metric_name, value in val_metrics.items():
                    val_str += f" - val_{metric_name}: {value:.4f}"

                print(
                    f"\rEpoch {epoch+1}/{epochs} - {epoch_time:.2f}s - loss: {epoch_loss:.4f}{metrics_str}{val_str}"
                )

            # 执行回调函数
            for callback in callbacks:
                callback(epoch, self.history)

        return self.history

    def compile(
        self,
        loss: Loss,
        optimizer: Optimizer,
        scheduler: Optional[Scheduler] = None,
        metrics: Optional[Union[List[str], List[Metric]]] = None,
    ):
        """编译模型，设置损失函数、优化器和评估指标

        Args:
            loss: 损失函数
            optimizer: 优化器
            scheduler: 学习率调度器
            metrics: 评估指标列表，可以是字符串列表（如['accuracy']）或Metric对象列表
        """
        self.loss_fn = loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = []

        # 处理指标
        if metrics:
            for metric in metrics:
                if isinstance(metric, Metric):
                    # 如果已经是Metric对象，直接添加
                    self.metrics.append(metric)
                else:
                    raise ValueError(
                        f"指标必须是字符串或Metric对象，而不是{type(metric)}"
                    )

    def summary(self):
        """打印模型结构摘要"""
        print("Model Summary:")
        print("=" * 80)
        print(f"{'Layer (type)':<30}{'Output Shape':<25}{'Param #':<15}")
        print("=" * 80)
        
        total_params = 0
        trainable_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_name = f"{i}: {layer.__class__.__name__}"
            
            # 计算参数数量
            params = 0
            # 检查是否有权重和偏置 (Dense层)
            if isinstance(layer, Dense):
                dense_layer = cast(Dense, layer)
                W = dense_layer.parameters()['W'].data
                b = dense_layer.parameters()['b'].data
                w_params = np.prod(W.shape)
                b_params = np.prod(b.shape)
                params = w_params + b_params
                trainable_params += params
                
                # 输出形状
                shape_str = f"({W.shape[0]}, {W.shape[1]})"
            # 检查是否有BatchNorm参数
            elif isinstance(layer, BatchNorm):
                bn_layer = cast(BatchNorm, layer)
                gamma = bn_layer.parameters()['gamma'].data
                beta = bn_layer.parameters()['beta'].data
                params = np.prod(gamma.shape) + np.prod(beta.shape)
                trainable_params += params
                
                # 输出形状
                shape_str = f"({gamma.shape[1]})"
            else:
                shape_str = "Unknown"
                
            total_params += params
            
            print(f"{layer_name:<30}{shape_str:<25}{params:<15}")
            
        print("=" * 80)
        print(f"Total params: {total_params}")
        print(f"Trainable params: {trainable_params}")
        print(f"Non-trainable params: {total_params - trainable_params}")
        print("=" * 80)
