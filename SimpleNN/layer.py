import numpy as np
from typing import Optional, Tuple, Dict
from .optimizer import Optimizer
from .parameter import Parameter


class Layer:
    """神经网络层的基类，定义了所有层必须实现的接口"""

    def __init__(self):
        """初始化层"""
        self._parameters: Dict[str, Parameter] = {}
        self._name: str = ""

    def register_parameter(self, name: str, param: Parameter) -> None:
        """注册参数

        Args:
            name: 参数名称
            param: 参数对象
        """
        self._parameters[name] = param

    def parameters(self) -> Dict[str, Parameter]:
        """获取层的所有参数

        Returns:
            参数字典
        """
        return self._parameters

    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播"""
        raise NotImplementedError

    def update(self, optimizer: Optional[Optimizer] = None) -> None:
        """参数更新"""
        pass  # 在新设计中，参数更新由优化器统一管理


class Dense(Layer):
    """全连接层

    :math:`y = x @ W + b`

    And the gradient should be represented as:

    :math:`dL/dW = dL/dy * dy/dW = x^T * dL/dy`

    :math:`dL/db = dL/dy * dy/db = dL/dy * 1 = sum(dL/dy, axis=0)`

    where :math:`dL/dy` is the gradient of the loss with respect to the output.

    """

    def __init__(self, input_dim: int, output_dim: int):
        """初始化全连接层

        Args:
            input_dim: 输入维度
            output_dim: 输出维度
        """
        super().__init__()
        self._name = f"Dense({input_dim}, {output_dim})"
        # He初始化
        W_data = np.random.randn(input_dim, output_dim) * np.sqrt(2 / input_dim)
        b_data = np.zeros((1, output_dim))

        # 创建参数并注册
        self.register_parameter("W", Parameter(W_data))
        self.register_parameter("b", Parameter(b_data))

        self.x: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播"""
        self.x = x
        W = self._parameters["W"].data
        b = self._parameters["b"].data
        return x @ W + b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播

        数学推导:
        前向传播: y = x @ W + b

        对W的梯度:
        dL/dW = dL/dy * dy/dW = x^T * dL/dy
        其中dL/dy就是输入的grad参数

        对b的梯度:
        dL/db = dL/dy * dy/db = dL/dy * 1 = sum(dL/dy, axis=0)

        对x的梯度:
        dL/dx = dL/dy * dy/dx = dL/dy * W^T
        """
        if self.x is None:
            raise ValueError("无法对没有输入的层进行反向传播")

        W = self._parameters["W"].data

        # 计算梯度
        dW = self.x.T @ grad
        db = np.sum(grad, axis=0, keepdims=True)

        # 设置梯度
        self._parameters["W"].grad = dW
        self._parameters["b"].grad = db

        return grad @ W.T


class BatchNorm(Layer):
    """BatchNorm层

    :math:`y = gamma * (x - mean(x)) / sqrt(var(x) + eps) + beta`

    where :math:`mean(x)` is the mean for each CHANNEL of the input.
    and :math:`var(x)` is the variance for each CHANNEL of the input.

    :math:`mean(x) = 1/n * sum(x, axis=0)`

    :math:`var(x) = 1/n * sum((x - mean(x))^2, axis=0)`

    There are also two LEARNABLE parameters: :math:`gamma` and :math:`beta`.

    And the gradient should be represented as:

    :math:`dL / d gamma = dL / dy * dy / d gamma = (x - mean(x)) / sqrt(var(x) + eps)`
    :math:`dL / d beta = dL / dy`

    :math:`dL / dx = dL / dy * dy / dx = dL / dy * gamma`
    """

    def __init__(self, feature_shape: Tuple[int, ...], eps: float = 1e-5, momentum: float = 0.9):
        """初始化BatchNorm层

        Args:
            feature_shape: 输入的特征形状
            eps: 防止除零的小常数
            momentum: 用于更新移动平均的动量参数
        """
        super().__init__()
        self._name = f"BatchNorm({feature_shape})"
        self.eps = eps
        self.momentum = momentum
        self.feature_shape = feature_shape

        # 初始化可学习参数
        gamma = np.ones(feature_shape)
        beta = np.zeros(feature_shape)
        self.register_parameter("gamma", Parameter(gamma))
        self.register_parameter("beta", Parameter(beta))

        # 初始化移动平均参数
        self.running_mean = np.zeros(feature_shape)
        self.running_var = np.zeros(feature_shape)

        # 用于反向传播的中间变量
        self.x: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.var: Optional[np.ndarray] = None
        self.x_centered: Optional[np.ndarray] = None
        self.std_inv: Optional[np.ndarray] = None
        self.normalized: Optional[np.ndarray] = None

    def forward(self, x: np.ndarray, training=True) -> np.ndarray:
        """前向传播

        Args:
            x: 输入数据
            training: 是否为训练模式

        Returns:
            归一化后的数据
        """
        self.x = x
        gamma = self._parameters["gamma"].data
        beta = self._parameters["beta"].data

        if training:
            # 计算当前批次的统计量
            self.mean = np.mean(x, axis=0)
            self.var = np.var(x, axis=0, ddof=1)  # 使用无偏估计

            # 更新移动平均
            assert self.mean is not None and self.var is not None
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var
        else:
            # 使用移动平均的统计量
            self.mean = self.running_mean
            self.var = self.running_var

        # 归一化
        assert self.mean is not None and self.var is not None
        self.x_centered = x - self.mean
        self.std_inv = 1.0 / np.sqrt(self.var + self.eps)
        self.normalized = self.x_centered * self.std_inv

        return gamma * self.normalized + beta

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """反向传播

        Args:
            grad: 损失对输出的梯度

        Returns:
            损失对输入的梯度
        """
        if self.x is None or self.mean is None or self.var is None:
            raise ValueError("无法对没有输入的层进行反向传播")

        x = self.x
        gamma = self._parameters["gamma"].data
        N = x.shape[0]

        # 计算对gamma和beta的梯度
        assert self.normalized is not None
        dgamma = np.sum(grad * self.normalized, axis=0)
        dbeta = np.sum(grad, axis=0)

        # 计算对x的梯度
        dx_normalized = grad * gamma
        assert self.x_centered is not None and self.std_inv is not None
        dvar = np.sum(dx_normalized * self.x_centered * -0.5 * self.std_inv**3, axis=0)
        dmean = np.sum(dx_normalized * -self.std_inv, axis=0) + dvar * np.mean(-2 * self.x_centered, axis=0)
        dx = dx_normalized * self.std_inv + dvar * 2 * self.x_centered / N + dmean / N

        # 设置参数梯度
        self._parameters["gamma"].grad = dgamma
        self._parameters["beta"].grad = dbeta

        return dx
