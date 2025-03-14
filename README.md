# SimpleNN

SimpleNN是一个基于NumPy实现的轻量级神经网络框架，采用静态计算图设计，纯Toylike项目。

## 特性

- 基于NumPy的纯Python实现
- 静态计算图设计
- 模块化架构
- 支持常用的层类型和激活函数
- 提供多种优化器和损失函数
- 包含完整的训练和评估流程

## 安装

在当前文件夹下执行:
```bash
poetry install 
```
或
```bash
pip install .
```


## 快速开始

```python
import SimpleNN as snn

# 构建模型
model = snn.Model()
model.add(snn.Dense(784, 256))  # 输入维度784，输出维度256
model.add(snn.ReLU())
model.add(snn.Dense(256, 10))   # 输出维度10
model.add(snn.Softmax())

# 编译模型
model.compile(
    loss=snn.SoftmaxCrossEntropy(),
    optimizer=snn.Adam(lr=0.01),
    metrics=[snn.Accuracy()]
)

# 训练模型
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=32,
    epochs=10,
    validation_data=(X_val, y_val)
)
```

## 架构设计

SimpleNN采用模块化设计，主要包含以下核心组件：

### 1. 层（Layer）

层是网络的基本构建块，所有层都继承自基类`Layer`，必须实现以下接口：
- `forward(x)`: 前向传播
- `backward(grad)`: 反向传播
- `update(optimizer)`: 参数更新

支持的层类型：
- `Dense`: 全连接层
- `BatchNorm`: 批量归一化层

### 2. 激活函数（Activation）

激活函数模块提供多种非线性变换：
- `ReLU`: 整流线性单元
- `Sigmoid`: Sigmoid函数
- `Tanh`: 双曲正切函数
- `Softmax`: Softmax函数
- `Dropout`: Dropout正则化

### 3. 损失函数（Loss）

损失函数用于计算预测值与真实值之间的差异：
- `MSE`: 均方误差
- `SoftmaxCrossEntropy`: Softmax交叉熵
- `CrossEntropy`: 交叉熵
- `BinaryCrossEntropy`: 二元交叉熵
- `L1Loss`: L1损失
- `HuberLoss`: Huber损失

### 4. 优化器（Optimizer）

优化器负责更新网络参数：
- `SGD`: 随机梯度下降
- `MomentumSGD`: 带动量的SGD
- `Adam`: Adam优化器
- `RMSprop`: RMSprop优化器

### 5. 评估指标（Metric）

用于评估模型性能的指标：
- `Accuracy`: 准确率
- `F1Score`: F1分数

### 6. 模型（Model）

Model类是整个框架的核心，负责：
- 层的管理
- 前向传播
- 反向传播
- 参数更新
- 训练和评估流程

## 使用示例

### 二分类问题

```python
import SimpleNN as snn

# 构建模型
model = snn.Model()
model.add(snn.Dense(2, 16))
model.add(snn.ReLU())
model.add(snn.Dense(16, 1))
model.add(snn.Sigmoid())

# 编译模型
model.compile(
    loss=snn.BinaryCrossEntropy(),
    optimizer=snn.Adam(lr=0.01),
    metrics=[snn.Accuracy()]
)

# 训练模型
history = model.fit(X_train, y_train, epochs=100)
```

### 多分类问题

```python
import SimpleNN as snn

# 构建模型
model = snn.Model()
model.add(snn.Dense(784, 256))
model.add(snn.ReLU())
model.add(snn.Dense(256, 10))
model.add(snn.Softmax())

# 编译模型
model.compile(
    loss=snn.SoftmaxCrossEntropy(),
    optimizer=snn.Adam(lr=0.01),
    metrics=[snn.Accuracy()]
)

# 训练模型
history = model.fit(X_train, y_train, epochs=100)
```
