import numpy as np
from pyparsing import C
import SimpleNN as snn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# 加载鸢尾花数据集
def load_iris_dataset():
    """加载鸢尾花数据集
    
    Returns:
        X_train: 训练数据
        y_train: 训练标签（类别索引）
        X_val: 验证数据
        y_val: 验证标签（类别索引）
        class_names: 类别名称
    """
    # 加载数据集
    iris = load_iris()
    X = iris.data  # type: ignore
    y = iris.target  # type: ignore
    class_names = iris.target_names  # type: ignore
    
    # 数据标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, y_train, X_val, y_val, class_names


# 加载鸢尾花数据集
X_train, y_train, X_val, y_val, class_names = load_iris_dataset()

# 可视化数据集
plt.figure(figsize=(12, 5))

# 绘制特征分布
plt.subplot(1, 2, 1)
for i in range(3):
    plt.scatter(X_train[y_train == i, 0], X_train[y_train == i, 1], 
                label=class_names[i], alpha=0.7)
plt.title('Iris Dataset - First Two Features')
plt.xlabel('Sepal Length (standardized)')
plt.ylabel('Sepal Width (standardized)')
plt.legend()

# 绘制特征分布（另一对特征）
plt.subplot(1, 2, 2)
for i in range(3):
    plt.scatter(X_train[y_train == i, 2], X_train[y_train == i, 3], 
                label=class_names[i], alpha=0.7)
plt.title('Iris Dataset - Last Two Features')
plt.xlabel('Petal Length (standardized)')
plt.ylabel('Petal Width (standardized)')
plt.legend()

plt.tight_layout()
plt.savefig('iris_dataset.png')
plt.close()

# 构建MLP模型
model = snn.Model(
    layers=[
        snn.Dense(4, 16),
        snn.ReLU(),
        snn.Dense(16, 3),
        snn.Tanh()
    ]
)
# 编译模型
model.compile(
    loss=snn.SoftmaxCrossEntropy(),  # 多分类使用Softmax交叉熵损失
    optimizer=snn.Adam(lr=0.01),
    scheduler=snn.LinearDecayScheduler(final_lr=0.001, total_steps=300),
    metrics=[snn.Accuracy()],  # 使用准确率指标
)

# 打印模型结构
model.summary()

# 训练模型
history = model.fit(
    x=X_train,
    y=y_train,
    batch_size=64,                                                                                                  
    epochs=100,  # 增加训练轮数
    validation_data=(X_val, y_val),
    shuffle=True,
    verbose=1
)

# 评估模型
eval_results = model.evaluate(X_val, y_val)
print(f"验证损失: {eval_results['loss']:.4f}")        # type: ignore
print(f"验证准确率: {eval_results['accuracy']:.4f}")  # type: ignore

# 预测
predictions = model.predict(X_val)
predictions_class = np.argmax(predictions, axis=1)

# 计算准确率
accuracy = np.mean(predictions_class == y_val)
print(f"验证准确率: {accuracy:.4f}")

# 打印分类报告
print("\n分类报告:")
print(classification_report(y_val, predictions_class, target_names=class_names))

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, predictions_class)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# 绘制训练历史
plt.figure(figsize=(12, 5))

# 绘制损失
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# 绘制准确率
plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('iris_training_history.png')
plt.close()



