import numpy as np
import pandas as pd
import SimpleNN as snn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_tesla_data():
    """加载特斯拉股票数据
    
    Returns:
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
    """
    # 读取数据
    df = pd.read_csv('data/tesla.csv')
    
    # 准备特征
    features = ['Open', 'High', 'Low', 'Volume']
    X = df[features].values
    
    # 添加前一天的收盘价作为特征
    prev_close = df['Close'].shift(1).values.astype(np.float64)
    X = np.column_stack((X, prev_close))
    X = X[1:]  # 删除第一行（因为第一天没有前一天的收盘价）
    
    # 准备标签（收盘价）
    y = df['Close'].values[1:]
    
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_val, y_val, scaler_y

def main():
    # 加载数据
    X_train, y_train, X_val, y_val, scaler_y = load_tesla_data()
    
    # 构建MLP模型
    model = snn.Model(
        layers=[
            snn.Dense(5, 32),
            snn.ReLU(),
            snn.Dropout(0.2),
            snn.Dense(32, 16),
            snn.ReLU(),
            snn.Dropout(0.2),
            snn.Dense(16, 1)
        ]
    )
    
    # 编译模型
    model.compile(
        loss=snn.MSE(),
        optimizer=snn.Adam(lr=0.01),
        scheduler=snn.LinearDecayScheduler(final_lr=0.001, total_steps=300),
    )
    
    # 打印模型结构
    model.summary()
    
    # 训练模型
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=32,
        epochs=1000,
        validation_data=(X_val, y_val),
        shuffle=True,
        verbose=1
    )
    
    # 评估模型
    eval_results = model.evaluate(X_val, y_val)
    print(f"验证损失: {eval_results:.4f}")
    
    # 预测
    predictions = model.predict(X_val)
    
    # 反标准化预测结果
    predictions = np.array(scaler_y.inverse_transform(predictions.reshape(-1, 1))).flatten()
    y_val_original = np.array(scaler_y.inverse_transform(y_val.reshape(-1, 1))).flatten()
    
    # 计算RMSE
    rmse = np.sqrt(np.mean((predictions - y_val_original) ** 2))
    print(f"RMSE: ${rmse:.2f}")
    
    # 绘制预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_val_original, label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.title('Tesla Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.savefig('tesla_prediction.png')
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
    
    
    plt.tight_layout()
    plt.savefig('tesla_training_history.png')
    plt.close()

if __name__ == '__main__':
    main()
