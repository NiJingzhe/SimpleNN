import numpy as np
import pandas as pd
import SimpleNN as snn
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_weather_data(sequence_length=24):
    """加载天气数据，并准备用于时间序列预测
    
    Args:
        sequence_length: 用于预测的前几个小时数量
        
    Returns:
        X_train: 训练数据
        y_train: 训练标签
        X_val: 验证数据
        y_val: 验证标签
    """
    # 读取数据
    df = pd.read_csv('data/weather_data.csv')
    
    # 提取需要的特征
    features = ['Temp_C', 'Dew Point Temp_C', 'Rel Hum_%', 'Wind Speed_km/h', 'Visibility_km', 'Press_kPa']
    
    # 将日期列转换为datetime类型
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])
    
    # 按时间排序
    df = df.sort_values('Date/Time')
    
    # 准备数据集
    X, y = [], []
    
    # 构建序列
    for i in range(len(df) - sequence_length):
        # 获取前sequence_length个时间点的所有特征
        X.append(df[features].values[i:i + sequence_length])
        # 预测下一个时间点的温度
        y.append(df['Temp_C'].values[i + sequence_length])
    
    X = np.array(X)
    y = np.array(y)
    
    # 重塑X为2D数组，以适应SimpleNN
    X = X.reshape(X.shape[0], -1)  # 将所有特征展平
    
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    # 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, y_train, X_val, y_val, scaler_y, df['Temp_C'].values

def main():
    # 加载数据
    seq_length = 24  # 使用前24小时的数据预测下一小时的温度
    X_train, y_train, X_val, y_val, scaler_y, original_temps = load_weather_data(seq_length)
    
    # 构建MLP模型
    input_size = X_train.shape[1]  # 输入特征数量
    model = snn.Model(
        layers=[
            snn.Dense(input_size, 64),
            snn.ReLU(),
            snn.Dropout(0.2),
            snn.Dense(64, 32),
            snn.ReLU(),
            snn.Dropout(0.2),
            snn.Dense(32, 1)
        ]
    )
    
    # 编译模型
    model.compile(
        loss=snn.MSE(),
        optimizer=snn.Adam(lr=0.001),
        scheduler=snn.LinearDecayScheduler(final_lr=0.0005, total_steps=500),
    )
    
    # 打印模型结构
    model.summary()
    
    # 训练模型
    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=128,
        epochs=500,
        validation_data=(X_val, y_val),
        shuffle=True,
        verbose=1,
        callbacks=[
            lambda _, history: ( True if len(history['val_loss']) > 150 and history['val_loss'][-1] - min(history['val_loss']) > 0.05 else False)
        ]
    )
    
    # 评估模型
    eval_results = model.evaluate(X_val, y_val)
    print(f"验证损失: {eval_results:.4f}")
    
    # 预测
    predictions = model.predict(X_val)
    
    # 反标准化预测结果
    predictions = np.array(scaler_y.inverse_transform(predictions.reshape(-1, 1))).flatten()
    y_val_original = np.array(scaler_y.inverse_transform(y_val.reshape(-1, 1))).flatten()
    
    # 计算RMSE和MAE
    rmse = np.sqrt(np.mean((predictions - y_val_original) ** 2))
    mae = np.mean(np.abs(predictions - y_val_original))
    print(f"RMSE: {rmse:.2f}°C")
    print(f"MAE: {mae:.2f}°C")
    
    # 绘制预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(y_val_original[-20:], label='Actual Temperature')
    plt.plot(predictions[-20:], label='Predicted Temperature')
    plt.title('Temperature Prediction')
    plt.xlabel('Time Point')
    plt.ylabel('Temperature (°C)')
    plt.legend()
    plt.savefig('weather_prediction.png')
    plt.close()
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    
    # 绘制损失
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Function')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制学习率
    plt.subplot(1, 2, 2)
    plt.plot(history['lr'])
    plt.title('Learning Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    
    plt.tight_layout()
    plt.savefig('weather_training_history.png')
    plt.close()

if __name__ == '__main__':
    main()