import pandas as pd
import numpy as np
import os

def sliding_window(data, predict_length, base_length=96):
    # 获取带有时间戳的数据
    x_data, timestamp = [], []
    for i in range(base_length):
        x_data.append(data[i])
    x_data = np.array(x_data)

    y_data = []
    for i in range(base_length,base_length + predict_length):
        y_data.append(data[i])
    y_data = np.array(y_data)
    return x_data, y_data

# 读入数据
data_path = os.path.join('..', 'data', 'ETTh1.csv')
data = pd.read_csv(data_path).values[:,1:] # 从.csv文件读入数据

split_train_test = int(0.6*data.shape[0])
split_test_valid = int(0.8*data.shape[0])

# hyper-parameter
base_length = 96        # set to be 96
predict_length = 96     # 96 or 336
window_slide = base_length + predict_length # 滑动窗口*2 = 数据窗口 + 预测窗口

# 制作滑动窗口数据
processed_data, processed_label = [], []

for i in range(0,len(data)-(window_slide)):
    x, y = sliding_window(data[i:window_slide+i], predict_length)
    processed_data.append(x)
    processed_label.append(y)

# 将数据保存为(17228, 96, 8)的张量，将标签保存为(17228, 96)的张量
processed_data = np.array(processed_data)
processed_label = np.array(processed_label)

print(processed_data.shape)
print(processed_label.shape)

# 保存数据
# saved_path = os.path.join("..", "data", "saved_data_{}".format(str(predict_length)))
# np.savez(saved_path, x=processed_data, y=processed_label)