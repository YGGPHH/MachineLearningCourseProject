from sklearn.model_selection import train_test_split
# from tools.utils import makedir
import numpy as np
import os

# hyper-parameter
base_length = 96        # set to be 96
predict_length = 96     # 96 or 336
window_slide = base_length + predict_length

load_path = os.path.join("..", "data", "saved_data_{}.npz".format(str(predict_length)))
data = np.load(load_path, allow_pickle=True)
X, y = data['x'], data['y']

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=int(0.2*X.shape[0]), shuffle=False)
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,test_size=int(0.2*X.shape[0]), shuffle=False)
# makedir(os.path.join("..", "data", "split_data"))
print(train_X.shape)
print(val_X.shape)
print(test_X.shape)
saved_path = os.path.join("..", "data", "split_data", "data_{}".format(str(predict_length)))
np.savez(saved_path, train_X=train_X, train_y=train_y, test_X=test_X, test_y=test_y, val_X=val_X, val_y=val_y)