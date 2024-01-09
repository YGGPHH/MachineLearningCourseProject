from datetime import datetime
import argparse

import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from model.pureLSTM import pureLSTM_96, pureLSTM_336
from model.TimeSeriesTransformer import Transformer_96, Transformer_336
from model.TimeSeriesCNN import TimeSeriesCNN_96, TimeSeriesCNN_336
from tools.data_load import load_data
from tools.utils import *

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# hyper-parameter
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--predict_length", type=int, default=96, help="96 or 336")
parser.add_argument("--model", type=str, default='pureLSTM', help='pureLSTM/Transformer')
parser.add_argument("--loss", type=str, default='MSE', help='MSE/MAE')
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

load_path = os.path.join('data', 'split_data', 'data_{}.npz'.format(args.predict_length))
_, _, test_X, test_y, _, _ = load_data(load_path)
test_X, test_y = test_X.astype(float), test_y.astype(float)
test_X, test_y = torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)
test_Dataset = TensorDataset(test_X, test_y)
test_loader = DataLoader(dataset=test_Dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)

if args.model == 'pureLSTM':
    model_from_args = pureLSTM_96 if args.predict_length == 96 else pureLSTM_336
elif args.model == 'Transformer':
    model_from_args = Transformer_96 if args.predict_length == 96 else Transformer_336
else:
    model_from_args = TimeSeriesCNN_96 if args.predict_length == 96 else TimeSeriesCNN_336
    
model = model_from_args()
# args.loss = 'MAE' # using MAE to train and MSE to test
checkpoint = os.path.join(BASE_DIR, "results", "checkpoint", "{}_{}_{}".format(args.model, str(args.predict_length), args.loss), "model_checkpoint.pkl")
reload_state = torch.load(checkpoint)
state_dict = reload_state['model_state_dict']
model.load_state_dict(state_dict)
# args.loss = 'MSE'
criterion = torch.nn.MSELoss() if args.loss == 'MSE' else torch.nn.L1Loss() # 使用MSE（均方误差）/MAE作为时间序列预测的评估指标

total_loss = []
model.eval()
model.to(device)
with torch.no_grad():
    for i, data in enumerate(test_loader):
        loss_sigma = []
        # 对dataloader中的数据进行遍历，一次遍历取走的是
        # 一个batch的数据。故dataloader的size为[total_size / batch_size]
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        # print(inputs.shape)
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss_sigma.append(loss.item())
        total_loss.append(loss.item())

        print("Testing: Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
             i + 1, len(test_loader), np.mean(loss_sigma)))

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
test_log = os.path.join(BASE_DIR, "results", 'test_log.txt')
test_result = "model:{}, predict_len:{}, loss_f:{}, {}, test_loss: {:.4f}\n".format(args.model, str(args.predict_length), args.loss, time_str, np.mean(total_loss))
print(test_result)
with open(test_log, 'a+') as f:
    f.write(test_result)