from datetime import datetime
import argparse

import os
import torch
import numpy as np
from model.pureLSTM import pureLSTM_96, pureLSTM_336
from model.TimeSeriesTransformer import Transformer_96, Transformer_336
from model.TimeSeriesCNN import TimeSeriesCNN_96, TimeSeriesCNN_336
from tools.data_load import load_data
from tools.utils import makedir, plot_predict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
label_list = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']

# hyper-parameter
parser = argparse.ArgumentParser()
parser.add_argument("--predict_length", type=int, default=96, help="96 or 336")
parser.add_argument("--model", type=str, default='pureLSTM', help='pureLSTM/Transformer')
parser.add_argument("--loss", type=str, default='MSE', help='MSE/MAE')
args = parser.parse_args()

if args.model == 'pureLSTM':
    model_from_args = pureLSTM_96 if args.predict_length == 96 else pureLSTM_336
elif args.model == 'Transformer':
    model_from_args = Transformer_96 if args.predict_length == 96 else Transformer_336
else:
    model_from_args = TimeSeriesCNN_96 if args.predict_length == 96 else TimeSeriesCNN_336
    
model = model_from_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

checkpoint = os.path.join(BASE_DIR, "results", "checkpoint", "{}_{}_{}".format(args.model, str(args.predict_length), args.loss), "model_checkpoint.pkl")
reload_state = torch.load(checkpoint)
state_dict = reload_state['model_state_dict']
model.load_state_dict(state_dict)

load_path = os.path.join('data', 'split_data', 'data_{}.npz'.format(args.predict_length))
_, _, test_X, test_y, _, _ = load_data(load_path)
test_X, test_y = test_X.astype(float), test_y.astype(float)
test_X, test_y = torch.tensor(test_X, dtype=torch.float32), torch.tensor(test_y, dtype=torch.float32)

predict_X, predict_y = test_X[0].unsqueeze(0), test_y[0].detach().numpy()
before_y = predict_X.squeeze(0).detach().numpy()
predict_y = np.concatenate((before_y, predict_y), axis=0)

model.eval()
model.to(device)
predict_X = predict_X.to(device)
output = model(predict_X).cpu().squeeze(0).detach().numpy()
timestamp_predict = np.array(range(predict_X.shape[1]+1, predict_y.shape[0]+1))
timestamp_actual = np.array(range(1, predict_y.shape[0]+1))

now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
out_dir = os.path.join(BASE_DIR, 'results', 'predict', '{}_{}_{}'.format(args.model, str(args.predict_length), args.loss), time_str)
makedir(out_dir)

for i in range(output.shape[1]):
    plot_predict(predict_y[:,i], output[:,i], timestamp_actual, timestamp_predict, label_list[i], out_dir)