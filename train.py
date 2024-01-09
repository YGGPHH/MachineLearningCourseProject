from datetime import datetime
import argparse

import torch.nn
from torch.utils.data import TensorDataset, DataLoader

from model.pureLSTM import pureLSTM_336, pureLSTM_96
from model.TimeSeriesTransformer import Transformer_96, Transformer_336
from model.TimeSeriesCNN import TimeSeriesCNN_96, TimeSeriesCNN_336
from tools.data_load import load_data
from tools.model_trainer import *
from tools.utils import *

# hyper-parameter
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=256, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate, when training Transformer, prefer 0.0005")
parser.add_argument("--epochs", type=int, default=50, help="number of epochs of training")
parser.add_argument("--predict_length", type=int, default=96, help="96 or 336")
parser.add_argument("--model", type=str, default='pureLSTM', help='pureLSTM/Transformer')
parser.add_argument("--loss", type=str, default='MSE', help='MSE/MAE')
args = parser.parse_args()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

if args.model == 'pureLSTM':
    model_from_args = pureLSTM_96 if args.predict_length == 96 else pureLSTM_336
elif args.model == 'Transformer':
    model_from_args = Transformer_96 if args.predict_length == 96 else Transformer_336
else:
    model_from_args = TimeSeriesCNN_96 if args.predict_length == 96 else TimeSeriesCNN_336

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
now_time = datetime.now()
time_str = datetime.strftime(now_time, '%m-%d_%H-%M')
log_dir = os.path.join(BASE_DIR, "results", 'train', '{}_{}_{}'.format(args.model, str(args.predict_length), args.loss), time_str)
makedir(log_dir)

# load data and make it to tensor
load_path = os.path.join('data', 'split_data', 'data_{}.npz'.format(args.predict_length))
train_X, train_y, test_X, test_y, val_X, val_y = load_data(load_path)
train_X, train_y = train_X.astype(float), train_y.astype(float)
train_X, train_y = torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32)
train_Dataset = TensorDataset(train_X, train_y)

val_X, val_y = val_X.astype(float), val_y.astype(float)
val_X, val_y = torch.tensor(val_X, dtype=torch.float32), torch.tensor(val_y, dtype=torch.float32)
val_Dataset = TensorDataset(val_X, val_y)

# 注意DataLoader当中的shuffle, 在进行validation的时候应该重置DataLoader并将shuffle置为False
train_loader = DataLoader(dataset=train_Dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
valid_loader = DataLoader(dataset=val_Dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)
# model = pureLSTM_96()
model = model_from_args()
model.to(device=device)
# model = TimeSeriesTransformer()
optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
criterion = torch.nn.MSELoss() if args.loss == 'MSE' else torch.nn.L1Loss() # 使用MSE（均方误差）/MAE作为时间序列预测的评估指标

loss_rec = {"train": [], "valid": []}
final_epoch, final_loss_train, final_loss_valid = 0, 0, 0
tmp_model_state_dict, tmp_optimizer_state_dict, best_checkpoint = None, None, None
for epoch in range(args.epochs):
    loss_train = ModelTrainer.train(train_loader, model, criterion, optimizer, epoch, device,
                                                          args.epochs)
    loss_valid = ModelTrainer.valid(valid_loader, model, criterion, device)
    print("Epoch[{:0>3}/{:0>3}] Train loss:{:.4f} Valid loss:{:.4f} LR:{}".format(
        epoch + 1, args.epochs, loss_train, loss_valid, optimizer.param_groups[0]["lr"]))

    loss_rec["train"].append(loss_train), loss_rec["valid"].append(loss_valid)
    plt_x = np.arange(1, epoch + 2)
    plot_line(plt_x, loss_rec["train"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)

    if epoch > args.epochs / 2 and final_loss_valid * 10 < loss_valid and tmp_model_state_dict != None:
        best_checkpoint = {"model_state_dict": tmp_model_state_dict,
                           "optimizer_state_dict": tmp_optimizer_state_dict,
                           "epoch": final_epoch,
                           "loss_train": final_loss_train,
                           "loss_valid": final_loss_valid}
        break

    final_epoch, final_loss_train, final_loss_valid = epoch, loss_train, loss_valid
    tmp_model_state_dict, tmp_optimizer_state_dict = model.state_dict(), optimizer.state_dict()

checkpoint = {"model_state_dict": model.state_dict(),
              "optimizer_state_dict": optimizer.state_dict(),
              "epoch": final_epoch,
              "loss_train": final_loss_train,
              "loss_valid": final_loss_valid}
checkpoint_dir = os.path.join(BASE_DIR, "results", "checkpoint", "{}_{}_{}".format(args.model, str(args.predict_length), args.loss))
makedir(checkpoint_dir)
path_checkpoint = os.path.join(checkpoint_dir, "model_checkpoint.pkl")
path_best_checkpoint = os.path.join(checkpoint_dir, "best_model_checkpoint.pkl")
torch.save(checkpoint, path_checkpoint)
if best_checkpoint != None:
    torch.save(best_checkpoint, path_best_checkpoint)

train_log = os.path.join(BASE_DIR, "results", 'train_log.txt')
train_result = "model:{}, predict_len:{}, loss_f:{}, {}, train_loss: {:.4f}; valid_loss: {:.4f}\n".format(args.model,
                                                                                                          str(args.predict_length),
                                                                                                          args.loss, time_str,
                                                                                                          final_loss_train,
                                                                                                          final_loss_valid)
with open(train_log, 'a+') as f:
    f.write(train_result)