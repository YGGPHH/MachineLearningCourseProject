import numpy as np
import os
import matplotlib.pyplot as plt

class ModelTrainer(object):
    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_id, device, max_epoch):
        model.train()

        loss_sigma = []

        for i, data in enumerate(data_loader):
            # 对dataloader中的数据进行遍历，一次遍历取走的是
            # 一个batch的数据。故dataloader的size为[total_size / batch_size]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # 输入到模型当中的是一个batch的数据(batch_size=64)

            outputs = model(inputs)

            optimizer.zero_grad()
            loss = loss_f(outputs, labels)
            loss.backward()
            optimizer.step()

            # 统计loss
            loss_sigma.append(loss.item())

            # 每10个iteration 打印一次训练信息，loss为10个iteration的平均
            if i % 50 == 50 - 1:
                print("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f}".format(
                    epoch_id + 1, max_epoch, i + 1, len(data_loader), np.mean(loss_sigma)))

        return np.mean(loss_sigma)

    @staticmethod
    def valid(data_loader, model, loss_f, device):
        model.eval()

        loss_sigma = []

        for i, data in enumerate(data_loader):

            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_f(outputs, labels)

            # 统计loss
            loss_sigma.append(loss.item())

        return np.mean(loss_sigma)

def plot_line(train_x, train_y, valid_x, valid_y, mode, out_dir):
    """
    绘制训练和验证集的loss曲线/acc曲线
    :param train_x: epoch
    :param train_y: 标量值
    :param valid_x:
    :param valid_y:
    :param mode:  'loss' or 'acc'
    :param out_dir:
    :return:
    """
    plt.plot(train_x, train_y, label='Train')
    plt.plot(valid_x, valid_y, label='Valid')

    plt.ylabel(str(mode))
    plt.xlabel('Epoch')

    location = 'upper right' if mode == 'loss' else 'upper left'
    plt.legend(loc=location)

    plt.title('_'.join([mode]))
    plt.savefig(os.path.join(out_dir, mode + '.png'))
    plt.close()