import os
import torch
import matplotlib.pyplot as plt

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def plot_predict(groundtruth, predict, timestamp_actual, timestamp_predict, attribute, out_dir):
    plt.plot(timestamp_predict, predict, label='predict')
    plt.plot(timestamp_actual, groundtruth, label='groundtruth')

    plt.ylabel(str(attribute))
    plt.xlabel('timestamp')

    plt.legend(loc='lower left')

    plt.title('_'.join([attribute]))
    plt.savefig(os.path.join(out_dir, attribute + '.png'))
    plt.close()