import os
import numpy as np


def load_data(load_path):
    load_path = load_path
    data = np.load(load_path, allow_pickle=True)
    return data['train_X'], data['train_y'], data['test_X'], data['test_y'], data['val_X'], data['val_y']