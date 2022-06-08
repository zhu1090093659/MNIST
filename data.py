# load MNIST data
import os

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def load_data(path):
    # 加载训练数据
    data = np.fromfile(open(os.path.join(path, "train-images.idx3-ubyte")), dtype=np.uint8)
    train_img = data[16:].reshape(-1, 28, 28, 1).astype(np.float32)
    # normalize to [0, 1]
    train_img /= 255.0
    # 加载训练标签
    data = np.fromfile(open(os.path.join(path, "train-labels.idx1-ubyte")), dtype=np.uint8)
    train_label = data[8:].reshape(-1).astype(np.int64)
    # 加载测试数据
    data = np.fromfile(open(os.path.join(path, "t10k-images.idx3-ubyte")), dtype=np.uint8)
    test_img = data[16:].reshape(-1, 28, 28, 1).astype(np.float32)
    # normalize to [0, 1]
    test_img /= 255.0
    # 加载测试标签
    data = np.fromfile(open(os.path.join(path, "t10k-labels.idx1-ubyte")), dtype=np.uint8)
    test_label = data[8:].reshape(-1).astype(np.int64)
    return train_img, train_label, test_img, test_label


# 创建pytorch数据集
def create_dataset(train_img, train_label, test_img, test_label, batch_size, shuffle=True):
    # 将numpy数组转为pytorch tensor
    train_img = torch.from_numpy(train_img)
    train_label = torch.from_numpy(train_label)
    test_img = torch.from_numpy(test_img)
    test_label = torch.from_numpy(test_label)
    # 从numpy数组中创建pytorch数据集
    train_dataset = TensorDataset(train_img, train_label)
    # 创建验证数据据
    tran_size = int(0.95 * len(train_dataset))
    val_size = len(train_dataset) - tran_size
    train_dataset, val_dataset = random_split(train_dataset, [tran_size, val_size])
    # 创建测试数据集
    test_dataset = TensorDataset(test_img, test_label)
    # 创建训练数据集加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=shuffle)  # num_workers越大越慢？目前实验是这样 训练设置为0最快4最慢
    # 创建验证数据集加载器
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    # 创建测试数据集加载器
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)  # test的num_workers设置为4，训练时设置为0
    return train_loader, val_loader, test_loader
