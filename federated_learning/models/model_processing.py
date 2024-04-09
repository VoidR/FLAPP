import os
import csv
import torch
import logging
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F

import argparse
# import sqlite3

from sklearn import datasets as skdatasets
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

# from federated_learning.models import *
from federated_learning.models.SimpleModel import SimpleModel
from federated_learning.models.ResNet import resnet20
from federated_learning.models.MLP import MLP
from federated_learning.models.LogisticRegression import LogisticRegressionModel
from federated_learning.models.LeNet import LeNet
from federated_learning.models.AlexNet import AlexNet


def get_model(training_config):
    """
    根据配置返回模型实例。
    input: 无
    output: 模型
    """
    dim_in = None
    num_classes = None
    if training_config.get("dataset") == "MNIST":
        dim_in = 28*28
        num_channels = 1
        num_classes = 10
    elif training_config.get("dataset") == "CIFAR10":
        num_channels = 3
        num_classes = 10
    elif training_config.get("dataset") == "Iris":
        dim_in = 4
        num_classes = 3
    elif training_config.get("dataset") == "Wine":
        dim_in = 13
        num_classes = 3
    elif training_config.get("dataset") == "Breast_cancer":
        dim_in = 30
        num_classes = 2

    if training_config.get("model") == "NN":
        model = SimpleModel(dim_in, num_classes)
    elif training_config.get("model") == "ResNet20":
        model = resnet20(num_classes=10, num_channels=3)
    elif training_config.get("model") == "LR":
        model = LogisticRegressionModel(dim_in, num_classes)
    elif training_config.get("model") == "LeNet":
        model = LeNet(dim_in=num_channels,dim_out=num_classes)
    elif training_config.get("model") == "AlexNet":
        model = AlexNet(num_classes=num_classes)

    return model


def load_data(training_config, train=True):
    """
    加载数据集进行训练或测试。
    input: train (bool): 如果为True，则加载训练集，否则加载测试集。
    output: DataLoader实例，用于训练或测试数据的迭代。
    """

    # CIFAR10的全局平均值和标准差
    cifar10_means = (0.4914, 0.4822, 0.4465)
    cifar10_stds = (0.2470, 0.2435, 0.2616)

    # MNIST的全局平均值和标准差
    mnist_mean = (0.1307,)
    mnist_std = (0.3081,)

    if training_config.get("dataset") == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mnist_mean, mnist_std)
        ])
        dataset = datasets.MNIST(root='federated_learning/client/data', train=train, download=True, transform=transform)
    elif training_config.get("dataset") == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_means, cifar10_stds)
        ])
        dataset = datasets.CIFAR10(root='federated_learning/client/data', train=train, download=True, transform=transform)
    elif training_config.get("dataset") == "Iris":
        # 加载鸢尾花数据集
        iris = skdatasets.load_iris()
        # 数据和标签转换为Tensor
        data_tensor = torch.tensor(iris.data, dtype=torch.float32)
        target_tensor = torch.tensor(iris.target, dtype=torch.long)
        # 创建TensorDataset
        dataset = TensorDataset(data_tensor, target_tensor)
    elif training_config.get("dataset") == "Wine":
        wine_dataset = skdatasets.load_wine()
        data_tensor = torch.tensor(wine_dataset.data, dtype=torch.float32)
        target_tensor = torch.tensor(wine_dataset.target, dtype=torch.long)
        dataset = TensorDataset(data_tensor, target_tensor)
    elif training_config.get("dataset") == "Breast_cancer":
        breast_cancer_dataset = skdatasets.load_breast_cancer()
        data_tensor = torch.tensor(breast_cancer_dataset.data, dtype=torch.float32)
        target_tensor = torch.tensor(breast_cancer_dataset.target, dtype=torch.long)
        dataset = TensorDataset(data_tensor, target_tensor)
    else:
        raise ValueError("Unsupported dataset. Please choose either 'MNIST' or 'CIFAR10'.")

    loader = DataLoader(dataset, batch_size=training_config.get("batch_size", 64), shuffle=train)
    return loader


def load_test_data(training_config):
    """
    加载测试数据集。
    input: 无
    output: DataLoader实例，用于测试数据的迭代。
    """
    return load_data(training_config, train=False)


def get_optimizer(model,training_config):
    """
    根据配置选择并返回优化器。
    """
    lr = training_config.get("learning_rate", 0.01)
    if training_config.get("optimizer", "SGD") == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    else:  # 默认使用SGD
        return optim.SGD(model.parameters(), lr=lr)

def get_loss_function(training_config):
    """
    根据配置返回损失函数。
    """
    if training_config.get("loss", "CrossEntropy") == "CrossEntropy":
        return F.cross_entropy
    # 添加其他损失函数的处理逻辑
    return F.cross_entropy  # 默认返回交叉熵损失函数


def tensor_to_list(state_dict):
    """
    将模型中的参数转换为列表。方便以JSON格式传输。
    """
    return {k: v.tolist() for k, v in state_dict.items()}
    