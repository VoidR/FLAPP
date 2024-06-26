import os
import csv
import torch
import logging
import requests
from torchvision import transforms
from torchvision import datasets as torch_datasets
from torch.utils.data import DataLoader, TensorDataset,random_split
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

import argparse
# import sqlite3

from sklearn import datasets as sk_datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

# from federated_learning.models import *
from federated_learning.models.SimpleModel import SimpleModel
from federated_learning.models.ResNet import resnet20
from federated_learning.models.MLP import MLP
from federated_learning.models.LogisticRegression import LogisticRegressionModel
from federated_learning.models.LeNet import LeNet
from federated_learning.models.AlexNet import AlexNet
from federated_learning.models.SimpleCNN import SimpleCNN

def get_device():
    """
    获取可用的设备
    input: 无
    output: torch.device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def get_model(training_config):
    """
    根据配置返回模型实例。
    input: 无
    output: 模型
    """
    model = None
    dim_in = None
    num_classes = None
    if training_config.get("dataset") == "MNIST":
        img_size = 28
        num_channels = 1
        num_classes = 10
        dim_in = img_size*img_size
    elif training_config.get("dataset") == "CIFAR10":
        num_channels = 3
        num_classes = 10
        img_size = 32
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
        model = LogisticRegressionModel(dim_in=dim_in, dim_out=num_classes)
    elif training_config.get("model") == "LeNet":
        model = LeNet(dim_in=num_channels,num_classes=num_classes,img_size=img_size)
    elif training_config.get("model") == "AlexNet":
        model = AlexNet(dim_in=num_channels,num_classes=num_classes,img_size=img_size)
    elif training_config.get("model") == "CNN":
        model = SimpleCNN(dim_in=num_channels,num_classes=num_classes,img_size=img_size)
    return model


def load_data(training_config, train=True, client_count=1, client_index=0):
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
        dataset = torch_datasets.MNIST(root='federated_learning/client/data', train=train, download=True, transform=transform)
    elif training_config.get("dataset") == "CIFAR10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(cifar10_means, cifar10_stds)
        ])
        dataset = torch_datasets.CIFAR10(root='federated_learning/client/data', train=train, download=True, transform=transform)
    elif training_config.get("dataset") in ["Iris", "Wine", "Breast_cancer"]:
        if training_config.get("dataset") == "Iris":
            data = sk_datasets.load_iris()
        elif training_config.get("dataset") == "Wine":
            data = sk_datasets.load_wine()
        elif training_config.get("dataset") == "Breast_cancer":
            data = sk_datasets.load_breast_cancer()

        X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)  
        y_test = torch.tensor(y_test, dtype=torch.long)

        if train:
            dataset = TensorDataset(X_train, y_train)
        else:
            dataset = TensorDataset(X_test, y_test)
    # elif training_config.get("dataset") == "Iris":
    #     # 加载鸢尾花数据集
    #     iris = sk_datasets.load_iris()
    #     # 数据和标签转换为Tensor
        
    # elif training_config.get("dataset") == "Wine":
    #     wine_dataset = sk_datasets.load_wine()
    #     data_tensor = torch.tensor(wine_dataset.data, dtype=torch.float32)
    #     target_tensor = torch.tensor(wine_dataset.target, dtype=torch.long)
    #     dataset = TensorDataset(data_tensor, target_tensor)
    # elif training_config.get("dataset") == "Breast_cancer":
    #     breast_cancer_dataset = sk_datasets.load_breast_cancer()
    #     data_tensor = torch.tensor(breast_cancer_dataset.data, dtype=torch.float32)
    #     target_tensor = torch.tensor(breast_cancer_dataset.target, dtype=torch.long)
    #     dataset = TensorDataset(data_tensor, target_tensor)
    else:
        raise ValueError("Unsupported dataset. Please choose either 'MNIST' or 'CIFAR10'.")

    # 计算每个客户端的数据量
    data_per_client = len(dataset) // client_count
    lengths = [data_per_client] * client_count
    # 如果数据不能被client_count整除，将剩余的数据分配给最后一个客户端
    lengths[-1] += len(dataset) % client_count
    # 使用random_split进行数据划分
    datasets = torch.utils.data.random_split(dataset, lengths)
    # print(f"Client {client_index} has {len(datasets)} samples.")
    loader = DataLoader(datasets[client_index-1], batch_size=training_config.get("batch_size", 64), shuffle=train)
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
    if training_config.get("optimizer", "Adam") == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    else:  # 默认使用SGD
        return optim.SGD(model.parameters(), lr=lr)

def get_loss_function(training_config):
    """
    根据配置返回损失函数。
    """
    if training_config.get("loss", "CrossEntropy") == "CrossEntropy":
        return nn.CrossEntropyLoss()
    # 添加其他损失函数的处理逻辑
    return nn.CrossEntropyLoss()  # 默认返回交叉熵损失函数

def test_model(model, training_config, loss_function, current_round,save_file='results.csv'):
    """
    在测试数据集上测试给定的模型状态字典, 根据training_config.get("metrics")中的选项保存测试结果。
    input: model_state_dict (dict): 模型状态字典。
    """
    # 此处的测试逻辑可能需要根据training_config中的metrics进行调整，以支持不同的评估指标
    device = get_device()
    test_loader = load_test_data(training_config)
    model = model.to(device)
    model.eval()
    y_true = []
    y_pred = []
    test_loss = 0  # 初始化测试损失
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = loss_function(outputs, target)  
            test_loss += loss.item()
            y_true.extend(target.tolist())
            y_pred.extend(predicted.tolist())
    
    test_loss /= len(test_loader)  # 计算平均损失

    metrics = training_config.get("metrics", [])
    results = {}

    if "Loss" in metrics:
        results = {'Loss': test_loss}  # 将损失添加到结果字典中

    if "Accuracy" in metrics:
        # correct = sum([1 for true, pred in zip(y_true, y_pred) if true == pred])
        # total = len(y_true)
        # accuracy = correct / total
        accuracy = accuracy_score(y_true, y_pred)
        results['Accuracy'] = accuracy

    if "Precision" in metrics:
        precision = precision_score(y_true, y_pred, average='weighted')
        results['Precision'] = precision

    if "Recall" in metrics:
        recall = recall_score(y_true, y_pred, average='weighted')
        results['Recall'] = recall

    if "F1" in metrics:
        f1 = f1_score(y_true, y_pred, average='weighted')
        results['F1'] = f1

    file_exists = os.path.isfile(save_file)
    # 将结果写入CSV文件
    with open(save_file, 'a+', newline='') as csvfile:
        fieldnames = ['Round'] + metrics
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        results['Round'] = current_round
        writer.writerow(results)


def tensor_to_list(state_dict):
    """
    将模型中的参数转换为列表。方便以JSON格式传输。
    """
    return {k: v.tolist() for k, v in state_dict.items()}
    
def tensors_to_lists(obj):
    """
    递归地将给定对象中的所有Tensor对象转换为列表。
    支持的对象类型包括字典、列表和Tensor。
    """
    if isinstance(obj, dict):
        # 如果对象是字典，对每个键值对应用相同的处理
        return {k: tensors_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, tuple):
        # 如果对象是元组，对每个元素应用相同的处理
        return tuple(tensors_to_lists(e) for e in obj)
    elif isinstance(obj, list):
        # 如果对象是列表，对每个元素应用相同的处理
        return [tensors_to_lists(e) for e in obj]
    elif isinstance(obj, torch.Tensor):
        # 如果对象是Tensor，转换为列表
        return obj.tolist()
    else:
        # 如果对象既不是字典、列表也不是Tensor，直接返回
        return obj


def criterion(y_pred, y_cls):
    c = torch.nn.CrossEntropyLoss()
    # y_cls = torch.squeeze(y_cls, dim=1)
    # print('y_cls',y_cls.shape)
    # print('y_pred:',y_pred.shape,'y_cls:',y_cls.shape,'argmax:',torch.argmax(y_cls, dim = -1).shape)
    # return c(y_pred, torch.argmax(y_cls, dim = -1))
    return c(y_pred, y_cls)