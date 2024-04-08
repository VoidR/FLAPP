#FLAPP/federated_learning/client/client.py
import os
import csv
import torch
import logging
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from flask import Flask, request, jsonify

import socket
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

from federated_learning.client.utils.DP import dp_protection


# 解析命令行参数
parser = argparse.ArgumentParser(description='客户端启动配置')
parser.add_argument('-p', '--port', type=int, default=5001, help='客户端端口号，默认为5001')
parser.add_argument('-s', '--server', type=str, default="http://127.0.0.1:5000", help='服务器IP，默认为http://127.0.0.1:5000')
args = parser.parse_args()

app = Flask(__name__)

def get_local_ip():
    """
    获取本地 IP 地址
    input: 无
    output: 本地 IP 地址
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # doesn't even have to be reachable
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

# 配置变量
server_url = args.server
# server_url = "http://192.168.1.115:5000"
client_IP = get_local_ip()
client_port = args.port  # 从命令行参数获取或使用默认值
client_id = None
training_config = {}
# training_config = {
#   "model":"LR",
#   "dataset":"Iris",
#   "optimizer":"Adam",
#   "loss":"CrossEntropy",
#   "metrics":["Accuracy"],
#   "global_epochs":20,
#   "local_epochs":10,
#   "batch_size":64,
#   "learning_rate":0.001,
#   "client_use_differential_privacy": True,
#   "differential_privacy": {
#     "epsilon": 1.0,   
#     "delta": 1e-5,   
#   }
# }

def get_model():
    """
    根据配置返回模型实例。
    input: 无
    output: 模型
    """
    dim_in = None
    num_classes = None
    if training_config.get("dataset") == "MNIST":
        dim_in = 28
        num_classes = 10
    elif training_config.get("dataset") == "CIFAR10":
        num_channels = 3
        num_classes = 10
    elif training_config.get("dataset") == "Iris":
        dim_in = 4
        num_classes = 3
    if training_config.get("model") == "NN":
        model = SimpleModel(dim_in, num_classes)
    elif training_config.get("model") == "ResNet20":
        model = resnet20(num_classes=10, num_channels=3)
    elif training_config.get("model") == "LR":
        model = LogisticRegressionModel(dim_in, num_classes)
    elif training_config.get("model") == "LeNet":
        model = LeNet(dim_out=num_classes)
    elif training_config.get("model") == "AlexNet":
        model = AlexNet(num_classes=num_classes)

    return model




def register_client():
    """
    向服务器注册客户端，以参与联邦学习。
    input: 无
    output: 无，但函数会更新全局变量`client_id`和`training_config`
    """
    response = requests.post(f"{server_url}/api/register_client", json={"client_url": f"http://{client_IP}:{client_port}"})
    if response.status_code == 200:
        global client_id, training_config
        client_id = response.json()['client_id']
        training_config = response.json()['training_config']
        print(f"注册成功，ID: {client_id}, 训练配置: {training_config}")

def unregister_client():
    """
    从服务器注销客户端，清理与此客户端相关的服务器端状态。
    input: 无
    output: bool, 表示是否成功注销客户端。
    """
    global client_id
    if client_id is None:
        print("客户端未注册。")
        return False
    url = f"{server_url}/api/unregister_client/{client_id}"
    response = requests.delete(url)
    if response.status_code == 200:
        print("客户端注销成功。")
        client_id = None
        return True
    else:
        print("客户端注销失败。")
        return False

def load_data(train=True):
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
    else:
        raise ValueError("Unsupported dataset. Please choose either 'MNIST' or 'CIFAR10'.")

    loader = DataLoader(dataset, batch_size=training_config.get("batch_size", 64), shuffle=train)
    return loader


def load_test_data():
    """
    加载测试数据集。
    input: 无
    output: DataLoader实例，用于测试数据的迭代。
    """
    return load_data(train=False)


def get_optimizer(model):
    """
    根据配置选择并返回优化器。
    """
    lr = training_config.get("learning_rate", 0.01)
    if training_config.get("optimizer", "SGD") == "Adam":
        return optim.Adam(model.parameters(), lr=lr)
    else:  # 默认使用SGD
        return optim.SGD(model.parameters(), lr=lr)

def get_loss_function():
    """
    根据配置返回损失函数。
    """
    if training_config.get("loss", "CrossEntropy") == "CrossEntropy":
        return F.cross_entropy
    # 添加其他损失函数的处理逻辑
    return F.cross_entropy  # 默认返回交叉熵损失函数

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """
    API端点，用于接收全局模型状态，进行训练和测试。
    input: 通过POST请求的JSON体接收的全局模型状态。
    output: JSON响应，包含本地模型更新和准确率。
    """
    received_data = request.json

    local_model_update = train_model_one_round(received_data)
    
    local_model_update = apply_security_measures(local_model_update)

    return jsonify({"model_update": local_model_update})


def train_model_one_round(global_model_info):
    """
    使用服务器提供的全局模型状态字典训练模型一个周期。
    input: global_model_info(json): 服务器提供的json, 包括全局模型参数和当前全局轮数。
    output: dict, 训练一个周期后模型的状态字典。
    """
    global_model_state_dict = global_model_info['global_model']
    current_round = global_model_info["current_round"]

    print("开始训练，全局轮数:", current_round)
    model = get_model()  # 此处应根据training_config['model']选择不同的模型
    model.load_state_dict({k: torch.tensor(v) for k, v in global_model_state_dict.items()})

    # 检查是否有可用的CUDA
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 将模型移动到正确的设备
    model.to(device)

    data_loader = load_data(train=True)
    optimizer = get_optimizer(model)
    loss_function = get_loss_function()
    model.train()
    for _ in range(training_config.get("local_epochs", 1)):  # 进行多轮本地训练
        for data, target in data_loader:
            # data = data.view(data.shape[0], -1)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
    test_model(model,current_round)
    return {k: v.tolist() for k, v in model.state_dict().items()}

def test_model(model,current_round):
    """
    在测试数据集上测试给定的模型状态字典, 根据training_config.get("metrics")中的选项保存测试结果。
    input: model_state_dict (dict): 模型状态字典。
    """
    # 此处的测试逻辑可能需要根据training_config中的metrics进行调整，以支持不同的评估指标

    test_loader = load_test_data()
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            y_true.extend(target.tolist())
            y_pred.extend(predicted.tolist())

    metrics = training_config.get("metrics", [])
    results = {}

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

    file_exists = os.path.isfile('results.csv')
    # 将结果写入CSV文件
    with open('results.csv', 'a+', newline='') as csvfile:
        fieldnames = ['Round'] + metrics
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        results['Round'] = current_round
        writer.writerow(results)


def apply_security_measures(model_update):
    """
    应用安全措施到模型更新，当前使用差分隐私。
    可以在这里添加更多的安全措施。
    """
    if training_config['client_use_differential_privacy']:
        dp_params = training_config['differential_privacy']
        # 应用差分隐私机制，比如添加高斯噪声
        model_update = dp_protection(model_update, dp_params)
    # 将来可能添加其他安全措施
    return model_update

def mytest():
    print(get_local_ip())

if __name__ == "__main__":
    register_client()  # 注册客户端以获取ID
    try:
        app.run(host='0.0.0.0', port=client_port, debug=False)
    finally:
        unregister_client()  # 确保应用退出时注销客户端
    # mytest()




