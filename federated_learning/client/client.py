#FLAPP/federated_learning/client/client.py
import torch
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from flask import Flask, request, jsonify

import argparse
# import sqlite3

from sklearn import datasets 


# from federated_learning.models import *
from federated_learning.models.SimpleModel import SimpleModel
from federated_learning.models.ResNet import resnet20
from federated_learning.models.MLP import MLP
from federated_learning.models.LogisticRegression import LogisticRegressionModel

from federated_learning.client.utils.DP import dp_protection


# 解析命令行参数
parser = argparse.ArgumentParser(description='客户端启动配置')
parser.add_argument('-p', '--port', type=int, default=5001, help='客户端端口号，默认为5001')
args = parser.parse_args()

app = Flask(__name__)

# 配置变量
server_url = "http://127.0.0.1:5000"
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
    return model

def register_client():
    """
    向服务器注册客户端，以参与联邦学习。
    input: 无
    output: 无，但函数会更新全局变量`client_id`和`training_config`
    """
    response = requests.post(f"{server_url}/api/register_client", json={"port": client_port})
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
        iris = datasets.load_iris()
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

def train_model_one_round(global_model_state_dict):
    """
    使用服务器提供的全局模型状态字典训练模型一个周期。
    input: global_model_state_dict (dict): 服务器下发的全局模型状态字典。
    output: dict, 训练一个周期后模型的状态字典。
    """
    print("开始训练")
    model = get_model()  # 此处应根据training_config['model']选择不同的模型
    model.load_state_dict({k: torch.tensor(v) for k, v in global_model_state_dict.items()})
    data_loader = load_data(train=True)
    optimizer = get_optimizer(model)
    loss_function = get_loss_function()
    model.train()
    for _ in range(training_config.get("local_epochs", 1)):  # 进行多轮本地训练
        for data, target in data_loader:
            # data = data.view(data.shape[0], -1)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_function(output, target)
            loss.backward()
            optimizer.step()
    return {k: v.tolist() for k, v in model.state_dict().items()}

def test_model(model_state_dict):
    """
    在MNIST测试数据集上测试给定的模型状态字典。
    input: model_state_dict (dict): 模型状态字典。
    output: float, 模型在测试集上的准确率。
    """
     # 此处的测试逻辑可能需要根据training_config中的metrics进行调整，以支持不同的评估指标

    test_loader = load_test_data()
    model = get_model()
    model.load_state_dict({k: torch.tensor(v) for k, v in model_state_dict.items()})
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            # data = data.view(data.size(0), -1)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = correct / total
    print(f'模型在测试图像上的准确率: {accuracy * 100:.2f}%')
    return accuracy

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """
    API端点，用于接收全局模型状态，进行训练和测试。
    input: 通过POST请求的JSON体接收的全局模型状态。
    output: JSON响应，包含本地模型更新和准确率。
    """
    received_data = request.json
    global_model_state_dict = received_data['global_model']

    local_model_update = train_model_one_round(global_model_state_dict)
    accuracy = test_model(local_model_update)

    local_model_update = apply_security_measures(local_model_update)
    return jsonify({"model_update": local_model_update, "accuracy": accuracy})


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
    load_data(train=True)

if __name__ == "__main__":
    register_client()  # 注册客户端以获取ID
    try:
        app.run(host='0.0.0.0', port=client_port, debug=False)
    finally:
        unregister_client()  # 确保应用退出时注销客户端
    # mytest()




