import torch
import requests
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from flask import Flask, request, jsonify
from federated_learning.models.SimpleModel import SimpleModel
import argparse

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

def register_client():
    """
    向服务器注册客户端，以参与联邦学习。
    入参: 无
    出参: 无，但函数会更新全局变量`client_id`和`training_config`
    """
    response = requests.post(f"{server_url}/api/register_client", json={"port": client_port})
    if response.status_code == 200:
        global client_id, training_config
        client_id = response.json()['client_id']
        training_config = response.json()['training_config']
        print(f"注册成功，ID: {client_id}, 训练配置: {training_config}")

def load_data():
    """
    加载MNIST数据集进行训练或测试。
    入参: 无
    出参: DataLoader实例，用于训练或测试数据的迭代。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='federated_learning/client/data', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader

def train_model_one_epoch(global_model_state_dict):
    """
    使用服务器提供的全局模型状态字典训练模型一个周期。
    入参: global_model_state_dict (dict): 服务器下发的全局模型状态字典。
    出参: dict, 训练一个周期后模型的状态字典。
    """
    model = SimpleModel()
    model.load_state_dict({k: torch.tensor(v) for k, v in global_model_state_dict.items()})
    data_loader = load_data()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    model.train()
    for data, target in data_loader:
        data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    return {k: v.tolist() for k, v in model.state_dict().items()}

def test_model(model_state_dict):
    """
    在MNIST测试数据集上测试给定的模型状态字典。
    入参: model_state_dict (dict): 模型状态字典。
    出参: float, 模型在测试集上的准确率。
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='federated_learning/client/data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    model = SimpleModel()
    model.load_state_dict({k: torch.tensor(v) for k, v in model_state_dict.items()})
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.view(data.size(0), -1)
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
    入参: 通过POST请求的JSON体接收的全局模型状态。
    出参: JSON响应，包含本地模型更新和准确率。
    """
    received_data = request.json
    global_model_state_dict = received_data['global_model']

    local_model_update = train_model_one_epoch(global_model_state_dict)
    accuracy = test_model(local_model_update)

    return jsonify({"model_update": local_model_update, "accuracy": accuracy})

def unregister_client():
    """
    从服务器注销客户端，清理与此客户端相关的服务器端状态。
    入参: 无
    出参: bool, 表示是否成功注销客户端。
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

if __name__ == "__main__":
    register_client()  # 注册客户端以获取ID
    try:
        app.run(host='0.0.0.0', port=client_port, debug=False)
    finally:
        unregister_client()  # 确保应用退出时注销客户端
