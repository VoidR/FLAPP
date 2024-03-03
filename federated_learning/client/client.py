import torch
import json
import requests
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from flask import Flask, request, jsonify
from federated_learning.models.SimpleModel import SimpleModel


app = Flask(__name__)

client_id = None  # 客户端唯一标识符

# 定义数据加载函数
def load_data():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = datasets.MNIST(root='federated_learning/client/data/MNIST/raw', train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    return loader


def train_model(global_model_state_dict):
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
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
    print(f'Accuracy of the model on the test images: {accuracy * 100:.2f}%')
    return accuracy


def register_client():
    global client_id
    url = "http://127.0.0.1:5000/api/register_client"
    response = requests.post(url)
    if response.status_code == 200:
        client_id = response.json()['client_id']
        print(f"Client registered with ID: {client_id}")
    else:
        print("Failed to register client.")
    return client_id


def unregister_client():
    global client_id
    if client_id is None:
        print("Client is not registered.")
        return False
    url = f"http://127.0.0.1:5000/api/unregister_client/{client_id}"
    response = requests.delete(url)
    if response.status_code == 200:
        print("Client unregistered successfully.")
        client_id = None
        return True
    else:
        print("Failed to unregister client.")
        return False


@app.route('/api/start_training', methods=['GET'])
def start_training():
    if client_id is None:
        print("Client is not registered.")
        return jsonify({"message": "Client is not registered."}), 400

    global_model_response = requests.get('http://127.0.0.1:5000/api/distribute_model')
    if global_model_response.status_code == 200:
        global_model_data = global_model_response.json()
        local_model_update = train_model(global_model_data['global_model'])
        accuracy = test_model(local_model_update)
        send_update_response = requests.post('http://127.0.0.1:5000/api/update_model', json={"client_id": client_id, "model_update": local_model_update, "accuracy": accuracy})
        if send_update_response.status_code == 200:
            return jsonify({"message": "Training completed and update sent successfully."}), 200
        else:
            return jsonify({"message": "Failed to send model update."}), send_update_response.status_code
    else:
        return jsonify({"message": "Failed to get global model from server."}), global_model_response.status_code

if __name__ == "__main__":
    client_id = register_client()  # 注册客户端以获取ID
    try:
        app.run(host='0.0.0.0', port=5001, debug=False)
    finally:
        unregister_client()  # 确保在应用退出时注销客户端
