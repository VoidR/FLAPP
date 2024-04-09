#FLAPP/federated_learning/server/server.py
import requests
import torch
from flask import Flask, request, jsonify

# from federated_learning.models import *
from federated_learning.models.LogisticRegression import LogisticRegressionModel
from federated_learning.models.SimpleModel import SimpleModel
from federated_learning.models.ResNet import resnet20
from federated_learning.models.LeNet import LeNet
from federated_learning.models.AlexNet import AlexNet

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
# import sqlite3
# import pickle


app = Flask(__name__)

# 全局模型变量初始化
global_model = None
# 客户端注册表，存储客户端ID, 端口号和最后一次更新轮次
client_registry = {}
# 线程锁，确保对全局变量的操作是线程安全的
lock = threading.Lock()
current_round = None
# 训练配置
model_config = ["NN", "ResNet20", "MLP","LR", "LeNet","AlexNet"]
dataset_config = ["MNIST", "CIFAR10","Iris","Wine","Breast_cancer"]
metrics_config = ["Accuracy", "Loss", "Precision", "Recall", "F1"]

training_config = {
  "model":"NN",
  "dataset":"MNIST",
  "optimizer":"Adam",
  "loss":"CrossEntropy",
  "metrics":["Accuracy","Loss"],
  "global_epochs":60,
  "local_epochs":1,
  "batch_size":64,
  "learning_rate":0.001,
  "client_use_differential_privacy": True,
  "differential_privacy": {
    "epsilon": 1.0,   
    "delta": 1e-5,  
    "sensitivity": 1.0 
  }
}

# DATABASE_PATH = 'federated_learning/server/federated_learning.db'
# def init_db():
    
#     with sqlite3.connect(DATABASE_PATH) as conn:
#         cursor = conn.cursor()
#         cursor.execute('''
#         CREATE TABLE IF NOT EXISTS models (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             round_number INTEGER NOT NULL,
#             model_state BLOB NOT NULL,
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP NOT NULL
#         )
#         ''')
#         conn.commit()


def init_model():
    """
    初始化全局模型。
    返回:
        dict: 模型的初始状态字典，键为层名，值为权重和偏置的列表。
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
        model = LeNet(dim_out=num_classes)
    elif training_config.get("model") == "AlexNet":
        model = AlexNet(num_classes=num_classes)
    return {k: v.tolist() for k, v in model.state_dict().items()}

def aggregate_model_updates(model_updates, round_number):
    """
    聚合来自多个客户端的模型更新。
    参数:
        model_updates (list): 包含每个客户端模型更新的列表，每个更新为一个字典。
    返回:
        dict: 聚合后的模型更新，格式与单个模型更新相同。
    """
    aggregated_update = {k: torch.zeros_like(torch.tensor(v)) for k, v in model_updates[0].items()}
    for update in model_updates:
        for k, v in update.items():
            aggregated_update[k] += torch.tensor(v)
    num_updates = len(model_updates)
    aggregated_update = {k: v / num_updates for k, v in aggregated_update.items()}
    # Log aggregation result
    with open('federated_learning/server/training_log.txt', 'a') as log_file:
        log_file.write(f'Round {round_number}: Aggregation completed.\n')
    print(f'Round {round_number}: Aggregation completed.')
    return {k: v.tolist() for k, v in aggregated_update.items()}

@app.route('/api/register_client', methods=['POST'])
def register_client():
    """
    注册新客户端，为其分配唯一ID并保存其信息。
    返回:
        Flask Response: 包含客户端ID和训练配置的JSON响应。
    """
    with lock:
        client_data = request.json
        client_url= client_data['client_url']
        client_id = str(uuid.uuid4())
        client_registry[client_id] = {"client_url": client_url, "last_update": None}
        print(f"Client {client_id} registered with url {client_url}.")
        print_client_count()
    return jsonify({"client_id": client_id, "training_config": training_config}), 200

@app.route('/api/unregister_client/<client_id>', methods=['DELETE'])
def unregister_client(client_id):
    """
    注销客户端，从注册表中删除其信息。
    参数:
        client_id (str): 要注销的客户端ID。
    返回:
        Flask Response: 注销成功或失败的JSON响应。
    """
    with lock:
        if client_id in client_registry:
            del client_registry[client_id]
            print(f"Client {client_id} unregistered successfully.")
            print_client_count()
            return jsonify({"message": "Client unregistered successfully."}), 200
        else:
            print(f"Client ID {client_id} not found for unregistration.")
            return jsonify({"message": "Client ID not found."}), 404

@app.route('/api/update_training_config', methods=['POST'])
def update_training_config():
    """
    更新训练配置。
    返回:
        Flask Response: 包含更新后的训练配置的JSON响应。
    """
    global training_config
    new_training_config = request.json
    training_config = new_training_config
    return jsonify(training_config), 200

@app.route('/api/start_training', methods=['GET'])
def start_training():
    """
    启动模型训练的后台线程。
    返回:
        Flask Response: 表示训练开始的JSON响应。
    """
    global current_round
    current_round = 0
    start_training_thread = threading.Thread(target=start_training_rounds, args=(training_config["global_epochs"],))
    start_training_thread.start()
    return jsonify({"message": "Training started."}), 200

def start_training_rounds(rounds=training_config["global_epochs"]):
    """
    在指定轮数内进行模型训练。
    参数:
        rounds (int): 训练轮数。
    """
    global global_model, current_round
    if global_model is None:
        global_model = init_model()
    for round_number in range(1, rounds + 1):
        current_round = round_number
        with ThreadPoolExecutor(max_workers=len(client_registry)) as executor:
            future_to_client_id = {
                executor.submit(train_client_model, client_id, client_info, round_number): client_id
                for client_id, client_info in client_registry.items()
            }
            model_updates = []
            for future in as_completed(future_to_client_id):
                model_update = future.result()
                if model_update is not None:
                    model_updates.append(model_update)
            global_model = aggregate_model_updates(model_updates, round_number) #todo:客户端上传的json中包含当前轮数
    # Save the final model
    torch.save(global_model, 'federated_learning/server/final_model.pth')
    print("Training completed and model saved as final_model.pth")

def train_client_model(client_id, client_info, current_round):
    """
    向指定客户端发送模型训练请求。
    参数:
        client_id (str): 客户端ID。
        client_info (dict): 包含客户端信息的字典。
    返回:
        dict: 客户端返回的模型更新；如果请求失败，返回None。
    """
    client_url_train = f"{client_info['client_url']}/api/train_model"
    try:
        response = requests.post(client_url_train, json={"global_model": global_model, "current_round": current_round})
        if response.status_code == 200:
            return response.json()['model_update']
    except requests.exceptions.RequestException as e:
        print(f"Failed to send request to client {client_id}: {e}")
    return None

# 查询当前客户端数量
# @app.route('/api/get_client_count', methods=['GET'])
def print_client_count():
    client_count = len(client_registry)
    print("client_count: ",client_count)

# 获取当前训练状态
@app.route('/api/get_training_status', methods=['GET'])
def get_training_status():
    """
    获取当前训练状态，是否正在训练,训练是否结束。
    返回:
        Flask Response: 包含当前训练状态的JSON响应。
    """
    global current_round
    if current_round is None:
        return jsonify({"training_status": "Not started", "current_round": None}), 200
    elif current_round == training_config["global_epochs"]:
        return jsonify({"training_status": "Completed", "current_round": current_round}), 200
    else:
        return jsonify({"training_status": "In progress", "current_round": current_round}), 200

if __name__ == '__main__':
    # init_db()
    app.run(host='0.0.0.0', port=5000, debug=False)

    # init_model()
