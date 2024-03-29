#FLAPP/federated_learning/server/server.py
import requests
import torch
from flask import Flask, request, jsonify
from federated_learning.models.SimpleModel import SimpleModel
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

app = Flask(__name__)

# 全局模型变量初始化
global_model = None
# 客户端注册表，存储客户端ID, 端口号和最后一次更新轮次
client_registry = {}
# 线程锁，确保对全局变量的操作是线程安全的
lock = threading.Lock()
# 训练配置
training_config = {
  "model":"NN",
  "dataset":"MNIST",
  "optimizer":"Adam",
  "loss":"CrossEntropy",
  "metrics":["Accuracy"],
  "global_epochs":5,
  "local_epochs":2,
  "batch_size":32,
  "learning_rate":0.001
}


def init_model():
    """
    初始化全局模型。
    返回:
        dict: 模型的初始状态字典，键为层名，值为权重和偏置的列表。
    """
    model = SimpleModel()
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
        client_port = client_data['port']
        client_id = str(uuid.uuid4())
        client_registry[client_id] = {"port": client_port, "last_update": None}
        print(f"Client {client_id} registered with port {client_port}.")
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
            return jsonify({"message": "Client unregistered successfully."}), 200
        else:
            print(f"Client ID {client_id} not found for unregistration.")
            return jsonify({"message": "Client ID not found."}), 404

@app.route('/api/start_training', methods=['GET'])
def start_training():
    """
    启动模型训练的后台线程。
    返回:
        Flask Response: 表示训练开始的JSON响应。
    """
    start_training_thread = threading.Thread(target=start_training_rounds, args=(training_config["global_epochs"],))
    start_training_thread.start()
    return jsonify({"message": "Training started."}), 200

def start_training_rounds(rounds=training_config["global_epochs"]):
    """
    在指定轮数内进行模型训练。
    参数:
        rounds (int): 训练轮数。
    """
    global global_model
    if global_model is None:
        global_model = init_model()
    for round_number in range(1, rounds + 1):
        with ThreadPoolExecutor(max_workers=len(client_registry)) as executor:
            future_to_client_id = {
                executor.submit(train_client_model, client_id, client_info): client_id
                for client_id, client_info in client_registry.items()
            }
            model_updates = []
            for future in as_completed(future_to_client_id):
                model_update = future.result()
                if model_update is not None:
                    model_updates.append(model_update)
            global_model = aggregate_model_updates(model_updates, round_number)
    # Save the final model
    torch.save(global_model, 'federated_learning/server/final_model.pth')
    print("Training completed and model saved as final_model.pth")

def train_client_model(client_id, client_info):
    """
    向指定客户端发送模型训练请求。
    参数:
        client_id (str): 客户端ID。
        client_info (dict): 包含客户端信息的字典。
    返回:
        dict: 客户端返回的模型更新；如果请求失败，返回None。
    """
    client_url = f"http://127.0.0.1:{client_info['port']}/api/train_model"
    try:
        response = requests.post(client_url, json={"global_model": global_model})
        if response.status_code == 200:
            return response.json()['model_update']
    except requests.exceptions.RequestException as e:
        print(f"Failed to send request to client {client_id}: {e}")
    return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
