import requests
import torch
from flask import Flask, request, jsonify
from federated_learning.models.SimpleModel import SimpleModel
import threading
import uuid

app = Flask(__name__)

# 假设有一个全局模型变量
global_model = None
round_number = 0
MAX_ROUNDS = 1000  # 假设我们想运行1000轮通信
client_registry = {}  # 存储客户端ID及其最后一次更新时间
lock = threading.Lock()  # 确保对全局变量的操作是线程安全的

# 假设聚合逻辑，这里简单地存储更新
model_updates = []

def init_model():
    model = SimpleModel()
    return {k: v.tolist() for k, v in model.state_dict().items()}

def aggregate_model_updates(model_updates):
    # 使用零初始化聚合更新
    aggregated_update = {k: torch.zeros_like(torch.tensor(v)) for k, v in model_updates[0].items()}
    for update in model_updates:
        for k, v in update.items():
            aggregated_update[k] += torch.tensor(v)
    # 平均更新
    num_updates = len(model_updates)
    aggregated_update = {k: v / num_updates for k, v in aggregated_update.items()}
    return {k: v.tolist() for k, v in aggregated_update.items()}

@app.route('/api/register_client', methods=['POST'])
def register_client():
    with lock:
        client_id = str(uuid.uuid4())  # 生成唯一的客户端ID
        client_registry[client_id] = {"last_update": None}  # 初始注册，没有最后更新时间
    print(f"Client {client_id} registered.")
    return jsonify({"client_id": client_id}), 200

@app.route('/api/unregister_client/<client_id>', methods=['DELETE'])
def unregister_client(client_id):
    with lock:
        if client_id in client_registry:
            del client_registry[client_id]
            print(f"Client {client_id} unregistered.")
            return jsonify({"message": "Client unregistered successfully."}), 200
        else:
            return jsonify({"message": "Client ID not found."}), 404

@app.route('/api/update_model', methods=['POST'])
def receive_update():
    global global_model, round_number
    client_id = request.json.get('client_id')
    update = request.json['model_update']
    accuracy = request.json['accuracy']
    with lock:
        if client_id not in client_registry:
            return jsonify({"message": "Client ID not registered."}), 400
        model_updates.append(update)
        # 更新客户端的最后更新时间
        client_registry[client_id]["last_update"] = round_number

    if len(model_updates) >= len(client_registry):  # 当所有注册的客户端都发送了更新时
        global_model = aggregate_model_updates(model_updates)
        model_updates.clear()  # 清空更新以准备下一轮
        round_number += 1
        with open('federated_learning/server/training_status.txt', 'a') as f:
            f.write(f"Round {round_number} completed with {len(client_registry)} clients. Accuracy: {accuracy}\n")
        if round_number >= MAX_ROUNDS:
            # 训练完成逻辑
            print(f"Training completed after {MAX_ROUNDS} rounds with {len(client_registry)} clients.")
            return jsonify({"message": "Training completed"}), 200
    return jsonify({"message": "Update received"}), 200

@app.route('/api/distribute_model', methods=['GET'])
def distribute_model():
    global global_model
    if global_model is None:
        global_model = init_model()
    return jsonify({"global_model": global_model}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
