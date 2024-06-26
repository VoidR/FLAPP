#FLAPP/federated_learning/server/server.py
import os
import csv
import json
import requests
import time
import argparse
import torch
from flask import Flask, request, jsonify

import threading
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
# import sqlite3
# import pickle
import federated_learning.models.model_processing as model_processing

parser = argparse.ArgumentParser(description='服务器启动配置')
parser.add_argument('-p', '--port', type=int, default=5000, help='服务器端口号，默认为5000')
args = parser.parse_args()

app = Flask(__name__)

# 全局模型变量初始化
global_model = None
# 客户端注册表，存储客户端ID, 端口号和最后一次更新轮次
client_registry = {}
# 线程锁，确保对全局变量的操作是线程安全的
lock = threading.Lock()
current_round = None
# 保存目录
save_dir = f'federated_learning/server/save/{time.strftime("%Y%m%d-%H%M%S")}'

# 训练配置
model_config = ["NN", "ResNet20", "MLP","LR", "LeNet","AlexNet","CNN"]
dataset_config = ["MNIST", "CIFAR10","Iris","Wine","Breast_cancer"]
metrics_config = ["Accuracy", "Loss", "Precision", "Recall", "F1"]
optimizer_config = ['SGD','Adam']

training_config = {
    "model":"ResNet20",
    "dataset":"CIFAR10",
    "optimizer":"Adam",
    "loss":"CrossEntropy",
    "metrics":["Accuracy","Loss"],
    "global_epochs":200,
    "local_epochs":1,
    "batch_size":64,
    "learning_rate":0.1,
    "client_use_differential_privacy": False,
    "differential_privacy": {
        "epsilon": 1.0,   
        "delta": 1e-5,  
        "sensitivity": 1.0 
    },
    "protect_global_model": True,
    "protect_client_models": False
}

aggregate_time_stats = {
    "aggregate_time": []
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

def save_stats(save_results):
    """
    保存统计信息到文件。
    input: save_results(Dict) 当前轮次的统计信息
    """
    save_file = f'{save_dir}/stats.csv'
    file_exists = os.path.isfile(save_file)
    with open(save_file, 'a+', newline='') as csvfile:
        fieldnames = ['round', 'aggregate_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(save_results)

def save_training_config():
    """
    保存训练配置到文件。
    input: 无
    output: 无
    """
    with open(f'{save_dir}/training_config.json', 'w') as f:
        json.dump(training_config, f)

def aggregate_model_updates(model_updates, round_number):
    """
    聚合来自多个客户端的模型更新。
    参数:
        model_updates (list): 包含每个客户端模型更新的列表，每个更新为一个字典。
    返回:
        dict: 聚合后的模型更新，格式与单个模型更新相同。
    """
    global global_model
    num_updates = len(model_updates)
    aggregated_update = {}
    if training_config.get("model") == "ResNet20" and training_config.get("protect_global_model") == True:  
        # 初始模型
        # current_modules = global_model.fl_modules()
        # counter = 1 
        # for client_model in model_updates:
        #     for m_n, m in current_modules.items():
        #         current_layer = current_modules[m_n]
        #         current_layer.aggregate_grad(current_layer.correction(client_model["gamma"], client_model["v"], client_model[m_n]["post_data"], torch.tensor(client_model[m_n]["grad"]), torch.tensor(client_model[m_n]["r"])), counter)
        #         current_layer.update(float(training_config.get("learning_rate",0.001)) / counter)
        #     counter += 1        
        # current_modules = global_model.fl_modules()
        counter = 1 
        for client_model in model_updates:
            for m_n, m in global_model.fl_modules().items():
                current_layer = global_model.fl_modules()[m_n]
                current_layer.aggregate_grad(current_layer.correction(client_model["gamma"], client_model["v"], client_model[m_n]["post_data"], torch.tensor(client_model[m_n]["grad"]), torch.tensor(client_model[m_n]["r"])), counter)
                current_layer.update(float(training_config.get("learning_rate",0.001)) / counter)
            counter += 1
        # for m in global_model.fl_modules().items():
        #     m[1].update(float(training_config.get("learning_rate",0.001)) / counter)
        aggregated_update = {k: v / num_updates for k, v in global_model.state_dict().items()}
    else:
        aggregated_update = {k: torch.zeros_like(torch.tensor(v)) for k, v in model_updates[0].items()}
        for update in model_updates:
            for k, v in update.items():
                aggregated_update[k] += torch.tensor(v)
        
        aggregated_update = {k: v / num_updates for k, v in aggregated_update.items()}
    # Log aggregation result
    # with open('federated_learning/server/training_log.txt', 'a') as log_file:
    #     log_file.write(f'Round {round_number}: Aggregation completed.\n')
    print(f'Round {round_number}: Aggregation completed.')
    # return {k: v.tolist() for k, v in aggregated_update.items()}
    # return model_processing.tensor_to_list(aggregated_update)
    return aggregated_update

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
        client_index = len(client_registry) + 1
        client_registry[client_id] = {"client_url": client_url, "last_update": None, "client_index": client_index}
        print(f"Client {client_id} registered successfully with url {client_url}. index: {client_index}")
        print_client_count()
    return jsonify({"client_id": client_id, "training_config": training_config, "client_index":client_index}), 200

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
    save_training_config()
    if len(client_registry) > 0:
        for client_id, client_info in client_registry.items():
            client_url_update = f"{client_info['client_url']}/api/update_training_config"
            response = requests.post(client_url_update, json=training_config)
            if response.status_code != 200:
                print(f"Failed to update training config for client {client_id}.")
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
        global_model = model_processing.get_model(training_config)
    for round_number in range(1, rounds + 1):
        current_round = round_number
        save_results = {}
        save_results["round"] = current_round
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
            # global_model.load_state_dict() = aggregate_model_updates(model_updates, round_number) #todo:客户端上传的json中包含当前轮数
            start_agg_time = time.time()
            global_model.load_state_dict(aggregate_model_updates(model_updates, round_number)) #todo:客户端上传的json中包含当前轮数
            end_agg_time = time.time()
            save_results["aggregate_time"] = end_agg_time - start_agg_time
            # aggregate_time_stats["aggregate_time"].append(end_agg_time - start_agg_time)
        save_stats(save_results)
        model_processing.test_model(global_model, training_config, model_processing.get_loss_function(training_config), round_number,save_file=f'{save_dir}/results.csv')
        
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
    global global_model
    client_url_train = f"{client_info['client_url']}/api/train_model"

    # 在使用ResNet20模型，并需要保护全局模型时，加扰动
    # if training_config.get("model") == "ResNet20" and training_config.get("protect_global_model") == True:  
    #     distribute_model = global_model
    #     # distribute_model.randomize()
    #     model_dict = model_processing.tensor_to_list(distribute_model.state_dict())
    # else:
    #     model_dict = model_processing.tensor_to_list(global_model.state_dict())

    model_dict = model_processing.tensor_to_list(global_model.state_dict())
    try:
        response = requests.post(client_url_train, json={"global_model": model_dict, "current_round": current_round})
        if response.status_code == 200:
            return response.json()['model_update']
    except requests.exceptions.RequestException as e:
        print(f"Failed to send request to client {client_id}: {e}")
    return None

# 查询当前客户端数量
@app.route('/api/get_client_count', methods=['GET'])
def get_client_count():
    """
    查询当前客户端数量。
    返回:
        Flask Response: 包含当前客户端数量的JSON响应。
    """
    print_client_count()
    return jsonify({"client_count": len(client_registry)}), 200

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

# 获取训练配置
@app.route('/api/get_training_config', methods=['GET'])
def get_training_config():
    """
    获取当前训练配置。
    返回:
        Flask Response: 包含当前训练配置的JSON响应。
    """
    return jsonify(training_config), 200

@app.route('/api/get_metrics', methods=['GET'])
def get_metrics():
    """
    API端点，用于获取模型评估指标。
    input: 无
    output: JSON响应，包含模型评估指标。
    """
    metrics = {}
    with open(f'{save_dir}/results.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            round_num = row['Round']
            del row['Round']
            metrics[round_num] = row

    return jsonify(metrics)

if __name__ == '__main__':
    # init_db()
    # save_dir 创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_training_config()
    app.run(host='0.0.0.0', port=args.port, debug=False)

    # init_model()
