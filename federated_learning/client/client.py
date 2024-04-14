#FLAPP/federated_learning/client/client.py

import os
import csv
import json
import time
import torch
import logging
import requests
from flask import Flask, request, jsonify
import socket
import argparse
# import sqlite3

from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score

from federated_learning.client.utils.DP import dp_protection
import federated_learning.models.model_processing as model_processing


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
client_index = None
# 根据时间戳创建保存目录
save_dir = f'federated_learning/client/save/{time.strftime("%Y%m%d-%H%M%S")}'
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

# 计算时间统计
computation_time_stats = {
    "train_time": []
}
communication_stats = {
    "data_sent": [],
    "data_received": []
}

def save_training_config():
    """
    保存训练配置到文件。
    input: 无
    output: 无
    """
    with open(f'{save_dir}/training_config.json', 'w') as f:
        json.dump(training_config, f)

def save_stats(save_results):
    """
    保存统计信息到文件。
    input: save_results(Dict) 当前轮次的统计信息
    output: 无
    """
    save_file = f'{save_dir}/stats.csv'
    file_exists = os.path.isfile(save_file)
    with open(save_file, 'a+', newline='') as csvfile:
        fieldnames = ['round', 'train_time', 'data_sent', 'data_received']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(save_results)


def register_client():
    """
    向服务器注册客户端，以参与联邦学习。
    input: 无
    output: 无，但函数会更新全局变量`client_id`和`training_config`
    """
    response = requests.post(f"{server_url}/api/register_client", json={"client_url": f"http://{client_IP}:{client_port}"})
    if response.status_code == 200:
        global client_id, training_config, client_index
        client_id = response.json()['client_id']
        training_config = response.json()['training_config']
        client_index = response.json()['client_index']
        save_training_config()
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
    return jsonify(training_config), 200

@app.route('/api/train_model', methods=['POST'])
def train_model():
    """
    API端点，用于接收全局模型状态，进行训练和测试。
    input: 通过POST请求的JSON体接收的全局模型状态。
    output: JSON响应，包含本地模型更新和准确率。
    """
    received_data = request.json
    save_results = {}
    save_results["round"] = received_data["current_round"]
    save_results["data_received"] = len(json.dumps(received_data).encode('utf-8'))
    # communication_stats["data_received"].append(len(json.dumps(received_data).encode('utf-8')))

    start_training_time = time.time()
    local_updated_model = train_model_one_round(received_data)
    end_training_time = time.time()
    # computation_time_stats["train_time"].append(end_training_time - start_training_time)
    save_results["train_time"] = end_training_time - start_training_time
    local_model_update = apply_security_measures(local_updated_model)

    # communication_stats["data_sent"].append(len(json.dumps(local_model_update).encode('utf-8')))
    save_results["data_sent"] = len(json.dumps(local_model_update).encode('utf-8'))
    save_stats(save_results)
    return jsonify({"model_update": local_model_update})


def train_model_one_round(global_model_info):
    """
    使用服务器提供的全局模型状态字典训练模型一个周期。
    input: global_model_info(json): 服务器提供的json, 包括全局模型参数和当前全局轮数。
    output: dict, 训练一个周期后模型的状态字典。
    """
    global_model_state_dict = global_model_info['global_model']
    current_round = global_model_info["current_round"]
    current_client_num = requests.get(f"{server_url}/api/get_client_count").json()["client_count"]
    device = model_processing.get_device()
    print("开始训练，全局轮数:", current_round)
    model = model_processing.get_model(training_config)  # 此处应根据training_config['model']选择不同的模型
    model.load_state_dict({k: torch.tensor(v) for k, v in global_model_state_dict.items()})


    # 将模型移动到正确的设备
    model.to(device)

    # 目前扰动放在客户端 todo：改为服务器扰动。。。
    if training_config.get("model") == "ResNet20" and training_config.get("protect_global_model") == True:
        model.randomize()

    # 输出线性层是否会计算梯度
    # print("before training,linear:", model.linear.fc.weight.requires_grad)

    data_loader = model_processing.load_data(training_config, train=True, client_count=current_client_num,client_index=client_index)
    optimizer = model_processing.get_optimizer(model,training_config)
    loss_function = model_processing.get_loss_function(training_config)
    model.train()

    for _ in range(training_config.get("local_epochs", 1)):  # 进行多轮本地训练
        for data, target in data_loader:
            # data = data.view(data.shape[0], -1)
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            # output = model(data)

            loss = loss_function(model(data), target)
            # print("after loss,linear:", model.linear.fc.weight.requires_grad)
            if training_config.get("model") == "ResNet20" and training_config.get("protect_global_model") == True:
                model.post(target)
            # print("after post,linear:", model.linear.fc.weight.requires_grad)
                
            loss.backward()
            optimizer.step()
    if not training_config.get("protect_global_model"):
        model_processing.test_model(model, training_config, loss_function, current_round, save_file=f'{save_dir}/results.csv')
    return model


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

def apply_security_measures(local_updated_model):
    """
    应用安全措施到模型更新，当前使用差分隐私。
    可以在这里添加更多的安全措施。
    """
    model_update = {}
    if training_config.get("model") == "ResNet20" and training_config.get("protect_global_model") == True:
        for m_n, m in local_updated_model.fl_modules().items():
            model_update[m_n] = {}
            model_update["gamma"] = local_updated_model.gamma
            model_update["v"] = local_updated_model.v
            model_update[m_n]["post_data"] = m.post_data
            model_update[m_n]["grad"] = m.get_grad()
            model_update[m_n]["r"] = m.get_r()
        model_update = model_processing.tensors_to_lists(model_update)
    elif training_config.get("protect_client_model") == True:
        if training_config['client_use_differential_privacy']:
            dp_params = training_config['differential_privacy']
            # 应用差分隐私机制，比如添加高斯噪声
            model_update = dp_protection(local_updated_model, dp_params)
            model_update = model_processing.tensor_to_list(model_update.state_dict())
    else:
        model_update = model_processing.tensor_to_list(local_updated_model.state_dict())
    # 将来可能添加其他安全措施
    return model_update

def mytest():
    print(get_local_ip())

if __name__ == "__main__":
    # save_dir 创建
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    register_client()  # 注册客户端以获取ID
    try:
        app.run(host='0.0.0.0', port=client_port, debug=False)
    finally:
        # save_stats()
        unregister_client()  # 确保应用退出时注销客户端
    # mytest()
