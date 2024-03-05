# FLAPP 联邦学习框架

FLAPP（Federated Learning APPlication）是一个基于PyTorch和Flask的简易联邦学习框架，旨在提供一个轻量级、易于理解和扩展的联邦学习实验平台。

## 系统要求

Python 3.x
PyTorch
Flask
torchvision
requests
## 安装

确保您的系统已安装Python 3及以上版本，并通过以下命令安装所需的依赖包：
```bash
pip install torch torchvision flask requests
```
## 结构概述

- federated_learning/
    - client/: 客户端实现，包括模型训练、数据加载、向服务器注册和注销等功能。
        - client.py: 客户端主要逻辑实现。
    - server/: 服务器端实现，包括模型初始化、客户端管理、模型更新聚合等功能。
        - server.py: 服务器端主要逻辑实现。
    - models/: 包含简单的神经网络模型定义。
        -   SimpleModel.py: 简单模型的实现。
## 快速开始

1. 启动服务器: 在服务器机器上运行以下命令以启动联邦学习服务器。

```bash
python -m federated_learning.server.server 
```
2. 启动客户端: 在每个客户端机器上运行以下命令以连接到服务器并开始训练。
```bash
python -m federated_learning.client.client --port 5001
```
可以通过修改 --port 参数为不同值，在同一机器或不同机器上启动多个客户端。

3. 开始训练: 通过服务器端提供的API启动训练过程。
```bash
curl -X GET http://127.0.0.1:5000/api/start_training
```
4. 监控进度: 观察服务器和客户端的控制台输出以监控训练进度和结果。
## 扩展

可以通过修改SimpleModel.py来使用自定义模型，或者调整client.py和server.py中的配置来适应不同的训练参数和数据集。
