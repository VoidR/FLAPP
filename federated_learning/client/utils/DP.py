import numpy as np

def dp_protection(model_update, dp_params):
    """
    对模型更新应用差分隐私保护。
    
    参数:
        model_update (dict): 模型更新，键为参数名称，值为参数值的列表。
        dp_params (dict): 包含差分隐私参数的字典，例如epsilon和delta值。
    
    返回:
        dict: 添加了差分隐私噪声后的模型更新。
    """
    epsilon = dp_params.get('epsilon', 1.0)
    delta = dp_params.get('delta', 1e-5)
    sensitivity = dp_params.get('sensitivity', 1.0)

    sigma = np.sqrt(2 * np.log(1.25 / delta)) / epsilon

    noised_update = {}
    for k, v in model_update.items():
        # 针对每个参数值的列表，我们需要为列表中的每个元素添加噪声
        noised_value_list = []
        for value in v:
            # 确定每个参数值（可能是ndarray）的噪声
            noise = np.random.normal(0, 0.001, size=np.array(value).shape)
            noised_value = np.array(value) + noise
            # 将噪声添加后的参数值转换为列表
            noised_value_list.append(noised_value.tolist())
        noised_update[k] = noised_value_list

    return noised_update
