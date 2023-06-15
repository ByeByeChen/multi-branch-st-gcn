import os
import random
import shutil
import json

# 设置训练数据集和测试数据集的比例
train_ratio = 0.8
test_ratio = 0.2

# 设置文件夹路径
folder_path = "./jsonData"

# 获取文件列表
file_list = os.listdir(folder_path)
train_folder_path = "./kinetics-skeleton/kinetics_train"
test_folder_path = "./kinetics-skeleton/kinetics_val"
# 随机打乱文件列表
random.shuffle(file_list)

# 计算训练数据集和测试数据集的数量
train_num = int(len(file_list) * train_ratio)
test_num = len(file_list) - train_num

# 创建训练数据集和测试数据集的文件夹
#train_folder = os.path.join(train_folder_path, "train")
#test_folder = os.path.join(test_folder_path, "test")
os.makedirs(train_folder_path, exist_ok=True)
os.makedirs(test_folder_path, exist_ok=True)

# 初始化json数据
train_json = {}
test_json = {}
for i, file_name in enumerate(file_list):
    with open(os.path.join(folder_path, file_name), "r") as f:
        data = json.load(f)
        label = data["label"]
        label_index = data["label_index"]
    if i < train_num:
        # 复制文件到训练数据集的文件夹中
        new_file_name = os.path.splitext(file_name)[0]
        shutil.copy(os.path.join(folder_path, file_name), os.path.join(train_folder_path, file_name))
        # 添加数据到训练数据集的json中
        train_json[new_file_name] = {
            "has_skeleton": True,
            "label": label,
            "label_index": label_index
        }
    else:
        # 复制文件到测试数据集的文件夹中
        new_file_name = os.path.splitext(file_name)[0]
        shutil.copy(os.path.join(folder_path, file_name), os.path.join(test_folder_path, file_name))
        # 添加数据到测试数据集的json中
        test_json[new_file_name] = {
            "has_skeleton": True,
            "label": label,
            "label_index": label_index
        }

# 保存训练数据集和测试数据集的json文件
with open(os.path.join("./kinetics-skeleton", "kinetics_train_label.json"), "w") as f:
    json.dump(train_json, f)
with open(os.path.join("./kinetics-skeleton", "kinetics_val_label.json"), "w") as f:
    json.dump(test_json, f)
