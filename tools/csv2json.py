import pandas as pd
import json
import os
import glob
import numpy as np
import random

# 定义缩放因子、旋转角度和平移向量


fileIndex = 0
fileClass = 'head'
labelIndex = 0
frameWith = 720
frameHeight = 540
# 获取文件夹路径
folder_path = './csv_data/'+fileClass
# 遍历文件夹下的所有csv文件
for file_path in glob.glob(os.path.join(folder_path, '*.csv')):
    fileIndex = fileIndex + 1
    enhancementIndex = 1
    # 读取csv文件
    print("file_path:", file_path)
    print("fileIndex:", fileIndex)
    df = pd.read_csv(file_path, delimiter=',', skiprows=2)

    scale_factor = 1
    x_translation_vector = 0
    y_translation_vector = 0
    # 将数据转换为json格式
    for i in range(5):
        data = []
        for i in range(len(df)):
            frame_index = i + 1
            skeleton = []
            pose = []
            score = []
            for j in range(len(df.columns) - 2):
                if j % 3 == 0:
                    # # 旋转
                    # rotation_matrix = np.array([[np.cos(rotation_angle), -np.sin(rotation_angle)],
                    #                             [np.sin(rotation_angle), np.cos(rotation_angle)]])
                    # keypoints = np.dot(rotation_matrix, keypoints.T).T
                    pose.append((df.iloc[i, j+1] * scale_factor + x_translation_vector) / frameWith)
                    pose.append((df.iloc[i, j+2] * scale_factor + y_translation_vector) / frameHeight)
                    score.append(df.iloc[i, j+3])
            skeleton.append({"pose": pose, "score": score})
            pose = []
            score = []
            data.append({"frame_index": frame_index, "skeleton": skeleton})

        # 将结果保存为json文件
        result = {"data": data, "label": fileClass, "label_index": labelIndex}
        with open("./json_data/"+fileClass+"/"+fileClass+"_"+str(fileIndex)+"_"+str(enhancementIndex)+".json", "w") as f:
            json.dump(result, f)

        enhancementIndex = enhancementIndex + 1
        scale_factor = random.randint(97, 103) / 100
        print("scale_factor:", scale_factor)
        # rotation_angle = np.pi / 4
        x_translation_vector = random.randint(1, 3)
        y_translation_vector = random.randint(1, 3)
        print("x_translation_vector:", x_translation_vector)
        print("y_translation_vector:", y_translation_vector)
