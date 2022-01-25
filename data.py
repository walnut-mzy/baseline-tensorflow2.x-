import numpy as np
import tensorflow as tf
import hashlib
def data_process(poses_valid_2d):
    train_video=[]
    for i in poses_valid_2d:
        for k in i.tolist():
                train_video.append(k)
        #计算机内存不够只能减少训练数据
        if len(train_video)>=1000:
            print(np.array(train_video).shape)
            return train_video
    print(np.array(train_video).shape)
    return train_video
def data_process1(poses_valid_2d):
    train_video=[]
    for i in poses_valid_2d:
        for k in i.tolist():
            #print(np.array(k).shape)
            train_video.append(k)
        # 计算机内存不够只能减少训练数据
        if len(train_video) >= 1000:
           print(np.array(train_video).shape)
           return train_video
    print(np.array(train_video).shape)
    return train_video
def deterministic_random(min_value, max_value, data):
    digest = hashlib.sha256(data.encode()).digest()
    raw_value = int.from_bytes(digest[:4], byteorder='little', signed=False)
    return int(raw_value / (2**32 - 1) * (max_value - min_value)) + min_value

