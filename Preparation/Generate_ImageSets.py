# -*- coding: utf-8 -*-
# @Time    : 2021/2/13 15:16
# @Author  : Zeqi@@
# @FileName: Generate_ImageSets.py
# @Software: PyCharm

import os

data_path = 'D:/Coco_dataset'
imageset_path = os.path.join(data_path, 'ImageSets/Main')
os.makedirs(imageset_path, exist_ok=True)

imageset_train_file = os.path.join(imageset_path, 'train.txt')
imageset_val_file = os.path.join(imageset_path, 'val.txt')

path = 'D:\Coco_dataset'
train_path = os.path.join(path, 'train2017')
val_path = os.path.join(path, 'val2017')
train_ids = []
val_ids = []

for id in os.listdir(train_path):
    id_name = id.split('.')[0] + '\n'
    train_ids.append(id_name)

with open(os.path.join(imageset_path, 'train.txt'), 'w') as f_train:
    f_train.writelines(train_ids)
    f_train.close()

for id in os.listdir(val_path):
    id_name = id.split('.')[0] + '\n'
    val_ids.append(id_name)

with open(os.path.join(imageset_path, 'val.txt'), 'w') as f_val:
    f_val.writelines(val_ids)
    f_val.close()
