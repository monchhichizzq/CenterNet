# -*- coding: utf-8 -*-
# @Time    : 2020/12/28 21:02
# @Author  : Zeqi@@
# @FileName: data_loader.py
# @Software: PyCharm


import sys
sys.path.append('../../CenterNet')
import logging
import os
import cv2
import math
import numpy as np
from tensorflow.keras.utils import Sequence
from Preprocess.utils import Data_augmentation
from Preprocess.mosaic_utils import Data_augmentation_with_Mosaic
from Utils.utils import get_classes
from Preprocess.gt_utils import draw_gaussian, gaussian_radius
from Preprocess.utils import preprocess_image

# import pandas as pd
# pd.set_option('display.max_columns', 1000)
# pd.set_option('display.width', 1000)
# pd.set_option('display.max_colwidth', 1000)

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('CenterNet-DataLoader')
logger.setLevel(logging.DEBUG)

class Coco_DataGenerator(Sequence):
    def __init__(self, train, batch_size, plot = False, **kwargs):
        self.train_mode = train

        self.plot = plot
        self.batch_size = batch_size
        self.max_objects = kwargs.get('max_objects', 100)

        self.data_path = kwargs.get('data_path', 'D:/Coco_dataset/coco_voc/')
        self.annotation_path = kwargs.get('annotation_path', 'Preparation/data_txt')
        self.anno_train_txt = kwargs.get('anno_train_txt', 'coco_train.txt')
        self.anno_val_txt = kwargs.get('anno_val_txt', 'coco_val.txt')

        self.model_image_size = kwargs.get('input_shape', (448, 448, 3))
        self.input_shape = (self.model_image_size[0], self.model_image_size[1])
        self.output_shape = (int(self.input_shape[0]/4) , int(self.input_shape[1]/4)) # 输出是输入的4倍

        # self.consecutive_frames = kwargs.get('consecutive_frames', False)

        self.mosaic = kwargs.get('mosaic', False)

        classes_path = kwargs.get('classes_path', 'Preparation/data_txt/coco2017_classes.txt')
        class_names = get_classes(classes_path)
        self.num_classes = len(class_names)
        # print('Class names: \n {} \n Number of class: {} '.format(class_names, self.num_classes))
        logger.info('Activate mosaic: {}'.format(self.mosaic))


        if train:
            self.shuffle = True
            # self.image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/trainval.txt')).read().strip().split()
            self.annotation_lines = open(os.path.join(self.annotation_path, self.anno_train_txt)).readlines()
            self.num_train = len(self.annotation_lines)
            logger.info('Num train: {}'.format(len(self.annotation_lines)))

        else:
            self.shuffle = False
            # self.image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/test.txt')).read().strip().split()
            self.annotation_lines = open(os.path.join(self.annotation_path, self.anno_val_txt)).readlines()
            self.num_val = len(self.annotation_lines)
            logger.info('Number of class: {} '.format(self.num_classes))
            logger.info('Num val: {}'.format(len(self.annotation_lines)))

        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.annotation_lines))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.annotation_lines) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        mosaic_execution = np.random.rand() < 1.5 # 改动

        # Create batch placeholder
        w, h, c = self.model_image_size
        batch_images = np.zeros((self.batch_size, w, h, c), dtype=np.float32)
        batch_hms = np.zeros((self.batch_size, self.output_shape[0], self.output_shape[1], self.num_classes), dtype=np.float32)
        batch_whs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
        batch_regs = np.zeros((self.batch_size, self.max_objects, 2), dtype=np.float32)
        batch_reg_masks = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)
        batch_indices = np.zeros((self.batch_size, self.max_objects), dtype=np.float32)

        if self.mosaic and mosaic_execution and self.train_mode:

            image_batch, box_batch = [], []

            for k in indexes:
                # print(k, k+4)
                if k+4 > self.num_train:
                    four_annotation_lines = self.annotation_lines[self.num_train-4:self.num_train]    # in order to get a mosaic image, 4 normal images are needed
                else:
                    four_annotation_lines = self.annotation_lines[k:k + 4]

                mosaic_aug = Data_augmentation_with_Mosaic(four_annotation_lines, input_shape=self.input_shape, visual=self.plot)

                image_data, box_data = mosaic_aug.main()
                image_batch.append(image_data)
                box_batch.append(box_data)

            # print(index, np.shape(image_batch), np.shape(box_batch), 'mosaic')
        else:

            # Find list of IDs
            annotation_lines_batch = [self.annotation_lines[k] for k in indexes]

            data_aug = Data_augmentation(input_shape=self.input_shape, visual=self.plot)

            # Generate data
            image_batch, box_batch = [], []
            if self.train_mode:
                for annotation_line in annotation_lines_batch:
                    image, box = data_aug.main_train(annotation_line)
                    image_batch.append(image)
                    box_batch.append(box)
                # print(index, np.shape(image_batch), np.shape(box_batch), 'normal')
            else:
                for annotation_line in annotation_lines_batch:
                    image, box = data_aug.main_val(annotation_line)
                    image_batch.append(image)
                    box_batch.append(box)

        # Image batch: 0~1 (448, 448, 3)
        # label batch: (100, 5)
        image_batch, box_batch = np.array(image_batch), np.array(box_batch)
        # print(image_batch.shape, box_batch.shape)

        # 能不能把这个循环并掉：
        b = 0
        for i, (img, box) in enumerate(zip(image_batch, box_batch)):
            # box 为 box_batch 中的每一个 (100, 5) box cooredinates
            # 把box coordinates从输入框的尺寸转变为输出框 也就是缩放4倍
            box_coordinates = np.array(box[:, :4], dtype=np.float32)
            box_coordinates[:, 0] = box_coordinates[:, 0] / self.input_shape[1] * self.output_shape[1]
            box_coordinates[:, 1] = box_coordinates[:, 1] / self.input_shape[0] * self.output_shape[0]
            box_coordinates[:, 2] = box_coordinates[:, 2] / self.input_shape[1] * self.output_shape[1]
            box_coordinates[:, 3] = box_coordinates[:, 3] / self.input_shape[0] * self.output_shape[0]

            dst = img.copy()
            dst = cv2.resize(dst, (self.output_shape[1], self.output_shape[0]))
            # 每个box 中含此张图片的100个框
            for i in range(len(box)):
                bbox = box_coordinates[i].copy()
                bbox = np.array(bbox)
                bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, self.output_shape[1] - 1) # 将bbox的横坐标截断在[0, 111]之内
                bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, self.output_shape[0] - 1) # 将bbox的纵坐标截断在[0, 111]之内
                cls_id = int(box[i, -1]) # class index

                h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
                if h > 0 and w > 0:
                    # 框的中心点坐标
                    ct = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                    ct_int = ct.astype(np.int32)

                    # 获得热力图
                    radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                    radius = max(0, int(radius))
                    # CenterNet则没有直接抛弃掉周围的点，而是赋予了更小的权重，权重的计算就是简单的用高斯函数
                    # 这个方差用的是cornernet的，不是很合理（更具角坐标计算所得）
                    batch_hms[b, :, :, cls_id] = draw_gaussian(batch_hms[b, :, :, cls_id], ct_int, radius) # 为什么这个radius作用在center上

                    if self.plot:
                        # print(batch_hms[b, :, :, cls_id].shape)
                        gussion_plot = batch_hms[b, :, :, cls_id]
                        gussion_plot = cv2.cvtColor(gussion_plot, cv2.COLOR_GRAY2BGR)
                        dst = cv2.addWeighted(dst, 0.9, gussion_plot, 0.1, 0)
                        cv2.imshow('image', dst)
                        cv2.waitKey(10)

                    # bbox的宽高
                    batch_whs[b, i] = 1. * w, 1. * h
                    # 计算中心偏移量 float->int
                    # 由于降采样和feature map坐标是整数，
                    # 因此在feature map算出来的中心点坐标都是整数，小数点都被舍掉，从而在反解的时候，就会有很大偏差，
                    # 这种偏差会随着stride的增大和增大
                    # 因此，CenterNet也要去学这个偏差
                    batch_regs[b, i] = ct - ct_int
                    # 将对应的mask设置为1，用于排除多余的0
                    batch_reg_masks[b, i] = 1
                    # 表示第ct_int[1]行的第ct_int[0]个。
                    batch_indices[b, i] = ct_int[1] * self.output_shape[0] + ct_int[0]

            if self.plot:
                cv2.waitKey(1000)
            # 将RGB转化成BGR
            img = np.array(img, dtype=np.float32)[:, :, ::-1]
            batch_images[b] = preprocess_image(img)
            b = b + 1
            if b == self.batch_size:
                #ground_truth = np.concatenate([batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices])
                #print(ground_truth.shape)
                # return [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices], np.zeros((self.batch_size,))
                return [batch_images, batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices], np.zeros(self.batch_size)

if __name__ == '__main__':

    train_params = {'train': True,
                    'batch_size': 8,
                    'input_shape': (448, 448, 3),
                    'mosaic': False,
                    'data_path': 'D:/Coco_dataset/coco_voc/',
                    'annotation_path': '../Preparation/data_txt',
                    'classes_path': '../Preparation/data_txt/coco2017_classes.txt',
                    'plot': False}

    train_generator = Coco_DataGenerator(**train_params)
    _generator = iter(train_generator)
    for i in range(60000):
        batch_images, [batch_hms, batch_whs, batch_regs, batch_reg_masks, batch_indices] = next(_generator)
        print('')
        print('batch images', batch_images.shape, np.min(batch_images), np.max(batch_images)) # 原始图片
        print('batch heatmaps', batch_hms.shape)  # 中心点高斯热力图 及 classes
        print('batch whs', batch_whs.shape)  # 宽高
        print('batch_regs', batch_regs.shape) # 偏移量
        print('batch_reg_masks', batch_reg_masks.shape) # 有框无框mask
        print('batch_indices', batch_indices.shape) # 没看懂

        # print(image_batch.shape)
        # for label_ in label_batch:
        #     print(np.shape(label_))
        # print('')
