# -*- coding: utf-8 -*-
# @Time    : 2021/2/14 0:33
# @Author  : Zeqi@@
# @FileName: train.py
# @Software: PyCharm

import os
import logging
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from Callbacks.checkpoints import ModelCheckpoint
from Preprocess.data_loader import Coco_DataGenerator
from models.centernet import CenterNet


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('CenterNet-ResNet18')
logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    # Hyperparameters
    Lr = 1e-3
    epoch = 500
    input_shape = (448, 448, 3)
    batch_size = 32
    data_path = 'D:/Coco_dataset/coco_voc/'
    annotation_path = 'Preparation/data_txt'
    plot = False
    log_dir = 'logs'

    # Classes
    classes_path = 'Preparation/data_txt/coco2017_classes.txt'
    class_names = get_classes(classes_path)
    num_classes = len(class_names)

    # Model
    model = CenterNet(input_shape, num_classes, max_objects=100)(mode='train')
    # model.summary()
    logger.info('Bulid model!')
    # model.save('resnet18_centernet.h5')
    logger.info('Input: {}'.format(model.input))
    logger.info('Output: {}'.format(model.output))

    # Data loader
    train_params = {'train': True,
                    'batch_size': batch_size,
                    'input_shape': input_shape,
                    'mosaic': True,
                    'data_path': data_path ,
                    'annotation_path': annotation_path,
                    'classes_path': classes_path,
                    'plot': plot}
    val_params = {'train': False,
                    'batch_size': batch_size,
                    'input_shape': input_shape,
                    'mosaic': False,
                    'data_path': data_path,
                    'annotation_path': annotation_path,
                    'classes_path': classes_path,
                    'plot': plot}

    train_generator = Coco_DataGenerator(**train_params)
    val_generator = Coco_DataGenerator(**val_params)

    # out = model.predict(val_generator)
    # print(out)

    # Callbacks
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    os.makedirs("board_logs", exist_ok=True)
    logging = TensorBoard(log_dir="board_logs")
    checkpoint = ModelCheckpoint(log_dir + "/ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
                                 monitor='val_loss', save_weights_only=False, save_best_only=True, period=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=20, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=6, verbose=1)

    # Loss
    # center_loss = CenterNet_Loss(num_classes=num_classes)
    # eager mode
    # model.compile(loss=center_loss.total_loss, optimizer=Adam(Lr) )
    #               # metrics=[CenterNet_Loss().hm_loss, CenterNet_Loss.wh_loss, CenterNet_Loss.reg_loss])

    # 输入参数是y_true, y_pred: y_pred，代表模型的真实值和预测值，该匿名函数的返回值是y_pred
    model.compile(loss={'centernet_loss': lambda y_true, y_pred: y_pred}, optimizer=Adam(Lr))

    # Train
    model.fit(train_generator,
              validation_data=val_generator,
              epochs=epoch,
              verbose=1,
              callbacks=[checkpoint, reduce_lr])

