# -*- coding: utf-8 -*-
# @Time    : 2021/2/13 23:03
# @Author  : Zeqi@@
# @FileName: head.py
# @Software: PyCharm


from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import ReLU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Sigmoid


# Multi-scale feature maps fusion

def centernet_head(x,num_classes):
    x = Dropout(rate=0.5)(x)
    #-------------------------------#
    #   解码器
    #-------------------------------#
    num_filters = 256
    # 14, 14, 512  ->  28, 28, 256 -> 56, 56, 128 -> 112, 112, 64
    for i in range(3):
        conv_name= 'up_convtranspose_' + str(i)
        # 进行上采样
        x = Conv2DTranspose(num_filters // pow(2, i), (4, 4), strides=2, use_bias=False, padding='same',
                            kernel_initializer='he_normal',
                            kernel_regularizer=l2(5e-4),
                            name=conv_name)(x)
        x = BatchNormalization(name=conv_name+'_bn')(x)
        x = ReLU(name=conv_name+'_relu')(x)

    # 最终获得128,128,64的特征层
    # hm header
    hm_name = 'hm_header'
    y1 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name=hm_name)(x)
    y1 = BatchNormalization(name=hm_name+'_bn')(y1)
    y1 = ReLU(name=hm_name+'_relu')(y1)
    y1 = Conv2D(num_classes, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name=hm_name + '_out')(y1)
    y1 = Sigmoid()(y1)

    # wh header
    wh_name = 'wh_header'
    y2 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name=wh_name)(x)
    y2 = BatchNormalization(name=wh_name+'_bn')(y2)
    y2 = ReLU(name=wh_name+'_relu')(y2)
    y2 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name=wh_name + '_out')(y2)

    # reg header
    reg_name = 'reg_header'
    y3 = Conv2D(64, 3, padding='same', use_bias=False, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name=reg_name)(x)
    y3 = BatchNormalization(name=reg_name+'_bn')(y3)
    y3 = ReLU(name=reg_name+'_relu')(y3)
    y3 = Conv2D(2, 1, kernel_initializer='he_normal', kernel_regularizer=l2(5e-4), name=reg_name+'_out')(y3)
    return y1, y2, y3