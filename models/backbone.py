# -*- coding: utf-8 -*-
# @Time    : 2021/2/13 20:12
# @Author  : Zeqi@@
# @FileName: backbone.py
# @Software: PyCharm

import sys
sys.path.append('../../CenterNet')
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import ReLU

def BasicBlock(input_tensor, kernel_size, filter_num, stage, block, strides=(1, 1)):

    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Conv2D(filter_num, kernel_size, strides,  padding='same', name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=conv_name_base + '2a' + '_bn')(x)
    x = ReLU(name=conv_name_base + '2a'+'_relu')(x)

    x = Conv2D(filter_num, kernel_size, (1, 1), padding='same', name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=conv_name_base + '2b' + '_bn')(x)

    if strides != (1, 1):
        shortcut = Conv2D(filter_num, (1, 1), strides=strides,
                          name=conv_name_base + '1', use_bias=False)(input_tensor)
        shortcut = BatchNormalization(name=conv_name_base + '1' + '_bn')(shortcut)

        x = Add(name=conv_name_base + '_add')([shortcut, x])
        out = ReLU(name=conv_name_base + '_add_relu')(x)
    else:
        x = Add(name=conv_name_base + '_add')([input_tensor, x])
        out = ReLU(name=conv_name_base + '_add_relu')(x)

    return out


def ResNet18(inputs):
    #  layer_params=[2, 2, 2, 2]

    # 448, 448, 3 --> 224, 224, 64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    x = ReLU(name='conv1_relu')(x)

    # 224, 224, 64 -> 112,112,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 112, 112, 64 --> 112, 112, 64
    x = BasicBlock(x, (3, 3), 64, stage=1, block='a')
    x = BasicBlock(x, (3, 3), 64, stage=1, block='b')

    # 112, 112, 64 --> 56, 56, 128
    x = BasicBlock(x, (3, 3), 128, stage=2, block='a', strides=(2, 2))
    x = BasicBlock(x, (3, 3), 128, stage=2, block='b')

    # 56, 56, 256 --> 28, 28, 256
    x = BasicBlock(x, (3, 3), 256, stage=3, block='a', strides=(2, 2))
    x = BasicBlock(x, (3, 3), 256, stage=3, block='b')

    # 28, 28, 256 --> 14, 14, 512
    x = BasicBlock(x, (3, 3), 512, stage=4, block='a', strides=(2, 2))
    x = BasicBlock(x, (3, 3), 512, stage=4, block='b')

    return x




def identity_block(input_tensor, kernel_size, filters, stage, block):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=conv_name_base + '2a' + '_bn')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,padding='same', name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=conv_name_base + '2b' + '_bn')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=conv_name_base + '2c' + '_bn')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):

    filters1, filters2, filters3 = filters

    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = BatchNormalization(name=conv_name_base + '2a' + '_bn')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b', use_bias=False)(x)
    x = BatchNormalization(name=conv_name_base + '2b' + '_bn')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = BatchNormalization(name=conv_name_base + '2c' + '_bn')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = BatchNormalization(name=conv_name_base + '1' + '_bn')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(inputs):
    # 512x512x3
    x = ZeroPadding2D((3, 3))(inputs)
    # 256,256,64
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = BatchNormalization(name='bn_conv1')(x)
    x = Activation('relu')(x)

    # 256,256,64 -> 128,128,64
    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)

    # 128,128,64 -> 128,128,256
    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    # 128,128,256 -> 64,64,512
    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    # 64,64,512 -> 32,32,1024
    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    # 32,32,1024 -> 16,16,2048
    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    return x


if __name__ == '__main__':
    inputs = Input(shape=(224, 224, 3))
    out = ResNet18(inputs)
    model = Model(inputs=inputs, outputs=out, name='resnet18')
    model.summary()

    from pregraite.utils.network_profiler import test

    test(model_name='resnet18', model=model)