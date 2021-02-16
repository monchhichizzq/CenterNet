# -*- coding: utf-8 -*-
# @Time    : 2021/2/13 21:05
# @Author  : Zeqi@@
# @FileName: centernet.py
# @Software: PyCharm


import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from models.backbone import ResNet18, ResNet50
from models.head import centernet_head
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Lambda
from Loss.model_loss import loss

class CenterNet:
    def __init__(self, input_shape, num_classes, max_objects=100):
        self.num_classes = num_classes
        self.output_size = input_shape[0] // 4
        self.image_input = Input(shape=input_shape)
        self.hm_input = Input(shape=(self.output_size, self.output_size, num_classes))
        self.wh_input = Input(shape=(max_objects, 2))
        self.reg_input = Input(shape=(max_objects, 2))
        self.reg_mask_input = Input(shape=(max_objects,))
        self.index_input = Input(shape=(max_objects,))

    def __call__(self, mode='train', backbone='ResNet18', *args, **kwargs):
        if backbone=='ResNet18':
            backbone_out = ResNet18(self.image_input)
        elif backbone=='ResNet50':
            backbone_out = ResNet50(self.image_input)
        y1, y2, y3 = centernet_head(backbone_out, self.num_classes)
        if mode == "train":
            loss_ = Lambda(loss, output_shape=(1,), name='centernet_loss')(
                [y1, y2, y3, self.hm_input, self.wh_input, self.reg_input, self.reg_mask_input, self.index_input])
            model = Model(inputs=[self.image_input, self.hm_input, self.wh_input, self.reg_input, self.reg_mask_input, self.index_input], outputs=loss_)
            return model


        # else:
        #     detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
        #                                          num_classes=num_classes))([y1, y2, y3])
        #     prediction_model = Model(inputs=image_input, outputs=detections)
        #     return prediction_model


# def centernet(input_shape, num_classes, backbone='resnet50', max_objects=100, mode="train", num_stacks=2):
#     assert backbone in ['resnet50', 'hourglass']
#     output_size = input_shape[0] // 4
#     image_input = Input(shape=input_shape)
#     hm_input = Input(shape=(output_size, output_size, num_classes))
#     wh_input = Input(shape=(max_objects, 2))
#     reg_input = Input(shape=(max_objects, 2))
#     reg_mask_input = Input(shape=(max_objects,))
#     index_input = Input(shape=(max_objects,))
#
#     if backbone=='resnet50':
#         #-----------------------------------#
#         #   对输入图片进行特征提取
#         #   512, 512, 3 -> 16, 16, 2048
#         #-----------------------------------#
#         C5 = ResNet50(image_input)
#         #--------------------------------------------------------------------------------------------------------#
#         #   对获取到的特征进行上采样，进行分类预测和回归预测
#         #   16, 16, 2048 -> 32, 32, 256 -> 64, 64, 128 -> 128, 128, 64 -> 128, 128, 64 -> 128, 128, num_classes
#         #                                                              -> 128, 128, 64 -> 128, 128, 2
#         #                                                              -> 128, 128, 64 -> 128, 128, 2
#         #--------------------------------------------------------------------------------------------------------#
#         y1, y2, y3 = centernet_head(C5,num_classes)
#
#         if mode=="train":
#             loss_ = Lambda(loss, name='centernet_loss')([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
#             model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=[loss_])
#             return model
#         else:
#             detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
#                                                 num_classes=num_classes))([y1, y2, y3])
#             prediction_model = Model(inputs=image_input, outputs=detections)
#             return prediction_model
#
#     else:
#         outs = HourglassNetwork(image_input,num_stacks,num_classes)
#
#         if mode=="train":
#             loss_all = []
#             for out in outs:
#                 y1, y2, y3 = out
#                 loss_ = Lambda(loss)([y1, y2, y3, hm_input, wh_input, reg_input, reg_mask_input, index_input])
#                 loss_all.append(loss_)
#             loss_all = Lambda(tf.reduce_mean,name='centernet_loss')(loss_all)
#
#             model = Model(inputs=[image_input, hm_input, wh_input, reg_input, reg_mask_input, index_input], outputs=loss_all)
#             return model
#         else:
#             y1, y2, y3 = outs[-1]
#             detections = Lambda(lambda x: decode(*x, max_objects=max_objects,
#                                                 num_classes=num_classes))([y1, y2, y3])
#             prediction_model = Model(inputs=image_input, outputs=[detections])
#             return prediction_model
