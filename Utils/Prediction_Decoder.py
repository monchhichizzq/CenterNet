# -*- coding: utf-8 -*-
# @Time    : 2021/2/15 23:03
# @Author  : Zeqi@@
# @FileName: Prediction_Decoder.py
# @Software: PyCharm

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import MaxPooling2D

def nms(heat, kernel=3):
    hmax = MaxPooling2D((kernel, kernel), strides=1, padding='SAME')(heat)
    heat = tf.where(tf.equal(hmax, heat), heat, tf.zeros_like(heat))
    return heat

def topk(hm, max_objects=100):
    #-------------------------------------------------------------------------#
    #   当利用512x512x3图片进行coco数据集预测的时候
    #   h = w = 128 num_classes = 80
    #   Hot map热力图 -> b, 128, 128, 80
    #   进行热力图的非极大抑制，利用3x3的卷积对热力图进行最大值筛选
    #   找出一定区域内，得分最大的特征点。
    #-------------------------------------------------------------------------#
    # print(hm, np.shape(hm), np.min(hm), np.max(hm))
    hm = nms(hm)
    b, h, w, c = tf.shape(hm)[0], tf.shape(hm)[1], tf.shape(hm)[2], tf.shape(hm)[3]
    #-------------------------------------------#
    #   将所有结果平铺，获得(b, 128 * 128 * 80)
    #-------------------------------------------#
    hm = tf.reshape(hm, (b, -1))
    #-----------------------------#
    #   (b, k), (b, k)
    #-----------------------------#
    scores, indices = tf.math.top_k(hm, k=max_objects, sorted=True)

    #--------------------------------------#
    #   计算求出种类、网格点以及索引。
    #--------------------------------------#
    class_ids = indices % c
    xs = indices // c % w
    ys = indices // c // w
    indices = ys * w + xs
    return scores, indices, class_ids, xs, ys


def decode(hm, wh, reg, max_objects=100, num_classes=20):
    # -----------------------------------------------------#
    #   hm          b, 128, 128, num_classes
    #   wh          b, 128, 128, 2
    #   reg         b, 128, 128, 2
    #   scores      b, max_objects
    #   indices     b, max_objects
    #   class_ids   b, max_objects
    #   xs          b, max_objects
    #   ys          b, max_objects
    # -----------------------------------------------------#
    scores, indices, class_ids, xs, ys = topk(hm, max_objects=max_objects)
    b = tf.shape(hm)[0]

    # -----------------------------------------------------#
    #   wh          b, 128 * 128, 2
    #   reg         b, 128 * 128, 2
    # -----------------------------------------------------#
    reg = tf.reshape(reg, [b, -1, 2])
    wh = tf.reshape(wh, [b, -1, 2])
    length = tf.shape(wh)[1]

    # -----------------------------------------------------#
    #   找到其在1维上的索引
    #   batch_idx   b, max_objects
    # -----------------------------------------------------#
    batch_idx = tf.expand_dims(tf.range(0, b), 1)
    batch_idx = tf.tile(batch_idx, (1, max_objects))
    full_indices = tf.reshape(batch_idx, [-1]) * tf.cast(length, tf.int32) + tf.reshape(indices, [-1])

    # -----------------------------------------------------#
    #   取出top_k个框对应的参数
    # -----------------------------------------------------#
    topk_reg = tf.gather(tf.reshape(reg, [-1, 2]), full_indices)
    topk_reg = tf.reshape(topk_reg, [b, -1, 2])

    topk_wh = tf.gather(tf.reshape(wh, [-1, 2]), full_indices)
    topk_wh = tf.reshape(topk_wh, [b, -1, 2])

    # -----------------------------------------------------#
    #   利用参数获得调整后预测框的中心
    #   topk_cx     b,k,1
    #   topk_cy     b,k,1
    # -----------------------------------------------------#
    topk_cx = tf.cast(tf.expand_dims(xs, axis=-1), tf.float32) + topk_reg[..., 0:1]
    topk_cy = tf.cast(tf.expand_dims(ys, axis=-1), tf.float32) + topk_reg[..., 1:2]

    # -----------------------------------------------------#
    #   计算预测框左上角和右下角
    #   topk_x1     b,k,1       预测框左上角x轴坐标
    #   topk_y1     b,k,1       预测框左上角y轴坐标
    #   topk_x2     b,k,1       预测框右下角x轴坐标
    #   topk_y2     b,k,1       预测框右下角y轴坐标
    # -----------------------------------------------------#
    topk_x1, topk_y1 = topk_cx - topk_wh[..., 0:1] / 2, topk_cy - topk_wh[..., 1:2] / 2
    topk_x2, topk_y2 = topk_cx + topk_wh[..., 0:1] / 2, topk_cy + topk_wh[..., 1:2] / 2

    # -----------------------------------------------------#
    #   scores      b,k,1       预测框得分
    #   class_ids   b,k,1       预测框种类
    # -----------------------------------------------------#
    scores = tf.expand_dims(scores, axis=-1)
    class_ids = tf.cast(tf.expand_dims(class_ids, axis=-1), tf.float32)

    # -----------------------------------------------------#
    #   detections  预测框所有参数的堆叠
    #   前四个是预测框的坐标，后两个是预测框的得分与种类
    # -----------------------------------------------------#
    detections = tf.concat([topk_x1, topk_y1, topk_x2, topk_y2, scores, class_ids], axis=-1)

    return detections