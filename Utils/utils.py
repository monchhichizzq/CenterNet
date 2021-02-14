# -*- coding: utf-8 -*-
# @Time    : 2021/1/30 20:11
# @Author  : Zeqi@@
# @FileName: utils.py
# @Software: PyCharm

import numpy as np

# ---------------------------------------------------#
#   获得类和先验框
# ---------------------------------------------------#
def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)


def nms(results, nms):
    outputs = []
    # 对每一个图片进行处理
    for i in range(len(results)):
        # ------------------------------------------------#
        #   具体过程可参考
        #   https://www.bilibili.com/video/BV1Lz411B7nQ
        # ------------------------------------------------#
        detections = results[i]
        unique_class = np.unique(detections[:, -1])

        best_box = []
        if len(unique_class) == 0:
            results.append(best_box)
            continue
        # 对种类进行循环，
        # 非极大抑制的作用是筛选出一定区域内属于同一种类得分最大的框，
        # 对种类进行循环可以帮助我们对每一个类分别进行非极大抑制。
        for c in unique_class:
            cls_mask = detections[:, -1] == c

            detection = detections[cls_mask]
            scores = detection[:, 4]
            # 根据得分对该种类进行从大到小排序。
            arg_sort = np.argsort(scores)[::-1]
            detection = detection[arg_sort]
            while np.shape(detection)[0] > 0:
                # 每次取出得分最大的框，计算其与其它所有预测框的重合程度，重合程度过大的则剔除。
                best_box.append(detection[0])
                if len(detection) == 1:
                    break
                ious = iou(best_box[-1], detection[1:])
                detection = detection[1:][ious < nms]
        outputs.append(best_box)
    return outputs


def iou(b1, b2):
    b1_x1, b1_y1, b1_x2, b1_y2 = b1[0], b1[1], b1[2], b1[3]
    b2_x1, b2_y1, b2_x2, b2_y2 = b2[:, 0], b2[:, 1], b2[:, 2], b2[:, 3]

    inter_rect_x1 = np.maximum(b1_x1, b2_x1)
    inter_rect_y1 = np.maximum(b1_y1, b2_y1)
    inter_rect_x2 = np.minimum(b1_x2, b2_x2)
    inter_rect_y2 = np.minimum(b1_y2, b2_y2)

    inter_area = np.maximum(inter_rect_x2 - inter_rect_x1, 0) * \
                 np.maximum(inter_rect_y2 - inter_rect_y1, 0)

    area_b1 = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    area_b2 = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    iou = inter_area / np.maximum((area_b1 + area_b2 - inter_area), 1e-6)
    return iou


def centernet_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape

    box_yx = np.concatenate(((top+bottom)/2,(left+right)/2),axis=-1)
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)

    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)
    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)
    return boxes