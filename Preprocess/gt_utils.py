# -*- coding: utf-8 -*-
# @Time    : 2021/2/13 17:56
# @Author  : Zeqi@@
# @FileName: gt_utils.py
# @Software: PyCharm

import numpy as np

def gaussian_radius(det_size, min_overlap=0.7):
    '''
    只要预测的corners在top-left/bottom-right点的某一个半径r内，
    并且其与GTbox的IOU大于一个阈值(一般设为0.7), 我们将将这些点的标签不直接置为0

    问题:
    那问题现在就变成了如何确定半径r，使得IOU与GT box大于0.7的预测框不被直接阉割掉

    :param det_size: h_ceiling, w_ceiling of bounding box
    :param min_overlap:
    :return:
    '''
    height, width = det_size


    # 情况一，预测的框和GTbox两个角点以r为半径的圆外切
    # overlap = h * w / (h+2r)(w+2r)
    # 把他转换成一个一元二次方程然后解出r
    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    # 情况二，预测的框和GTbox两个角点以r为半径的圆内切
    # overlap = (h-2r)(w-2r)/h*w
    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    # 情况三， 预测的框和GTbox两个角点以r为半径的圆一个边内切，一个边外切
    # overlap = (h-r)*(w-r)/[2*h*w - (h-r)*(w-r)]
    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    # 根据这个最小半径设计一个高斯散射核
    # 取最大的r其他的情况就不满足了
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap