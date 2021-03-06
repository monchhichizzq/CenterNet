# -*- coding: utf-8 -*-
# @Time    : 2021/2/14 3:51
# @Author  : Zeqi@@
# @FileName: mAP_Callbacks.py
# @Software: PyCharm


# Opensource libs
import os
import cv2
import shutil
import time
import logging
import colorsys
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont, ImageDraw

# utils
from Preprocess.utils import preprocess_image
from Utils.utils import get_classes, get_anchors, nms, centernet_correct_boxes
from Utils.Prediction_Decoder import decode

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Model
# from tensorflow.keras.mixed_precision import experimental as mixed_precision

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger('Prediction - CenterNet')
logger.setLevel(logging.DEBUG)


class Prediction:
    def __init__(self, **kwargs):
        self.write_down = kwargs.get('write_down', True)
        self.input_shape = kwargs.get('input_shape', (448, 448, 3))

        self.nms_threshold = kwargs.get('nms_threshold', 0.5)
        self.confidence = kwargs.get('confidence', 0.01)
        self.max_boxes = kwargs.get('max_boxes', 100)

        classes_path = kwargs.get('classes_path', None)
        self.class_names = get_classes(classes_path)
        self.pre_path = kwargs.get('pre_path', "input/detection-results")
        self.set_colors()

    def set_colors(self):
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))


    def letterbox_image(self, image, size):
        iw, ih = image.size
        w, h = size
        scale = min(w/iw, h/ih)
        nw = int(iw*scale)
        nh = int(ih*scale)

        image = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
        return new_image

    def detect_image(self, image, image_id, model, *args, **kwargs):
        if self.write_down:
            self.detect_txtfile = open(os.path.join(self.pre_path, image_id + ".txt"), "w")

        # 调整图片使其符合输入要求
        new_image_size = (self.input_shape[1], self.input_shape[0])
        boxed_image = self.letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')[:, :, ::-1] # RGB 转 BGR
        image_data /= 255.
        image_data = preprocess_image(image_data)
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.


        eval_input = model.layers[0].input
        # input_1: 0
        # hm_header_out: 84
        # wh_header_out: 88
        # reg_header_out: 89
        eval_out = [model.layers[87].output,
                    model.layers[88].output,
                    model.layers[89].output]
        eval_net = Model(eval_input, eval_out)

        # prediction
        start = time.time()
        raw_preds = eval_net(image_data)
        end = time.time()
        inference_time = np.round((end - start) * 100, 2)
        fps = int(1 / (end - start))

        # print(raw_preds[0].shape, raw_preds[1].shape, raw_preds[2].shape)
        preds = decode(hm=raw_preds[0], wh=raw_preds[1], reg=raw_preds[2], max_objects=self.max_boxes, num_classes=len(self.class_names))

        # if self.nms:
        # preds = np.array(nms(preds, self.nms_threshold))
        preds = np.array(preds)
        # print('predictions: ', preds.shape)

        if len(preds[0]) <= 0:
            return

        # -----------------------------------------------------------#
        #   将预测结果转换成小数的形式
        # -----------------------------------------------------------#
        preds[0][:, 0:4] = preds[0][:, 0:4] / (self.input_shape[0] / 4)

        det_label = preds[0][:, -1]
        det_conf = preds[0][:, -2]

        det_xmin, det_ymin, det_xmax, det_ymax = preds[0][:, 0], preds[0][:, 1], preds[0][:, 2], preds[0][:, 3]
        # -----------------------------------------------------------#
        #   筛选出其中得分高于confidence的框
        # -----------------------------------------------------------#

        top_indices = [i for i, conf in enumerate(det_conf) if conf >= self.confidence]
        top_conf = det_conf[top_indices]
        top_label_indices = det_label[top_indices].tolist()
        top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(det_xmin[top_indices], -1), np.expand_dims(
            det_ymin[top_indices], -1), np.expand_dims(det_xmax[top_indices], -1), np.expand_dims(
            det_ymax[top_indices], -1)

        image_shape = np.array(np.shape(image)[0:2])
        boxes = centernet_correct_boxes(top_ymin, top_xmin, top_ymax, top_xmax,
                                        np.array([self.input_shape[0], self.input_shape[1]]), image_shape)

        # 设置字体
        font = ImageFont.truetype(font='font/simhei.ttf',
                                  size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        small_pic = []
        for i, c in list(enumerate(top_label_indices)):
            c= int(c)
            predicted_class = self.class_names[int(c)]
            box = boxes[i]
            score = top_conf[i]

            top, left, bottom, right = box

            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            # print(label, top, left, bottom, right)

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[int(c)])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
            # print(draw)
            del draw

            if self.write_down:
                self.detect_txtfile.write("%s %s %s %s %s %s\n" % (
                predicted_class, str(score), str(int(left)), str(int(top)), str(int(right)), str(int(bottom))))



        logging.info('Inference time: {} ms, FPS: {}'.format(inference_time, fps))

        if self.write_down:
            self.detect_txtfile.close()

        num_image = np.array(image)
        bgr_image = cv2.cvtColor(num_image, cv2.COLOR_RGB2BGR)
        cv2.imshow('demo', bgr_image)
        cv2.waitKey(1000)

        return image


class get_gt():
    def __init__(self, data_path, gt_path):
        self.data_path = data_path
        self.gt_path = gt_path
        #os.makedirs(self.gt_path, exist_ok=True)

    def __call__(self, *args, **kwargs):
        # ground_truth = []
        image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/val.txt')).read().strip().split()
        for image_id in tqdm(image_ids):
            with open(os.path.join(self.gt_path, image_id + ".txt"), "w") as new_f:
                root = ET.parse(os.path.join(self.data_path, "Annotations/" + image_id + ".xml")).getroot()
                ground_truth_per_frame=[]
                for obj in root.findall('object'):
                    obj_name = obj.find('name').text
                    bndbox = obj.find('bndbox')
                    left = bndbox.find('xmin').text
                    top = bndbox.find('ymin').text
                    right = bndbox.find('xmax').text
                    bottom = bndbox.find('ymax').text
                    new_f.write("%s %s %s %s %s\n" % (obj_name, left, top, right, bottom))

def read_map(path):
    fp = open(path)
    lines = fp.readlines()
    for i, line in enumerate(lines):
        if line[:3] == 'mAP':
            mAP = float(line.split()[-1].split('%')[0])
    fp.close()
    return mAP

class VOC2012mAP_Callback(tf.keras.callbacks.Callback):

    def __init__(self,
                 input_shape = (448, 448, 3),
                 data_path = 'D:\Coco_dataset\coco_voc',
                 anntation_path = "Preparation/data_txt",
                 gt_path = "input/ground-truth",
                 pre_path = "input/detection-results",
                 results_path = 'input/',
                 val_txt = 'coco_val.txt',
                 classes_path = 'Preparation/data_txt/coco2017_classes.txt',
                 command_line = 'python3 get_map.py',
                 visual=True,
                 **kwargs):
        super(VOC2012mAP_Callback, self).__init__()
        self.visual = visual
        self.input_shape = input_shape

        self.data_path = data_path
        self.annotation_path = anntation_path
        self.gt_path = gt_path
        self.pre_path = pre_path
        self.results_path = results_path
        self.val_txt = val_txt
        self.classes_path = classes_path
        self.command_line = command_line

        self.image_ids = open(os.path.join(self.data_path, 'ImageSets/Main/val.txt')).read().strip().split()
        self.excute_map_calculation()

        # Get ground truth
        get_gt(data_path=self.data_path, gt_path=self.gt_path)()

        self.annotation_lines = open(os.path.join(self.annotation_path, self.val_txt)).readlines()
        self.num_val = len(self.annotation_lines)
        logger.info('Num validation: {}'.format(len(self.annotation_lines)))

    def excute_map_calculation(self):
        if os.path.exists(self.results_path):
            shutil.rmtree(self.results_path)
            print('Cleaned the existing folder: {}'.format(self.results_path))
        os.makedirs(self.results_path)
        print('Created a new folder : {}'.format(self.results_path))
        print('Created a new folder : {}'.format(self.gt_path))
        print('Created a new folder : {}'.format(self.pre_path))
        os.makedirs(self.gt_path, exist_ok=True)
        os.makedirs(self.pre_path, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_predict_begin(self, logs=None):
        self.get_prediction = Prediction(write_down=True,
                                         nms_threshold=0.5,
                                         confidence=0.01,
                                         max_boxes=100,
                                         input_shape=self.input_shape,
                                         classes_path=self.classes_path,
                                         pre_path=self.pre_path)

    def on_predict_batch_end(self, batch, logs=None):
        anno_line = self.annotation_lines[batch]
        print(anno_line)
        line = anno_line.split()
        raw_image = Image.open(line[0])
        self.get_prediction.detect_image(image=raw_image, image_id=self.image_ids[batch], model=self.model)

    def on_predict_end(self, logs=None):
        # self.get_prediction = Prediction(write_down=True,
        #                                  nms_threshold=0.5,
        #                                  confidence=0.01,
        #                                  max_boxes=100,
        #                                  input_shape=self.input_shape,
        #                                  classes_path=self.classes_path,
        #                                  pre_path=self.pre_path)
        #
        # for i, anno_line in enumerate(tqdm(self.annotation_lines)):
        #     line = anno_line.split()
        #     raw_image = Image.open(line[0])
        #     self.get_prediction.detect_image(image=raw_image, image_id=self.image_ids[i], model = self.model)

        os.system(self.command_line)
        mAP = read_map('Callbacks/results/results.txt')
        # logs['mAP'] = mAP
        logger.info('mAP: {} %'.format(mAP))


