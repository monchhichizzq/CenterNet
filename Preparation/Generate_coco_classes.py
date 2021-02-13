# -*- coding: utf-8 -*-
# @Time    : 2021/2/13 15:45
# @Author  : Zeqi@@
# @FileName: Generate_coco_classes.py
# @Software: PyCharm

cat_2014 = 'D:/Coco_dataset/annotations/instances_val2014.json'
cat_2017 = 'D:/Coco_dataset/annotations/instances_val2017.json'
classname_txt = 'data_txt/coco2017_classes.txt'


import os
import json

def main(year='2017'):
    json_file = None
    if(year == '2014'):
        json_file = cat_2014
    else:
        json_file = cat_2017
    if json_file is not None:
        with open(json_file,'r') as COCO:
            js = json.loads(COCO.read())
            coco_cat = js['categories']
            print(json.dumps(js['categories']))

    classes = []
    class_txt = []

    for i, cat in enumerate(coco_cat):
        print(i, cat, cat['name'])
        classes.append(cat['name'])
        class_txt.append(cat['name'] + '\n')

    with open(classname_txt, 'w') as f:
        f.writelines(class_txt)
        f.close()

    return classes

if __name__ == "__main__":
    main()