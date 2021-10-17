# -*- coding: utf-8 -*-
import numpy as np
import xml.etree.ElementTree as ET
import os
import scipy
from PIL import Image, ImageDraw,ImageFont


classes = ('airplane','ship','storagetank','bridge','dam','vehicle','trainstation',
           'baseballfield','basketballcourt','chimney','expressway-service-area',
                         'expressway-toll-station','airport','harbor','golffield','groundtrackfield',
                         'stadium','tenniscourt',
                         'windmill','overpass')
classes_cn = ("飞机", "船只", "油罐", "桥梁", "水坝", "车辆", "火车站",
              "棒球场", "篮球场", "烟囱", "公路服务区", "公路收费站", "飞机场", "港口",
              "高尔夫球场", "田径场", "体育场", "网球场", "风车房", "立交桥" )

num_classes = len(classes)
class_to_ind = dict(zip(classes, range(num_classes)))
class_to_cn = dict(zip(classes, classes_cn))



def load_pascal_annotation(filename):
    """
    Load image and bounding boxes info from XML file in the PASCAL VOC
    format.
    """
    tree = ET.parse(filename)
    img_file = img_path + filename.split('.')[0].split('/')[-1] + ".jpg"
    objs = tree.findall('object')
    # if not self.config['use_diff']:
    #     # Exclude the samples labeled as difficult
    #     non_diff_objs = [
    #         obj for obj in objs if int(obj.find('difficult').text) == 0]
    #     # if len(non_diff_objs) != len(objs):
    #     #     print 'Removed {} difficult objects'.format(
    #     #         len(objs) - len(non_diff_objs))
    #     objs = non_diff_objs
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)
    gt_classes = np.zeros((num_objs), dtype=np.int32)
    # "Seg" area for pascal is just the box area
    ishards = np.zeros((num_objs), dtype=np.int32)

    # Load object bounding boxes into a data frame.
    wh = tree.find('size')
    w, h = int(wh.find('width').text), int(wh.find('height').text)
    img_info[img_file] = []
    for ix, obj in enumerate(objs):
        bbox = obj.find('bndbox')
        # Make pixel indexes 0-based
        x1 = float(bbox.find('xmin').text)
        y1 = float(bbox.find('ymin').text)
        x2 = float(bbox.find('xmax').text)
        y2 = float(bbox.find('ymax').text)
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, w)
        y2 = min(y2, h)

        cls = class_to_cn[obj.find('name').text.lower().strip()]
        boxes_temp= (x1, y1, x2, y2)
        class_info[cls].append({
            "img_path": img_file,
            "bbox": boxes_temp
        })

        img_info[img_file].append({
            "gt_class": cls,
            "bbox": boxes_temp
        })


if __name__ == '__main__':
    global dior_path
    dior_path = '/media/2T/lizeng/data/DIOR/'
    anno_path = dior_path + 'Annotations/'
    global img_path
    img_path = dior_path + 'JPEGImages/'

    anno_file_list = os.listdir(anno_path)
    global class_info
    class_info = {}
    global img_info
    img_info = {}

    for class_name in classes_cn:
        class_info[class_name] = []

    for file_name in anno_file_list:
        file = anno_path + file_name
        load_pascal_annotation(file)

    image = np.zeros((800 * 20 + 1600 , 800 * 20 + 1600, 3)).astype(np.uint8)  #横轴方向两边空200
    image[:, :, :] = 255


    class_num = 0
    for class_name in classes_cn:
        infoes = class_info[class_name]
        count = 0
        for cls_info in infoes:
            if count == 20: break
            img_path = cls_info["img_path"]
            for img_info_ in img_info[img_path]:
                img = Image.open(img_path)
                draw = ImageDraw.Draw(img)
                if img_info_["gt_class"] == class_name:
                    draw.rectangle(img_info_["bbox"], outline='green')

            image[800 + class_num * 800 : 800 + (class_num + 1) * 800,
                  800 + count * 800 : 800 + (count + 1) * 800,
                 :] = np.array(img)
            count += 1
        class_num += 1
    image = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('Hiragino Sans GB.ttc', 80)
    color = (0, 0, 0)

    class_num = 0
    for class_name in classes_cn:
        position = (850 + class_num * 800, 100)
        draw.text(position, class_num, font=font, fill=color)
        infoes = class_info[class_name]
        num = str(len(infoes))
        position = (850 + class_num * 800, 800 * 21 + 100)
        draw.text(position, num, font=font, fill=color)
        class_num += 1
    image.save("/home/zengli/HKRM/data/visual_dior.jpg")
