#-*- coding: utf-8 -*-
import os
import cv2
import numpy as np 
import copy
import xml.etree.ElementTree as ET 


class Load_data(object):

    def __init__(self, path, batch, CLASS):
        self.devkil_path = path + '/VOCdevkit'
        self.data_path = self.devkil_path + '/VOC2007'
        self.img_size = 448       # 网络的输入统一是448x448x3
        self.batch = batch
        self.CLASS = CLASS
        self.n_class = len(CLASS)
        self.class_id = dict(zip(CLASS, range(self.n_class)))
        self.id = 0
        self.run_this()

    def load_img(self, path):
        """
        提取path对应的图片，并且调整图片的尺寸
        """
        img = cv2.imread(path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.multiply(1.0/255, img)
        return img

    def load_xml(self, index):
        """
        提取每张图片对应的xml文件中的目标信息

        Args:
            index: xml文件对应的索引，也即文件名，和图片名对应
        """
        path = self.data_path + '/JPEGImages/' + index + '.jpg'
        print(path)
        xml_path = self.data_path + '/Annotations/' + index + '.xml'
        img = cv2.imread(path)
        w_scale = self.img_size / img.shape[1]
        h_scale = self.img_size / img.shape[0]
        
        _label = np.zeros((7, 7, 25))    # 网络的输出结果：7x7个网格，每个网格1个bbox，每个bbox只能对一个类别负责，1*5+20(数据集的类别数) = 25
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        for obj in objs:
            box = obj.find('bndbox')
            x1 = min(max(float(box.find('xmin').text)*w_scale, 0), self.img_size - 1)
            y1 = min(max(float(box.find('ymin').text)*h_scale, 0), self.img_size - 1)
            x2 = min(max(float(box.find('xmax').text)*w_scale, 0), self.img_size - 1)
            y2 = min(max(float(box.find('ymin').text)*h_scale, 0), self.img_size - 1)

            boxes = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]       # bounding box用中点坐标和宽高表示
            cls_id = self.class_id[obj.find('name').text.lower().strip()]  # 类别的id
            x_id = int(boxes[0] * 7 / self.img_size)     # 当前bbox位于7x7网格中的x索引
            y_id = int(boxes[1] * 7 / self.img_size)     # 当前bbox位于7x7网格中的y索引

            if _label[y_id, x_id, 0] == 1:         # 每个bbox只能有一个类别，因此如果该点的值已经为1，就不能在包含别的目标了
                continue

            _label[y_id, x_id, 0] = 1
            _label[y_id, x_id, 1:5] = boxes 
            _label[y_id, x_id, 5 + cls_id] = 1    # 将对应的类别位置置为1

        return _label, len(objs)

    def load_label(self):
        """
        加载所有的label，label位于'/ImageSets/Main/trianval.txt'文件中
        """
        path = self.data_path + '/ImageSets/Main/trainval.txt'
        with open(path, 'r') as fr:
            indexes = [x.strip() for x in fr.readlines()]
        labels = []
        for index in indexes:
            _label, num = self.load_xml(index)
            if num == 0:
                continue
            img_name = self.data_path + '/JPEGImages/' + index + '.jpg'
            labels.append({'img_name': img_name,
                           'label': _label})
        return labels

    def run_this(self):
        labels = self.load_label()
        np.random.shuffle(labels)
        self.truth_labels = labels
        return labels

    def get_data(self):
        batch_imgs = np.zeros((self.batch, self.img_size, self.img_size, 3))
        batch_labels = np.zeros((self.batch, 7, 7, 25))
        times = 0
        while times < self.batch:
            img_name = self.truth_labels[self.id]['img_name']
            batch_imgs[times, :, :, :] = self.load_img(img_name)
            batch_labels[times, :, :, :] = self.truth_labels[self.id]['label']
            self.id += 1
            times += 1
        return batch_imgs, batch_labels


if __name__ == '__main__':
    CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
               'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor']
    data_path = 'F:/program/datasets/VOCtrainval_06-Nov-2007/'
    load_data = Load_data(data_path, 10, CLASSES)
    batch_imgs, batch_labels = load_data.get_data()