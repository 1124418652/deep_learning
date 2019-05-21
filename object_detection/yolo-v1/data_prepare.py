#-*- coding: utf-8 -*-
import os
import cv2
import numpy as np 
import copy
import xml.etree.ElementTree as ET 


class load_data(object):

    def __init__(self, path, batch, CLASS):
        self.devkil_path = path + '/VOCdevkit'
        self.data_path = self.devkil_path + '/VOC2007'
        self.img_size = 448
        self.batch = batch
        self.CLASS = CLASS
        self.n_class = len(CLASS)
        self.class_id = dict(zip(CLASS, range(self.n_class)))

    def load_img(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.multiply(1.0/255, img)
        return img

    def load_xml(self, index):
        path = self.data_path + '/JPEGImages' + index + '.jpg'
        xml_path = self.data_path + '/Annotations/' + index + '.xml'
        img = cv2.imread(path)
        width = self.img_size / img.shape[1]
        height = self.img_size / img.shape[0]
        
        label = np.zeros((7, 7, 30))
        tree = ET.parse(xml_path)
        objs = tree.findall('object')
        for obj in objs:
            box = obj.find('bndbox')
            x1 = max(float(box.find('xmin').text), 0)
            y1 = max(float(box.find('ymin').text), 0)
            x2 = max(float(box.find('xmax').text), 0)
            y2 = max(float(box.find('ymin').text), 0)

