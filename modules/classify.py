#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- 分类函数 -- #

import csv
import json
import pathlib
import shutil


def images_classify(save_images, save_label, save_table):
    # 加载图像路径
    images_path = [str(path) for path in list(pathlib.Path(save_images).glob("*.jpg"))]
    images_path = sorted(images_path)  # 排序后一一对应

    # 加载标签文件
    with open(save_label, "r") as label_file:
        label_index = json.load(label_file)

    # 加载标记文件
    images_label = dict()
    with open(save_table, "r") as csv_file:
        info = csv.reader(csv_file)

        info = sorted(info)
        for index in info:
            images_label[index[0]] = index[1]

    # 创建分类文件夹
    new_folder = "train_img_handle/4.classify/"
    new_folder_path = dict()
    pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
    for ids, name in label_index.items():
        pathlib.Path(new_folder + name).mkdir(parents=True, exist_ok=True)
        new_folder_path[ids] = new_folder + name

    # 分类存放
    for path, label in zip(images_path, images_label.items()):
        shutil.copy(path, new_folder_path[label[1]])
