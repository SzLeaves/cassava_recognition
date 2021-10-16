#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- 启动程序 -- #

import csv
import pathlib
import shutil
import time

from modules import classify
from modules import evaluate
from modules import model
from modules import noise_handle

if __name__ == "__main__":
    print("-- START --")

    # -- 导入原始图片数据路径 -- #
    images_path = [str(path) for path in list(pathlib.Path("train_img").glob("*.jpg"))]

    # -- 裁剪图片尺寸 -- #

    print("--> resize images data...", end='')
    t = time.time()

    re_path = "train_img_handle/1.images_resize/"
    pathlib.Path(re_path).mkdir(parents=True, exist_ok=True)
    for index in images_path:
        noise_handle.resize_images(index, re_path)

    print('done', end=' -- ')
    print(f'time: {time.time() - t:.2f}s')

    # -- 处理模糊图片 -- #

    print("--> start blur images regonition...", end='')
    t = time.time()

    # 重新载入已裁剪的图片路径
    images_path = [str(path) for path in list(pathlib.Path(re_path).glob("*.jpg"))]
    images_path = sorted(images_path)

    new_folder = "train_img_handle/2.blur_recognition/"
    new_folder_blur = new_folder + "blur/"
    new_folder_clear = new_folder + "clear/"

    pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(new_folder_blur).mkdir(parents=True, exist_ok=True)
    pathlib.Path(new_folder_clear).mkdir(parents=True, exist_ok=True)

    # 木薯叶图片筛选阈值
    cas_th_lower = 1700
    cas_th_upper = 3500
    # 苹果叶图片筛选阈值
    al_th_lower = 100
    al_th_upper = 125

    for save_image in images_path:
        res = noise_handle.lapulase(save_image)
        if cas_th_lower < res < cas_th_upper or al_th_lower < res < al_th_upper:

            shutil.copy(save_image, new_folder_clear)
        else:
            shutil.copy(save_image, new_folder_blur)

    print('done', end=' -- ')
    print(f'time: {time.time() - t:.2f}s')

    # -- 处理非叶片图片 -- #

    print("--> start leaves regonition...", end='')
    t = time.time()

    new_folder = "train_img_handle/3.leaves_recognition/"
    new_folder_leaves = new_folder + "leaves/"
    new_folder_other = new_folder + "other/"

    pathlib.Path(new_folder).mkdir(parents=True, exist_ok=True)
    pathlib.Path(new_folder_leaves).mkdir(parents=True, exist_ok=True)
    pathlib.Path(new_folder_other).mkdir(parents=True, exist_ok=True)

    clear_images_path = [str(path) for path in pathlib.Path(new_folder_clear).glob("*.jpg")]
    for handle_images in clear_images_path:
        # 通过识别图片主体色调判断是否为叶片
        if noise_handle.green_filter(handle_images):
            shutil.copy(handle_images, new_folder_leaves)
        else:
            shutil.copy(handle_images, new_folder_other)

    print('done', end=' -- ')
    print(f'time: {time.time() - t:.2f}s')

    # -- 图片分类 -- #

    print("--> start images classify...", end='')
    t = time.time()

    # 重新生成已经筛选好图片的数据表文件
    source_label = dict()
    with open("data_file/train.csv", "r") as csv_file:
        info = csv.reader(csv_file)
        info.__next__()  # 跳过第一行标记
        info = sorted(info)

        for index in info:
            source_label[index[0]] = index[1]

    final_path = [str(path) for path in
                  list(pathlib.Path("train_img_handle/3.leaves_recognition/leaves").glob("*.jpg"))]
    final_path = sorted(final_path)
    label = dict()
    for path in final_path:
        name = path[45:]
        for src_label in source_label.items():
            if name == src_label[0]:
                label[name] = src_label[1]

    with open("data_file/train_clear.csv", "w") as csv_file:
        for index in label.items():
            csv_file.write(index[0] + ',' + index[1] + '\n')

    # 对预处理好的图片分类
    classify.images_classify("train_img_handle/3.leaves_recognition/leaves/",
                             "data_file/label_num_to_disease_map.json",
                             "data_file/train_clear.csv",
                             "train_img_handle/4.classify/")

    print('done', end=' -- ')
    print(f'time: {time.time() - t:.2f}s')

    # -- 模型训练 -- #

    print("--> training model...", end='')
    t = time.time()

    model.training()

    print('\ndone', end=' -- ')
    print(f'time: {time.time() - t:.2f}s')

    # -- 利用已训练模型进行数据预测 -- #

    print("--> predict data...", end='')
    t = time.time()

    evaluate.predict_data("model_output.h5")

    print('\ndone', end=' -- ')
    print(f'time: {time.time() - t:.2f}s')

    print('-- END --')
