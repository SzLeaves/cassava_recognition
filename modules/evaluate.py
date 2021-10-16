#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import pathlib

import numpy as np
import tensorflow as tf

from modules.model import AUTOTUNE, BATCH_SIZE
from modules.model import images_handle


def load_data():
    # 加载测试图片路径
    images_paths = list(pathlib.Path("train_img_handle/4.classify").glob("*/*"))
    # 读取标签数据
    with open("data_file/label_num_to_disease_map.json", "r") as labels_file:
        labels_data = json.load(labels_file)
    # 将路径保存为str
    images_paths = [str(path) for path in images_paths]

    # 映射标签和对应的类型
    labels_index = dict((name, index) for index, name in enumerate(labels_data.values()))
    images_labels = [labels_index[pathlib.Path(path).parent.name] for path in images_paths]

    # 将评估与预测用数据转换为tf数据集
    images_path_ds = tf.data.Dataset.from_tensor_slices(images_paths)
    images_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(images_labels, tf.int8))
    images_ds = images_path_ds.map(images_handle, num_parallel_calls=AUTOTUNE)

    test_ds = tf.data.Dataset.zip((images_ds, images_labels_ds))
    test_ds = test_ds.shuffle(buffer_size=len(images_paths))
    test_ds = test_ds.batch(BATCH_SIZE)
    images_ds = images_ds.batch(BATCH_SIZE)

    return test_ds, images_ds, images_paths


def predict_data(save_model):
    model = tf.keras.models.load_model(save_model)
    test_ds, images_ds, images_paths = load_data()

    # 评估模型
    model.evaluate(test_ds)

    # 预测数据
    predict_text = ['file_name,probability,predict_label']
    pre_res = model.predict(images_ds)
    classes = np.argmax(pre_res, axis=1)
    for ids, name, probability in zip(classes, images_paths, pre_res):
        print('file_name:' + name[28:])
        print('probability: ' + str(probability))
        print('predict label: ' + str(ids))
        print('')

        predict_text.append("%s,%s,%s" % (name[28:], str(probability), str(ids)))
