#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- 训练模型函数 -- #

import json
import os
import pathlib

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
AUTOTUNE = tf.data.experimental.AUTOTUNE  # 管道参数
BATCH_SIZE = 32  # 一个训练过程批次中的数据大小


# 图像处理
def images_handle(path):
    images = tf.io.read_file(path)
    images = tf.image.decode_jpeg(images, channels=3)
    images = tf.image.resize(images, [230, 230])
    images /= 255.0

    return images


# 数据归一化
def change_range(image, label):
    return (image * 2) - 1, label


# 数据集扩增
def augment_data(image, label):
    # 将图片随机进行水平翻转
    image = tf.image.random_flip_left_right(image)
    # 将图片随机进行垂直翻转
    image = tf.image.random_flip_up_down(image)
    # 随机设置图片的对比度
    image = tf.image.random_contrast(image, lower=0.0, upper=1.0)
    # 随机设置图片的亮度
    image = tf.image.random_brightness(image, max_delta=0.5)
    # 随机设置图片的饱和度
    image = tf.image.random_saturation(image, lower=0.3, upper=0.8)

    return image, label


# 自定义模型
def get_model(mobile_net, labels_data):
    model = tf.keras.models.Sequential([
        mobile_net,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(len(labels_data), activation='softmax')

    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=["accuracy"])

    # 打印出模型的概况信息
    model.summary()

    return model


def training():
    print("--> tensorflow version " + tf.__version__)

    # -- 读取已预分类的数据位置 -- #

    print("--> reading datasheet...", end='')

    images_paths = list(pathlib.Path("train_img_handle/4.classify").glob("*/*"))

    # 读取标签数据
    with open("data_file/label_num_to_disease_map.json", "r") as labels_file:
        labels_data = json.load(labels_file)

    # 将路径保存为str
    images_paths = [str(path) for path in images_paths]

    # 映射标签和对应的类型
    labels_index = dict((name, index) for index, name in enumerate(labels_data.values()))
    images_labels = [labels_index[pathlib.Path(path).parent.name] for path in images_paths]

    print("done")

    # -- 构建tensorflow数据集管理 -- #

    print("--> creating tensorflow datasheet...", end='')

    # 构建图片路径数据集
    images_path_ds = tf.data.Dataset.from_tensor_slices(images_paths)
    # 构建图片标签数据集
    images_labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(images_labels, tf.int8))
    # 构建图片数据集
    images_ds = images_path_ds.map(images_handle, num_parallel_calls=AUTOTUNE)

    # 对图片和标签打包作为训练集
    train_ds = tf.data.Dataset.zip((images_ds, images_labels_ds))
    print("done")

    # -- 导入/定义训练模型 -- #

    print("--> preparing training model(MobileNetV2)...")
    # 打乱数据集
    image_count = len(images_paths)
    train_ds = train_ds.shuffle(buffer_size=image_count)
    # 让数据集重复多次
    train_ds = train_ds.repeat()
    # 设置每个batch的大小
    train_ds = train_ds.batch(BATCH_SIZE)
    train = train_ds.prefetch(buffer_size=AUTOTUNE)
    train = train.map(augment_data)
    train = train.map(change_range)

    # 使用预设权重的模型
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(230, 230, 3),
                                                   include_top=False,
                                                   weights='imagenet')
    mobile_net.trainable = False

    # 获取定义后的迁移模型进行训练
    model = get_model(mobile_net, labels_data)
    train_res = model.fit(train_ds,
                          epochs=3,
                          steps_per_epoch=25,
                          )

    # test

    # 保存模型
    print("--> saving output model...")
    model.save("model_output.h5")


    return train_res
