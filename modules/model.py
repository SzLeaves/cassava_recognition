#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- 训练模型函数 -- #

import json
import os
import pathlib

import matplotlib.pyplot as plt
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


# 绘制训练集与测试集的loss/accuary
def plot_train_res(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)

    # 绘制准确率曲线
    plt.title('Training and Validation Accuracy')
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')

    # 绘制损失率曲线
    plt.title('Training and Validation Loss')
    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')

    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel('Cross Entropy')

    # 保存图片并显示
    plt.savefig("acc_loss.png")
    plt.show()


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

    # 从数据集划分30%作为验证集
    train_size = len(images_paths)
    valid_size = int(train_size * 0.3)
    valid_ds = train_ds.skip(int(train_size * 0.7)).take(valid_size)
    valid_ds = valid_ds.batch(BATCH_SIZE)

    print("--> preparing training model(MobileNetV2)...")
    # 打乱数据集
    train_ds = train_ds.shuffle(buffer_size=train_size)
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
    train_res = model.fit(train_ds, epochs=250, steps_per_epoch=25,
                          validation_data=valid_ds, validation_steps=25)
    # 保存模型
    print("--> saving output model...")
    model.save("model_output.h5")

    # 绘制loss/accuracy曲线图
    plot_train_res(train_res)
