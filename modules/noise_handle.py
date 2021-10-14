#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -- 数据清洗函数 -- #

import cv2
import numpy as np


# 颜色分量列表
def color_list():
    color_dict = dict()

    # 黑色
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 46])
    black = [lower_black, upper_black]
    color_dict['black'] = black

    # 白色
    lower_white = np.array([0, 0, 221])
    upper_white = np.array([180, 30, 255])
    white = [lower_white, upper_white]
    color_dict['white'] = white

    # 红色
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    red = [lower_red, upper_red]
    color_dict['red'] = red

    # 橙色
    lower_orange = np.array([11, 43, 46])
    upper_orange = np.array([25, 255, 255])
    orange = [lower_orange, upper_orange]
    color_dict['orange'] = orange

    # 黄色
    lower_yellow = np.array([26, 43, 46])
    upper_yellow = np.array([34, 255, 255])
    yellow = [lower_yellow, upper_yellow]
    color_dict['yellow'] = yellow

    # 绿色
    lower_green = np.array([35, 43, 46])
    upper_green = np.array([77, 255, 255])
    green = [lower_green, upper_green]
    color_dict['green'] = green

    # 蓝色
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    blue = [lower_blue, upper_blue]
    color_dict['blue'] = blue

    return color_dict


# 通道分量识别
def color_filter(path):
    maxsum = -100
    res = None

    image_path = cv2.imread(path)
    color_dict = color_list()

    # 建立hsv颜色模型
    hsv = cv2.cvtColor(image_path, cv2.COLOR_BGR2HSV)

    for color_name in color_dict:
        # 二值化和膨胀处理
        mask = cv2.inRange(hsv, color_dict[color_name][0], color_dict[color_name][1])
        binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        binary = cv2.dilate(binary, None, iterations=2)

        # 计算轮廓边缘之和
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        area_sum = 0
        for index in contours:
            area_sum += cv2.contourArea(index)
        if area_sum > maxsum:
            maxsum = area_sum
            res = color_name

    return res


# 非叶片筛选，判断主体颜色是否为绿色
def green_filter(path):
    res = color_filter(path)
    if res == 'green':
        return True
    else:
        return False


# Laplace梯度法，处理噪声图像
def lapulase(path):
    frame = cv2.imread(path)  # 读取图像
    reImg = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)
    img2gray = cv2.cvtColor(reImg, cv2.COLOR_BGR2GRAY)  # 将图片压缩为单通道的灰度图

    res = cv2.Laplacian(img2gray, cv2.CV_64F)
    score = res.var()

    return score


# 改变图片尺寸并保存
def resize_images(src_path, path):
    img = cv2.imread(src_path)
    src_size = (img.shape[1], img.shape[0])
    size = (800, 600)

    if src_size != size:
        shrink = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        crop_img = shrink[50:550, 210:710]
        cv2.imwrite(path + src_path[10:], crop_img)
    else:
        crop_img = img[50:550, 210:710]
        cv2.imwrite(path + src_path[10:], crop_img)
