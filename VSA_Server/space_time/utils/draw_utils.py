from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import os


def draw_dataset_map(name, list_train=0, list_test=0):
    plt.title(u'{map_name}'. \
              format(map_name=name), y=0)

    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部
    ax.yaxis.set_ticks_position('left')  # 将y轴的位置设置在右边
    ax.invert_yaxis()  # y轴反向
    ax.set_xlim(left=0, right=1280)
    ax.set_ylim(bottom=720, top=0)

    if list_train != 0:
        for train in list_train:
            x = train[0]
            y = train[1]
            plt.scatter(x, y, c="#436EEE", marker=".")
    if list_test != 0:
        for test in list_test:
            x = test[0]
            y = test[1]
            plt.scatter(x, y, c="#CD3700", marker=".")
    plt.show()


def draw_error_map(name, list_pixel, list_pred, list_true):
    plt.title(u'{map_name}'. \
              format(map_name=name), y=0)

    ax = plt.gca()
    ax.xaxis.set_ticks_position('top')  # 将x轴的位置设置在顶部
    ax.yaxis.set_ticks_position('left')  # 将y轴的位置设置在右边
    ax.invert_yaxis()  # y轴反向
    ax.set_xlim(left=0, right=1280)
    ax.set_ylim(bottom=720, top=0)

    for i in range(0, len(list_pred)):
        x_pred = list_pred[i][0]
        y_pred = list_pred[i][1]
        x_true = list_true[i][0]
        y_true = list_true[i][1]
        error = (abs(x_pred - x_true) + abs(y_pred - y_true)) / 2

        color = '#CD3700'
        green = '#02FE20'
        yellow = '#FFFF37'
        red = '#FE0202'

        if error <= 100:
            color = green
        elif error <= 200:
            color = yellow
        else:
            color = red
        plt.scatter(list_pixel[i][0], list_pixel[i][1], c=color, marker=".")
    plt.show()


if __name__ == '__main__':
    list_train = [[10, 20], [50, 70]]
    list_test = [[30, 20], [130, 70]]
    draw_dataset_map(u'test_hehe', list_train, list_test)
