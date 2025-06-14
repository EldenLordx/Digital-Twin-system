# -*- coding: utf-8 -*-
# coding=utf-8
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn import ensemble
from sklearn.model_selection._search import GridSearchCV

import numpy as np
import math
import os
import csv

def model_modify(camera_num):
    datapath = "C:/Users/icbc/Desktop/dwsjj/{cameraNum}.csv". \
        format(cameraNum=camera_num)
    content = csv2numpy(datapath)

    path_log_file = "C:/Users/icbc/Desktop/logging/modify_{cameraNum}.txt". \
        format(cameraNum=camera_num)
    if os.path.exists(path_log_file):
        os.remove(path_log_file)
    with open(path_log_file, 'a+') as f:
        temp_str = '==================== ' + str(camera_num) + ' ===================='
        print(temp_str)
        f.write(temp_str)

        train_pixel = content[:, 0:2]
        train_camera = content[:, -2:]

        train_camera_x = train_camera[:, 0]
        train_camera_y = train_camera[:, 1]

        param_test = {'n_estimators': range(20, 150, 10), 'max_depth': range(1, 10, 1)}
        # 原始数据
        lr_model_x = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(),
                                  param_grid=param_test, iid=False, cv=5)
        lr_model_y = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(),
                                  param_grid=param_test, iid=False, cv=5)

        lr_model_x.fit(train_pixel, train_camera_x)
        lr_model_y.fit(train_pixel, train_camera_y)

        temp_str = '-----' + 'x' + '-----'
        print(temp_str)
        f.write(temp_str + '\n')
        temp_str = '网格搜索-度量记录：' + '\n' + str(lr_model_x.cv_results_)  # 包含每次训练的相关信息
        print(temp_str)
        f.write(temp_str + '\n')
        temp_str = '网格搜索-最佳度量值:' + '\n' + str(lr_model_x.best_score_)  # 获取最佳度量值
        print(temp_str)
        f.write(temp_str + '\n')
        temp_str = '网格搜索-最佳参数：' + '\n' + str(lr_model_x.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
        print(temp_str)
        f.write(temp_str + '\n')
        temp_str = '网格搜索-最佳模型：' + '\n' + str(lr_model_x.best_estimator_)  # 获取最佳度量时的分类器模型
        print(temp_str)
        f.write(temp_str + '\n')

        temp_str = '-----' + 'y' + '-----'
        print(temp_str)
        f.write(temp_str + '\n')
        temp_str = '网格搜索-度量记录：' + '\n' + str(lr_model_y.cv_results_)  # 包含每次训练的相关信息
        print(temp_str)
        f.write(temp_str + '\n')
        temp_str = '网格搜索-最佳度量值:' + '\n' + str(lr_model_y.best_score_)  # 获取最佳度量值
        print(temp_str)
        f.write(temp_str + '\n')
        temp_str = '网格搜索-最佳参数：' + '\n' + str(lr_model_y.best_params_)  # 获取最佳度量值时的代定参数的值。是一个字典
        print(temp_str)
        f.write(temp_str + '\n')
        temp_str = '网格搜索-最佳模型：' + '\n' + str(lr_model_y.best_estimator_)  # 获取最佳度量时的分类器模型
        print(temp_str)
        f.write(temp_str + '\n')

def csv2numpy(path_csv):
    # 从文本文件中提取数据并转为numpy Array格式
    csv_file = csv.reader(open(path_csv, encoding='ISO-8859-1'))
    content = []  # 用来存储整个文件的数据，存成一个列表，列表的每一个元素又是一个列表，表示的是文件的某一行
    for index, list in enumerate(csv_file):
        if index == 0 or list[0] == '':
            continue
        list_item = []
        for index_sub, item in enumerate(list):
            list_item.append(int(item))
        content.append(list_item)
    content = np.array(content)
    return content

if __name__ == '__main__':
    camera_list = [18, 20, 24, 25, 27, 28, 36]
    for camera_num in camera_list:
        model_modify(camera_num)
    print('Done')
