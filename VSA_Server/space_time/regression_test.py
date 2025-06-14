# coding=utf-8
from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing

from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
from sklearn.tree import ExtraTreeRegressor

import matplotlib.pyplot as pl
import sklearn.metrics as sm
import numpy as np
import csv
import math
import os
import time


# 1 - 决策树回归
# 2 - 线性回归（多项式）
# 3 - SVM回归 **
# 4 - KNN回归 *
# 5 - 随机森林回归 *
# 6 - Adaboost回归 *
# 7 - GBDT回归 ****
# 8 - Bagging回归 **
# 9 - ExtraTree极端随机数回归
def model_generator(camera_num, method=0, flag_save_model=False):
    datapath = "C:/Users/icbc/Desktop/dwsjj/{cameraNum}.csv". \
        format(cameraNum=camera_num)
    content = csv2numpy(datapath)

    path_log_file = "C:/Users/icbc/Desktop/logging_2/modify_{cameraNum}.txt". \
        format(cameraNum=camera_num, method_num=method)
    if os.path.exists(path_log_file):
        os.remove(path_log_file)
    with open(path_log_file, 'a+') as f:

        target_echo_x = 0
        target_echo_y = 0
        echo_error_x = 99999
        echo_error_y = 99999

        target_degree_x = 0
        target_degree_y = 0
        min_degree = 1
        max_degree = 13
        list_degree_error_x = [99999] * (max_degree + 1)
        list_degree_error_y = [99999] * (max_degree + 1)
        for degree in range(min_degree, max_degree):
            quadratic_featurizer = PolynomialFeatures(degree=degree)
            degree_error_x = 0
            degree_error_y = 0
            degree_error_sum = 0
            temp_str = '=========================' + ' Degree is ' + str(degree) + ' ========================='
            f.write(temp_str + '\n')
            min_echo = 0
            max_echo = 11
            for echo in range(0, 11):
                temp_str = "-- " + " echo " + str(echo) + " -- " + " degree " + str(degree) + " --"
                f.write(temp_str + '\n')
                train_pixel, test_pixel, train_camera, test_camera = train_test_split(content[:, 0:2], content[:, -2:],
                                                                                      test_size=0.10,
                                                                                      random_state=echo)

                train_camera_x = train_camera[:, 0]
                train_camera_y = train_camera[:, 1]
                test_camera_x = test_camera[:, 0]
                test_camera_y = test_camera[:, 1]

                temp_echo_error_x = 99999
                temp_echo_error_y = 99999

                # 原始数据(多项式扩张)
                origin_train_pixel = quadratic_featurizer.fit_transform(train_pixel)  # x,y
                # origin_train_pixel = quadratic_featurizer.fit_transform(train_pixel[:, ::-1])  # y,x

                origin_test_pixel = quadratic_featurizer.fit_transform(test_pixel)  # x,y
                # origin_test_pixel = quadratic_featurizer.fit_transform(test_pixel[:, ::-1])  # y,x

                # 归一化(归一化后多项式扩张)
                # std_train = preprocessing.scale(train_pixel)
                # std_train_pixel = quadratic_featurizer.fit_transform(std_train)  # x,y
                # std_train_pixel_reversed = quadratic_featurizer.fit_transform(std_train[:, ::-1])  # y,x
                #
                # std_test = preprocessing.scale(test_pixel)
                # std_test_pixel = quadratic_featurizer.fit_transform(std_test)  # x,y
                # std_test_pixel_reversed = quadratic_featurizer.fit_transform(std_test[:, ::-1])  # y,x

                # 1 - 决策树回归
                # 2 - 线性回归（多项式）
                # 3 - SVM回归 **
                # 4 - KNN回归 *
                # 5 - 随机森林回归 *
                # 6 - Adaboost回归 *
                # 7 - GBDT回归 ****
                # 8 - Bagging回归 **
                # 9 - ExtraTree极端随机数回归
                lr_model_x, lr_model_y = return_model(method)
                lr_model_x.fit(origin_train_pixel, train_camera_x)
                lr_model_y.fit(origin_train_pixel, train_camera_y)

                MAE_train_x = calculate_metrics(train_pixel, train_camera, origin_train_pixel, lr_model_x, 0)
                MAE_train_y = calculate_metrics(train_pixel, train_camera, origin_train_pixel, lr_model_y, 1)
                temp_str = "MAE_train_x: " + str(MAE_train_x / 10.25) + ' cm'
                f.write(temp_str + '\n')
                temp_str = "MAE_train_y: " + str(MAE_train_y / 10.25) + ' cm'
                f.write(temp_str + '\n')

                MAE_test_x = calculate_metrics(test_pixel, test_camera, origin_test_pixel, lr_model_x, 0)
                MAE_test_y = calculate_metrics(test_pixel, test_camera, origin_test_pixel, lr_model_y, 1)
                temp_str = "MAE_test_x: " + str(MAE_test_x / 10.25) + ' cm'
                f.write(temp_str + '\n')
                temp_str = "MAE_test_y: " + str(MAE_test_y / 10.25) + ' cm'
                f.write(temp_str + '\n')

                temp_res_x = MAE_train_x + MAE_test_x
                temp_res_y = MAE_train_y + MAE_test_y

                # degree_error_x += temp_res_x
                # degree_error_y += temp_res_y

                degree_error_x += MAE_test_x
                degree_error_y += MAE_test_y

                temp_echo_error_x = min(temp_echo_error_x, temp_res_x)
                temp_echo_error_y = min(temp_echo_error_y, temp_res_y)

                if temp_echo_error_x < echo_error_x:
                    target_echo_x = echo
                    echo_error_x = temp_echo_error_x

                if temp_echo_error_y < echo_error_y:
                    target_echo_y = echo
                    echo_error_y = temp_echo_error_y

            list_degree_error_x[degree] = degree_error_x / (max_echo - min_echo)
            list_degree_error_y[degree] = degree_error_y / (max_echo - min_echo)
            temp_str = '===== ' + 'Degree metrics: ' + str(degree) + " ===="
            f.write(temp_str + '\n')
            temp_str = "Degree error_x is " + str(list_degree_error_x[degree] / 10.25) + ' cm'
            f.write(temp_str + '\n')
            temp_str = "Degree error_y is " + str(list_degree_error_y[degree] / 10.25) + ' cm'
            f.write(temp_str + '\n')

        target_degree_x = list_degree_error_x.index(min(list_degree_error_x))
        target_degree_y = list_degree_error_y.index(min(list_degree_error_y))

        temp_str = "================================================================="
        f.write(temp_str + '\n')
        temp_str = str(list_degree_error_x)
        f.write(temp_str + '\n')
        temp_str = str(list_degree_error_y)
        f.write(temp_str + '\n')
        print('=========================' + " Camera " + str(camera_num) + ' =========================')
        temp_str = "Min echo error_x " + str(echo_error_x / 10.25) + ' cm'
        f.write(temp_str + '\n')
        print(temp_str)
        temp_str = "Min echo error_y " + str(echo_error_y / 10.25) + ' cm'
        f.write(temp_str + '\n')
        print(temp_str)
        temp_str = "Degree average error_x is " + str(list_degree_error_x[target_degree_x] / 10.25) + ' cm'
        f.write(temp_str + '\n')
        print(temp_str)
        temp_str = "Degree average error_y is " + str(list_degree_error_y[target_degree_y] / 10.25) + ' cm'
        f.write(temp_str + '\n')
        print(temp_str)
        temp_str = "Camera " + str(camera_num)
        f.write(temp_str + '\n')
        print(temp_str)
        # temp_str = "Target echo_x is " + str(target_echo_x)
        # f.write(temp_str + '\n')
        # temp_str = "Target echo_y is " + str(target_echo_y)
        # f.write(temp_str + '\n')
        temp_str = "Target degree_x is " + str(target_degree_x)
        f.write(temp_str + '\n')
        print(temp_str)
        temp_str = "Target degree_y is " + str(target_degree_y)
        f.write(temp_str + '\n')
        print(temp_str)

        if flag_save_model:
            quadratic_featurizer_x = PolynomialFeatures(degree=target_degree_x)
            quadratic_featurizer_y = PolynomialFeatures(degree=target_degree_y)

            origin_all_pixel = content[:, 0:2]
            origin_all_pixel_x = quadratic_featurizer_x.fit_transform(origin_all_pixel)  # x,y
            # origin_all_pixel__x = quadratic_featurizer_x.fit_transform(origin_all_pixel[:, ::-1])  # y,x
            origin_all_pixel_y = quadratic_featurizer_y.fit_transform(origin_all_pixel)  # x,y

            origin_all_camera = content[:, -2:]
            origin_all_camera_x = origin_all_camera[:, 0]
            origin_all_camera_y = origin_all_camera[:, 1]

            # 归一化
            # std_content = preprocessing.scale(content)
            # std_train_pixel = std_content[:, 0:2]
            # std_train_pixel = quadratic_featurizer_x.fit_transform(std_train_pixel)  # x,y
            # std_train_pixel_reversed = quadratic_featurizer_x.fit_transform(std_train_pixel[:, ::-1])  # y,x

            lr_model_x, lr_model_y = return_model(method)

            lr_model_x.fit(origin_all_pixel_x, origin_all_camera_x)
            lr_model_y.fit(origin_all_pixel_y, origin_all_camera_y)

            path_save_model = '/home/syx/workspace/pixel_2_map/model'
            save_model(camera_num, path_save_model, lr_model_x, 0)
            save_model(camera_num, path_save_model, lr_model_y, 1)


def return_model(method):
    lr_model_x = LinearRegression()
    lr_model_y = LinearRegression()
    if method == 1:
        lr_model_x = tree.DecisionTreeRegressor()
        lr_model_y = tree.DecisionTreeRegressor()
    elif method == 2:
        lr_model_x = LinearRegression()
        lr_model_y = LinearRegression()
    elif method == 3:
        lr_model_x = svm.SVR()
        lr_model_y = svm.SVR()
    elif method == 4:
        lr_model_x = neighbors.KNeighborsRegressor()
        lr_model_y = neighbors.KNeighborsRegressor()
    elif method == 5:
        lr_model_x = ensemble.RandomForestRegressor(n_estimators=20)
        lr_model_y = ensemble.RandomForestRegressor(n_estimators=20)
    elif method == 6:
        lr_model_x = ensemble.AdaBoostRegressor(n_estimators=50)
        lr_model_y = ensemble.AdaBoostRegressor(n_estimators=50)
    elif method == 7:
        # 'ls', 'lad', 'huber', 'quantile'
        lr_model_x = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, loss='ls', max_depth=2)
        lr_model_y = ensemble.GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, loss='ls', max_depth=2)
    elif method == 8:
        lr_model_x = ensemble.BaggingRegressor()
        lr_model_y = ensemble.BaggingRegressor()
    elif method == 9:
        lr_model_x = ExtraTreeRegressor()
        lr_model_y = ExtraTreeRegressor()
    return lr_model_x, lr_model_y


def save_model(camera_num, path_save_model, lr_model, flag_x_y):
    if flag_x_y == 0:
        path_save_model = os.path.join(path_save_model, "{cameraNum}_x.model". \
                                       format(cameraNum=camera_num))
    elif flag_x_y == 1:
        path_save_model = os.path.join(path_save_model, "{cameraNum}_y.model". \
                                       format(cameraNum=camera_num))

    if os.path.exists(path_save_model):
        os.remove(path_save_model)
        print(path_save_model, 'has been removed')
    joblib.dump(lr_model, path_save_model)
    print('model for {cameraNum} has been saved to'. \
          format(cameraNum=camera_num), path_save_model)


def calculate_metrics(test_pixel, test_camera, test_pixel_new, lr_model, flag_x_y):
    list_pixel = []
    list_pred = []
    list_true = []
    for i in range(0, len(test_pixel_new)):
        x_y_material = np.array(test_pixel_new[i]).reshape(1, -1)
        x_y_Predict = lr_model.predict(x_y_material)[0]
        temp_pred = int(x_y_Predict)
        list_pred.append(temp_pred)

        temp_pixel = [test_pixel[i][0], test_pixel[i][1]]
        list_pixel.append(temp_pixel)

        temp_true = [test_camera[i][0], test_camera[i][1]]
        list_true.append(temp_true)

    list_pixel_x = np.array(list_pixel)[:, 0].tolist()
    list_pixel_y = np.array(list_pixel)[:, 1].tolist()
    list_true_x = np.array(list_true)[:, 0].tolist()
    list_true_y = np.array(list_true)[:, 1].tolist()

    MAE_test = 0.0
    if flag_x_y == 0:
        MAE_test = sm.mean_absolute_error(list_true_x, list_pred)
    elif flag_x_y == 1:
        MAE_test = sm.mean_absolute_error(list_true_y, list_pred)
    return MAE_test

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
    camera_list = [17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 35, 36]
    # camera_list = [18]
    for camera_num in camera_list:
        method = 7
        # flag_save_model = True
        flag_save_model = False
        model_generator(camera_num, method, flag_save_model)
    print('Done')
# 1 - 决策树回归
# 2 - 线性回归（多项式）
# 3 - SVM回归 **
# 4 - KNN回归
# 5 - 随机森林回归 ***
# 6 - Adaboost回归 ***
# 7 - GBDT回归 ****
# 8 - Bagging回归 ***
# 9 - ExtraTree极端随机数回归
