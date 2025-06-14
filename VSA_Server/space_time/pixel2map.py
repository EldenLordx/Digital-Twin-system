# -*- coding: utf-8 -*-
# @author:  shenyuxuan
# @contact: 1044808224@qq.com

from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

import sklearn.metrics as sm
import numpy as np
import csv
import math
import os
import matplotlib.pyplot as plt
import sys
import cv2

sys.path.append('/home/kdy/skdy/IDS')
from config_IDS import *


def pixel2camera(camera_num, pixel_x, pixel_y):
    path_model = os.path.join(path_IDS_root, 'model_IDS', 'location')
    filename_model_x = '{cameraNum}_x.model'. \
        format(cameraNum=camera_num)
    model_x = joblib.load(os.path.join(path_model, filename_model_x))
    filename_model_y = '{cameraNum}_y.model'. \
        format(cameraNum=camera_num)
    model_y = joblib.load(os.path.join(path_model, filename_model_y))

    degrees = [0] * 37
    degrees[17] = [3, 1]
    degrees[18] = [1, 1]
    degrees[19] = [1, 1]
    degrees[20] = [12, 1]
    degrees[21] = [1, 1]
    degrees[23] = [8, 1]
    degrees[24] = [1, 1]
    degrees[25] = [3, 1]
    degrees[27] = [2, 8]
    degrees[28] = [9, 3]
    degrees[35] = [1, 1]
    degrees[36] = [1, 1]

    quadratic_featurizer_x = PolynomialFeatures(degree=degrees[camera_num][0])
    quadratic_featurizer_y = PolynomialFeatures(degree=degrees[camera_num][1])

    content = [[pixel_x, pixel_y]]
    content = np.array(content)
    content = content[:, :]

    list_pixel_x = quadratic_featurizer_x.fit_transform(content)
    list_pixel_y = quadratic_featurizer_y.fit_transform(content)

    material_x = np.array(list_pixel_x[0]).reshape(1, -1)
    material_y = np.array(list_pixel_y[0]).reshape(1, -1)

    pred_x = model_x.predict(material_x)[0]
    pred_y = model_y.predict(material_y)[0]

    camera_x = int(pred_x)
    camera_y = int(pred_y)

    return camera_x, camera_y


def camera2map(camera_num, camera_x, camera_y):
    maps = [0] * 37
    # [x,y,angle]
    maps[17] = [1879, -13593, 100]
    maps[18] = [-2412, 13884, 280]
    maps[19] = [-24040, -18147, 280]
    maps[20] = [23654, 18161, 100]
    maps[21] = [24142, -18192, 80]
    maps[23] = [-1716, -13618, 260]
    maps[24] = [-23906, 18116, 260]
    maps[25] = [2481, 13883, 80]
    maps[27] = [782, 4000, 151]
    maps[28] = [761, -3898, 19]
    maps[35] = [4, -15261, 0]
    maps[36] = [-3, 15184, 180]

    x = maps[camera_num][0]
    y = maps[camera_num][1]
    angle = maps[camera_num][2]

    radian = math.radians(angle)
    sin = math.sin(radian)
    cos = math.cos(radian)
    tan = math.tan(radian)

    map_x, map_y = coordinate_trans(x, y, angle, camera_x, camera_y)

    return map_x, map_y


def pixel2map(camera_num, pixel_x, pixel_y):
    camera_x, camera_y = pixel2camera(camera_num, pixel_x, pixel_y)
    map_x, map_y = camera2map(camera_num, camera_x, camera_y)
    return map_x, map_y


def coordinate_trans(x0, y0, theta, x1, y1):
    #param x0，y0: 观察坐标系原点在世界坐标系下的坐标
    #param theta: 观察坐标系y轴顺时针旋转到世界坐标系的角
    #param x1，y1:观察点在观察坐标系下的坐标
    #return: x，y：观察点在世界坐标系下的坐标
    theta = math.radians(theta)
    x = x1 * math.cos(theta) - y1 * math.sin(theta) + x0
    y = x1 * math.sin(theta) + y1 * math.cos(theta) + y0
    return x, y


def get_location_from_txt_mot(camera_num, results_txt_path, flag_draw_trail=False):
    print('Locating Person From', results_txt_path, '...')
    trail_x = []
    trail_y = []
    new_lines = []
    with open(results_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            list_item = line.split(' ')
            frame_id = list_item[0]
            x0 = list_item[1]
            y0 = list_item[2]
            x1 = list_item[3]
            y1 = list_item[4]
            pixel_x = (int(x0) + int(x1)) / 2
            pixel_y = int(y1)
            map_x, map_y = pixel2map(camera_num, pixel_x, pixel_y)
            trail_x.append(map_x)
            trail_y.append(map_y)
            temp_line = line + ' ' + str(int(map_x)) + ' ' + str(int(map_y)) + '\n'
            new_lines.append(temp_line)
    os.remove(results_txt_path)

    with open(results_txt_path, 'w') as file:
        for index, line in enumerate(new_lines):
            # 3 634 53 690 193 1569755097.105 map_x map_y
            file.write(line)

    if flag_draw_trail:
        print('Drawing Map From', results_txt_path, '...')
        draw_map(results_txt_path)


def get_location_from_txt(camera_num, results_txt_path, trail_img_path, flag_draw_trail=False, flag_save_plt=True):
    print('Locating Person From', results_txt_path, '...')
    trail_x = []
    trail_y = []
    new_lines = []
    with open(results_txt_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            list_item = line.split(' ')
            frame_id = list_item[0]
            x0 = list_item[1]
            y0 = list_item[2]
            x1 = list_item[3]
            y1 = list_item[4]
            pixel_x = (int(x0) + int(x1)) / 2
            pixel_y = int(y1)
            map_x, map_y = pixel2map(camera_num, pixel_x, pixel_y)
            trail_x.append(map_x)
            trail_y.append(map_y)
            temp_line = line + ' ' + str(int(map_x)) + ' ' + str(int(map_y)) + '\n'
            new_lines.append(temp_line)
    os.remove(results_txt_path)

    with open(results_txt_path, 'w') as file:
        for index, line in enumerate(new_lines):
            # 3 634 53 690 193 0.98689 3_0 1569755097.105 map_x map_y
            file.write(line)

    if flag_draw_trail:
        draw_trail_from_list(trail_x, trail_y, trail_img_path, results_txt_path, flag_save_plt)


def draw_trail_from_list(trail_x, trail_y, trail_img_path, results_txt_path, flag_save_plt=False):
    plt.title(u'map', y=0)

    ax = plt.subplot()
    ax.set_xlim(left=-30000, right=30000)
    ax.set_ylim(bottom=-30000, top=30000)
    plt.grid()
    ax.plot(trail_x, trail_y, '-', label='ax', color='r', linewidth=0.5)

    if flag_save_plt and len(trail_x) != 0:
        # camera_num = results_txt_path.split('/')[-2]
        video_name = results_txt_path.split('/')[-1].split('.')[0]

        trail_img_path_eps = os.path.join(trail_img_path, video_name + '.pdf')
        plt.savefig(trail_img_path_eps, dpi=600, format='pdf')

        # trail_img_path_png = os.path.join(trail_img_path, video_name + '.png')
        # plt.savefig(trail_img_path_png)

        print('trail image has been saved in', trail_img_path)
        # plt.show()


def draw_trail(camera_num=20, path_txt=None):
    plt.title(u'map', y=0)

    trail = []
    trail_x = []
    trail_y = []

    ax = plt.gca()
    ax.set_xlim(left=-30000, right=30000)
    ax.set_ylim(bottom=-30000, top=30000)

    with open(path_txt, 'r') as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            line = line.strip()
            list_item = line.split(' ')
            frame_id = list_item[0]
            x0 = list_item[1]
            y0 = list_item[2]
            x1 = list_item[3]
            y1 = list_item[4]
            time_stamp = list_item[7]
            pixel_x = (int(x0) + int(x1)) / 2
            pixel_y = int(y1)
            camera_x, camera_y = pixel2camera(camera_num, pixel_x, pixel_y)
            map_x, map_y = camera2map(camera_num, camera_x, camera_y)
            print(index, map_x, map_y)
            temp_trail = [map_x, map_y]
            trail.append(temp_trail)
            trail_x.append(map_x)
            trail_y.append(map_y)
        trail_x.append(-10000)
        trail_y.append(-10000)
    ax.plot(trail_x, trail_y, '--', label='ax', color='r', linewidth=1)
    plt.show()


# Draw Map
def draw_map(path_txt_mot):
    list_path = path_txt_mot.split('/')
    camera_num = list_path[-2]
    video_name = list_path[-1].split('.')[0]
    temp_path_target = os.path.join(person_trail_img_path, camera_num, video_name)

    if not os.path.exists(temp_path_target):
        os.mkdir(temp_path_target)

    with open(path_txt_mot, 'r') as f:
        lines = f.readlines()
        min_frame_id = int(lines[0].strip().split(' ')[0])
        max_frame_id = int(lines[-1].strip().split(' ')[0])
        len_lines = len(lines)

        for index in range(min_frame_id, max_frame_id + 1):
            for line_index, line in enumerate(lines):

                line = line.strip()
                list_items = line.split(' ')
                frame_id = int(list_items[0])

                id = int(list_items[5])
                map_x = int(list_items[7])
                map_y = int(list_items[8])

                if index == frame_id:
                    list_position = []
                    temp_position = [map_x, map_y]
                    list_position.append(temp_position)

                    person_id = []
                    person_id.append(id)

                    colors = [(250, 0, 0), (0, 250, 0), (0, 0, 250)]

                    frame_error = 1
                    frame_left = max(0, frame_id - frame_error)
                    frame_right = min(frame_id + frame_error + 1, max_frame_id)

                    index_error = 5
                    index_left = max(line_index - index_error, 0)
                    index_right = min(line_index + index_error, len_lines)
                    for sub_index in range(index_left, index_right):
                        sub_line = lines[sub_index].strip()
                        sub_list_items = sub_line.split(' ')
                        sub_frame_id = int(sub_list_items[0])
                        sub_id = int(sub_list_items[5])
                        if (sub_id not in person_id) and (sub_frame_id >= frame_left and sub_frame_id <= frame_right):
                            person_id.append(sub_id)
                            sub_map_x = int(sub_list_items[7])
                            sub_map_y = int(sub_list_items[8])
                            temp_position = []
                            temp_position = [sub_map_x, sub_map_y]
                            list_position.append(temp_position)

                    path_img = os.path.join(project_path, 'static', 'F6-modify.png')
                    img = cv2.imread(path_img)
                    for index_person, list_pos in enumerate(list_position):
                        color = colors[person_id[index_person] % len(colors)]
                        # color = colors[1]
                        img = draw_with_cv2(img, list_pos[0], list_pos[1], color)

                    # print('choose', index, list_pos[0], list_pos[1], )

                    path_target = os.path.join(temp_path_target, str(index) + '.jpg')
                    cv2.imwrite(path_target, img)
                    break


def draw_with_cv2(img, map_x, map_y, color):
    w, h, _ = img.shape
    map_x = int((map_x + 30000) / 60000 * w)
    map_y = int(-(map_y - 30000) / 60000 * h)
    center = (map_x, map_y)
    radius = 5
    thickness = -1 
    cv2.circle(img, center=center, radius=radius, color=color, thickness=thickness)
    
    return img


if __name__ == '__main__':
    camera_num = 19
    pixel_x = 710
    pixel_y = 460

    camera_x, camera_y = pixel2camera(camera_num, pixel_x, pixel_y)
    print('camera:', camera_x, camera_y)
    map_x, map_y = camera2map(camera_num, camera_x, camera_y)
    print('map:', map_x, map_y)

    # path_txt = '/home/syx/workspace/pixel_2_map/1569754994.txt'
    # draw_trail(path_txt=path_txt)

    # path_txt = '/home/syx/mmap_vsa/1569754994.txt'
    # camera_num = 20
    # get_location_from_txt(self.camera_id, det_txt_path, self.person_trail_img_path,flag_draw_trail=True, flag_save_plt=True)

"""
maps[17] = [1879, -13593]
maps[18] = [-2412, 13884, 280]
maps[19] = [-24040, -18147, 280]
maps[20] = [23654, 18161, 100]
maps[21] = [24142, -18192, 80]
maps[23] = [-1716, -13618, ]
maps[24] = [-23906, 18116, 260]
maps[25] = [2481, 13883, ]
maps[27] = [782, 4000, 151]
maps[28] = [761, -3898, 19]
maps[35] = [4, -15261, 0]
maps[36] = [-3, 15184, 180]

18: 561,416     -2412, 13884, 280, 359, 5047        2622,14388
19: 710,460     -24040, -18147, 280, 0, 5745        -18382, -17153     
20: 630,520     23654, 18161, 100, -64, 4136        19592, 17452
21: 663,300     24142, -18192, 80, 0, 9334          14950,-16566
24: 969,534     -23906, 18116, 260, 728,4310        -19663, 17367
28: 863,391     761, -3898, 19, 1851, 7673          47,4000
35: 730,525     4, -15261, 0, 4, 7524               4,-7788
36: 700,590     -3, 15184, 180, 0, 7434             0, 7750
"""
