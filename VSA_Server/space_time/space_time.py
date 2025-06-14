# -*- coding: utf-8 -*-

from sklearn import linear_model
from sklearn.externals import joblib
#import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from threading import Timer#new

import sklearn.metrics as sm
import numpy as np
import csv
import math
import os
import matplotlib.pyplot as plt
import random
import cv2
import re#new
from delete_file import *
import time

def pixel2camera(camera_num, pixel_x, pixel_y):
    path_model = '../model_IDS/location/'
    filename_model_x = '{cameraNum}_x.model'. \
        format(cameraNum=camera_num)
    model_x = joblib.load(os.path.join(path_model, filename_model_x))
    filename_model_y = '{cameraNum}_y.model'. \
        format(cameraNum=camera_num)
    model_y = joblib.load(os.path.join(path_model, filename_model_y))
    #print(model_x)
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
    #param x0��y0: �۲�����ϵԭ������������ϵ�µ�����
    #param theta: �۲�����ϵy��˳ʱ����ת����������ϵ�Ľ�
    #param x1��y1:�۲���ڹ۲�����ϵ�µ�����
    #return: x��y���۲������������ϵ�µ�����
    theta = math.radians(theta)
    x = x1 * math.cos(theta) - y1 * math.sin(theta) + x0
    y = x1 * math.sin(theta) + y1 * math.cos(theta) + y0
    return x, y


#��ԭ�л����ϼ���һ���β�
def get_location_from_txt(camera_num, results_txt_path, save_txt_path, trail_img_path, flag_draw_trail=False, flag_save_plt=True):
    print("Locating Person From " + results_txt_path + '...')
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
    #os.remove(results_txt_path)

    #with open(results_txt_path, 'w+') as file:
    with open(save_txt_path, 'w+') as file:
        for index, line in enumerate(new_lines):
            # 3 634 53 690 193 0.98689 3_0 1569755097.105 map_x map_y
            file.write(line)

    if flag_draw_trail:
        draw_trail_from_list(trail_x, trail_y, trail_img_path, results_txt_path, flag_save_plt)


def get_location_time_from_txt(camera_num, results_txt_path, time_txt_path, save_txt_path):
    print("Locating Person From " + results_txt_path + '...')
    trail_x = []
    trail_y = []
    new_lines = []
    with open(results_txt_path, 'r') as f, open(time_txt_path, 'r') as t:
        lines = f.readlines()
        times = t.readlines()
        for line in lines:
            line = line.strip()
            list_item = line.split(' ')
            frame_id = list_item[0]
            x0 = list_item[1]
            y0 = list_item[2]
            x1 = list_item[3]
            y1 = list_item[4]
            ratio = float(list_item[6])
            pixel_x = (int(x0) + int(x1)) / 2
            if ratio>0.3:
                pixel_y = int((float(y1)-float(y0))/(1-ratio)+float(y0))
            else:
                pixel_y = int(y1)
            map_x, map_y = pixel2map(camera_num, pixel_x, pixel_y)
            trail_x.append(map_x)
            trail_y.append(map_y)
            time_line = times[int(frame_id)-1]
            time_line = time_line.strip()
            time_line = time_line.split(' ')
            time = time_line[1]
            temp_line = line + ' ' + str(int(map_x)) + ' ' + str(int(map_y)) + ' ' + str(time) + '\n'
            new_lines.append(temp_line)

    #with open(results_txt_path, 'w+') as file:
    with open(save_txt_path, 'w+') as file:
        for index, line in enumerate(new_lines):
            file.write(line)



def draw_trail_from_list(trail_x, trail_y, trail_img_path, results_txt_path, flag_save_plt=False):
    plt.title(u'map', y=0)

    path_img_background = '/home/syx/workspace/pixel_2_map/test/F6.wmf'
    img_background = plt.imread(path_img_background)

    fig = plt.figure()
    min_x = -30000
    max_x = 30000
    min_y = -30000
    max_y = 30000
    add_axes = [0, 0, 60000, 60000]
    ax0 = fig.add_axes(add_axes)
    ax0.imshow(img_background)

    ax1 = fig.add_axes(add_axes, frameon=False)
    ax1.scatter(trail_x, trail_y, '-', label='ax', color='r', linewidth=0.5)

    fig.show()

    flag_save_plt = False
    if flag_save_plt and len(trail_x) != 0:
        # camera_num = results_txt_path.split('/')[-2]
        video_name = results_txt_path.split('/')[-1].split('.')[0]

        trail_img_path_eps = os.path.join(trail_img_path, video_name + '.pdf')
        plt.savefig(trail_img_path_eps, dpi=600, format='pdf')

        # trail_img_path_png = os.path.join(trail_img_path, video_name + '.png')
        # plt.savefig(trail_img_path_png)

        print('trail image has been saved in', trail_img_path)
        # plt.show()


# BCK
# def draw_trail_from_list(trail_x, trail_y, trail_img_path, results_txt_path, flag_save_plt=False):
#     plt.title(u'map', y=0)
#
#     ax = plt.subplot()
#     ax.set_xlim(left=-30000, right=30000)
#     ax.set_ylim(bottom=-30000, top=30000)
#     plt.grid()
#     ax.plot(trail_x, trail_y, '-', label='ax', color='r', linewidth=0.5)
#
#     if flag_save_plt and len(trail_x) != 0:
#         # camera_num = results_txt_path.split('/')[-2]
#         video_name = results_txt_path.split('/')[-1].split('.')[0]
#
#         trail_img_path_eps = os.path.join(trail_img_path, video_name + '.pdf')
#         plt.savefig(trail_img_path_eps, dpi=600, format='pdf')
#
#         # trail_img_path_png = os.path.join(trail_img_path, video_name + '.png')
#         # plt.savefig(trail_img_path_png)
#
#         print('trail image has been saved in', trail_img_path)
#         # plt.show()


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


def draw_with_cv2(map_x, map_y):

    img = cv2.imread('/home/kdy/skdy/IDS/experiment/pixel_2_map/test/F6.png')
    h, w, _ = img.shape

    map_x = int((map_x + 30000) / 60000 * w)
    map_y = int(-(map_y - 30000) / 60000 * h)
    center = (map_x, map_y)
    radius = 5
    #color = (0, 250, 0)
    my_color=(0,0,255)
    thickness = -1
    cv2.circle(img, center=center, radius=radius, color=my_color, thickness=thickness)
    cv2.imwrite('/home/kdy/skdy/IDS/experiment/pixel_2_map/test/my_test.jpg',img)
    #cv2.imshow('2', img)    //����cv2��������ᱨ���ʽ�imshow������Ϊimwrite����



def my_loop():
    start_time=time.time()
    print('the now time:',time.time())
    results_path='./camera/txt_results/'
    #save_dir='/mnt/disk2/vsa/kdy/save_dir/'
    save_dir='./camera/txt_results'
    pattern=re.compile(r'person_mot(\d+)/')
    #/mnt/disk2/vsa/VSA_test/camera/txt_results/person_mot20/
    for root, dirs, files in os.walk(results_path):
    # root ��ʾ��ǰ���ڷ��ʵ��ļ���·��
    # dirs ��ʾ���ļ����µ���Ŀ¼��list
    # files ��ʾ���ļ����µ��ļ�list
        if len(files)<2:
            print('the length of files<2') #Ϊ���ų��ڸ�Ŀ¼�µ�����
            continue
        start=time.time()
        files.sort()
        result_txt=files[-2]
        result_txt_path=os.path.join(root,result_txt)
        #print(result_txt_path)
        results=pattern.search(result_txt_path)
        # if results is None:
        #     # print('results is none')
        #     continue
        # else:
        camera_id=int(36)#���û��ƥ��ɹ���Ӧ��ֱ�����������Ǳ���(Ӧ���Ǹ�detect��space-time)
        #print(camera_id)
        #get_location_from_txt(camera_num, results_txt_path, trail_img_path, flag_draw_trail=False, flag_save_plt=True)
        save_dir_path=os.path.join(save_dir,'person_space_time'+str(camera_id))
        if not os.path.exists(save_dir_path):
            os.makedirs(save_dir_path)
        save_txt_path=os.path.join(save_dir_path,result_txt)
        # print(save_txt_path)
        time_txt_path=os.path.join(results_path,'time'+str(camera_id),result_txt)
        if not os.path.exists(time_txt_path):
            print(time_txt_path+'is not exists')
            continue
        if os.path.exists(result_txt_path):
            get_location_time_from_txt(camera_id, result_txt_path, time_txt_path, save_txt_path)
        else:
            print('not exist:', result_txt_path)
        remove_old_file(save_dir_path,50)
        end=time.time()
        print('Running time: %s Seconds'%(end-start))
    end_time=time.time()
    return(end_time-start_time)


def loop_func(second):
    #ÿ��second��ִ��func����
    while True:
        spend_time=my_loop()
        sleep_time=second-spend_time
        if sleep_time>0:
          print('waiting for next minute...')
          time.sleep(second-spend_time)
        else:
          print('use time more than 1 minute,do not need to wait')
        #timer = Timer(second, func)
        #timer.start()
        #timer.join()

if __name__ == '__main__':
    loop_func(59.95)


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

