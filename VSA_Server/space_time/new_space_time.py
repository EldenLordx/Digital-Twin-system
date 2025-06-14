# -*- coding: utf-8 -*-

# from sklearn.externals import joblib
import joblib
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
import math
import os
import cv2
import time
from multiprocessing import Process
import threading





def pixel2camera(camera_num, pixel_x, pixel_y):
    path_model = './space_time/model/'
    filename_model_x = '{cameraNum}_x.model'. \
        format(cameraNum=camera_num)
    model_x = joblib.load(os.path.join(path_model, filename_model_x))
    filename_model_y = '{cameraNum}_y.model'. \
        format(cameraNum=camera_num)
    model_y = joblib.load(os.path.join(path_model, filename_model_y))
    # print(model_x)
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
    # param x0��y0: �۲�����ϵԭ������������ϵ�µ�����
    # param theta: �۲�����ϵy��˳ʱ����ת����������ϵ�Ľ�
    # param x1��y1:�۲���ڹ۲�����ϵ�µ�����
    # return: x��y���۲������������ϵ�µ�����
    theta = math.radians(theta)
    x = x1 * math.cos(theta) - y1 * math.sin(theta) + x0
    y = x1 * math.sin(theta) + y1 * math.cos(theta) + y0
    return x, y


def test(camera_num, videoname, results_txt_path, time_txt_path, save_txt_path):
    print("Locating Person From " + results_txt_path + '...')
    trail_x = []
    trail_y = []
    new_lines = []
    cap = cv2.VideoCapture()
    # cap.open(videoname)
    # print(cap.isOpened())
    n = 0
    with open(results_txt_path, 'r') as f, open(time_txt_path, 'r') as t:
        lines = f.readlines()
        times = t.readlines()
        for line in lines:
            line = line.strip()
            list_item = line.split(' ')
            frame_id = int(list_item[0])
            x0 = int(list_item[1])
            y0 = int(list_item[2])
            x1 = int(list_item[3])
            y1 = int(list_item[4])
            # ratio = float(list_item[6])
            ratio = 0
            pixel_x = (int(x0) + int(x1)) / 2
            # while n != frame_id:
            #     ret, frame = cap.read()
            #     n += 1
            # ratio = posed.halfdetect(frame[int(y0):int(y1 + 1), int(x0):int(x1 + 1)])
            if ratio > 0.2:
                pixel_y = int((float(y1) - float(y0)) / (1 - ratio) + float(y0))
            else:
                pixel_y = int(y0)
            map_x, map_y = pixel2map(camera_num, pixel_x, pixel_y)
            trail_x.append(map_x)
            trail_y.append(map_y)
            time_line = times[int(frame_id) - 1]
            time_line = time_line.strip()
            time_line = time_line.split(' ')
            time = time_line[1]
            temp_line = line + ' ' + str(int(map_x)) + ' ' + str(int(map_y)) + ' ' + str(time) + '\n'
            new_lines.append(temp_line)

    # with open(results_txt_path, 'w+') as file:
    with open(save_txt_path, 'w+') as file:
        for index, line in enumerate(new_lines):
            file.write(line)


def get_location_time_from_txt(camera_num, results_txt_path, time_txt_path, save_txt_path):
    # posed = PoseDetect()
    print("Locating Person From " + results_txt_path + '...')
    trail_x = []
    trail_y = []
    new_lines = []
    # videoname = './camera/origin/' + str(camera_num) + '/' + \
    #             time_txt_path.split('/')[-1].split('.')[0] + '.mp4'
    # cap = cv2.VideoCapture()
    # cap.open(videoname)
    # if not cap.isOpened():
    #     return
    n = 0
    with open(results_txt_path, 'r') as f, open(time_txt_path, 'r') as t:
        lines = f.readlines()
        times = t.readlines()
        for line in lines:
            line = line.strip()
            list_item = line.split(' ')
            frame_id = int(list_item[0])
            x0 = int(list_item[1])
            y0 = int(list_item[2])
            x1 = int(list_item[3])
            y1 = int(list_item[4])
            box = [x0, y0, x1, y1]
            for i in range(len(box)):
                if box[i] < 0: box[i] = 0
            x0, y0, x1, y1 = box
            # ratio = float(list_item[6])
            pixel_x = (int(x0) + int(x1)) / 2
            ratio = 0
            # while n != frame_id:
            #     ret, frame = cap.read()
            #     n += 1
            # ratio = posed.halfdetect(frame[int(y0):int(y1 + 1), int(x0):int(x1 + 1)])
            # if ratio==1:continue
            if camera_num == 35 or camera_num == 36:
                pixel_y = int(y0)
            else:
                pixel_y = int(y1)
            map_x, map_y = pixel2map(camera_num, pixel_x, pixel_y)
            trail_x.append(map_x)
            trail_y.append(map_y)
            time_line = times[int(frame_id) - 1]
            time_line = time_line.strip()
            time_line = time_line.split(' ')
            time = time_line[1]
            temp_line = line + ' ' + str(int(map_x)) + ' ' + str(int(map_y)) + ' ' + str(time) + '\n'
            new_lines.append(temp_line)

    # with open(results_txt_path, 'w+') as file:
    with open(save_txt_path, 'w+') as file:
        for index, line in enumerate(new_lines):
            file.write(line)


def my_loop(camera_id):
    # max_video = 50
    # mot_path = './camera/txt_results/person_reid' + str(camera_id)
    # time_path = './camera/txt_results/time' + str(camera_id)
    # # save_dir='/mnt/disk2/vsa/kdy/save_dir/'
    # save_dir = './camera/txt_results/person_space_time' + str(camera_id)
    max_video = 60
    mot_path = '../hikcamera/camera/txt_results/person_mot' + str(camera_id)
    time_path = '../hikcamera/camera/txt_results/time' + str(camera_id)
    # save_dir='/mnt/disk2/vsa/kdy/save_dir/'
    save_dir = '../hikcamera/camera/txt_results/person_space_time' + str(camera_id)
    while True:
        mot_list = os.listdir(mot_path)  # 原始视频数量的控制由摄像头接入部分控制
        mot_list.sort()
        if len(mot_list) < 2:  # 没有视频文件或者当前正在生成第一个视频文件
            print("waiting")
            time.sleep(0.1)  # 避免频繁时间片轮换
            # continue
        result_txt = mot_list[-2]
        result_txt_path = os.path.join(mot_path, result_txt)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        txt_list = os.listdir(save_dir)  # 结果视频存储目录
        txt_list.sort()
        while len(txt_list) >= max_video:
            os.remove(os.path.join(save_dir, txt_list[0]))
            txt_list = os.listdir(save_dir)
            txt_list.sort()

        save_txt_path = save_dir + '/' + result_txt
        # print(save_txt_path)
        time_txt_path = time_path + '/' + result_txt
        if not os.path.exists(save_txt_path):
            start = time.time()
            if not os.path.exists(time_txt_path):
                print(time_txt_path + 'is not exists')
                # continue
            if os.path.exists(result_txt_path):
                get_location_time_from_txt(camera_id, result_txt_path, time_txt_path, save_txt_path)
            else:
                print('not exist:', result_txt_path)
            end = time.time()
            print('Running time: %s Seconds' % (end - start))


def space_time(filename,camera_id):
    max_video = 60
    mot_path = '../hikcamera/camera/txt_results/person_mot' + str(camera_id)
    time_path = '../hikcamera/camera/txt_results/time' + str(camera_id)
    # save_dir='/mnt/disk2/vsa/kdy/save_dir/'
    save_dir = '../hikcamera/camera/txt_results/person_space_time' + str(camera_id)

    # mot_list = os.listdir(mot_path)  # 原始视频数量的控制由摄像头接入部分控制
    # mot_list.sort()
    # if len(mot_list) < 2:  # 没有视频文件或者当前正在生成第一个视频文件
    #     print("waiting")
    #     time.sleep(0.1)  # 避免频繁时间片轮换
    #     # continue
    result_txt = filename
    result_txt_path = os.path.join(mot_path, result_txt)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    txt_list = os.listdir(save_dir)  # 结果视频存储目录
    txt_list.sort()
    while len(txt_list) >= max_video:
        os.remove(os.path.join(save_dir, txt_list[0]))
        txt_list = os.listdir(save_dir)
        txt_list.sort()

    save_txt_path = save_dir + '/' + result_txt
    # print(save_txt_path)
    time_txt_path = time_path + '/' + result_txt
    if not os.path.exists(save_txt_path):
        start = time.time()
        if not os.path.exists(time_txt_path):
            print(time_txt_path + 'is not exists')
            # continue
        if os.path.exists(result_txt_path):
            get_location_time_from_txt(camera_id, result_txt_path, time_txt_path, save_txt_path)
        else:
            print('not exist:', result_txt_path)
        end = time.time()
        print('Running time: %s Seconds' % (end - start))

def loop_func():
    No = [17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 35, 36]
    #No = [17, 19, 20, 21, 23, 27, 28, 35]
    for camera_id in No:
        #my_loop(camera_id)
        Process(target=my_loop, args=(camera_id,)).start()
        # threading.Thread(target=my_loop, args=(camera_id,)).start()


if __name__ == '__main__':
    loop_func()

    # test(camera_num=36, videoname='36-1.mp4', results_txt_path='36-1.txt', time_txt_path='time.txt',
    #      save_txt_path='36.txt')

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
