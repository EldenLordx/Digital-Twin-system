import cv2
import time
import datetime
import os
import numpy as np
import shutil


def find_max_min_pixel(camera_num):
    max_x = 0
    max_y = 0

    min_x = 2000
    min_y = 2000

    path_txts = os.path.join('C:/Users/sheny/Desktop/camera/txt_results/person_detect', str(camera_num))
    list_txts = os.listdir(path_txts)
    for file_txt in list_txts:
        file_txt_full = os.path.join(path_txts, file_txt)
        with open(file_txt_full, 'r') as file:
            lines = file.readlines()
            for index, line in enumerate(lines):
                # 帧号 左上角x 左上角y 右下角x 右下角y 置信度 图片名 时间戳
                # 97 397 110 453 360 0.90021 97_0 1568946903.600
                line = line.strip()
                list_line = line.split(' ')
                frame_id = list_line[0]
                x1 = int(list_line[1])
                y1 = int(list_line[2])
                x2 = int(list_line[3])
                y2 = int(list_line[4])
                foot_x = (x1 + x2) / 2
                foot_y = y2

                if foot_x > max_x:
                    max_x = foot_x
                elif foot_x < min_x:
                    min_x = foot_x

                if foot_y > max_y:
                    max_y = foot_y
                elif foot_y < min_y:
                    min_y = foot_y
    return max_x, max_y, min_x, min_y


if __name__ == '__main__':
    max_x, max_y, min_x, min_y = find_max_min_pixel(20)
    print('max_y', max_y)
    print('min_y', min_y)
    print('max_x', max_x)
    print('min_x', min_x)
