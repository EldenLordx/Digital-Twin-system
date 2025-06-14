import os
import time
import numpy as np
import cv2

from mot.deepsort.person_deep_sort import PersonDeepSort
from config import *
from utils.util import draw_bboxes
from config import *


def getDetections(detect_txt_path):
    detections = {}
    with open(detect_txt_path, 'r') as file:
        while True:
            infos = file.readline()
            infos = infos.split(' ')
            if len(infos)>1:
                frame_id = int(infos[0])
                lt_x = int(infos[1])
                lt_y = int(infos[2])
                br_x = int(infos[3])
                br_y = int(infos[4])
                conf = float(infos[5])
                img_path = infos[6].strip()
                w = br_x - lt_x
                h = br_y - lt_y
                cx = lt_x + int(0.5*w)
                cy = lt_y + int(0.5*h)
                bbox_xywh = [cx, cy, w, h]
                if not detections.__contains__(frame_id):
                    detections[frame_id] = []
                detect_item = []
                detect_item.append(bbox_xywh)
                detect_item.append(conf)
                detect_item.append(img_path)
                detections[frame_id].append(detect_item)
            else:
                break
    return detections


def getCurrentDetection(detection_frame):
    bbox_xywh = []
    cls_conf = []
    img_path = []
    for i in range(len(detection_frame)):
        item = detection_frame[i]
        bbox_xywh.append(item[0])
        cls_conf.append(item[1])
        img_path.append(item[2])
    return bbox_xywh, cls_conf, img_path


class PersonMOTTask(object):
    def __init__(self, camera_id, gpu_id):
        self.camera_id = camera_id
        self.gpu_id = gpu_id
        self.tracker = PersonDeepSort(self.gpu_id)


    def work(self, input_arr, detect_txt_path):
        output_arr = []
        detections = getDetections(detect_txt_path)
        video_name = detect_txt_path.split('/')[-1][:-4]
        # track info
        root_path_mot = person_mot_txt_path + str(self.camera_id)
        if not os.path.exists(root_path_mot):
            os.makedirs(root_path_mot)
        self.result_file_path = root_path_mot + '/' + video_name + '.txt'
        #self.result_file_path= '/mnt/disk2/vsa/VSA_Server/camera/txt_results/' + video_name + '.txt'
        print("save mot result at:"+self.result_file_path)
        self.frame_index = 1
        self.output_txt = open(self.result_file_path, 'w')
        for img in input_arr:
            if detections.__contains__(self.frame_index):  # 跳帧处理
                #if self.frame_index% detect_skip!=0:break
                detection_frame = detections[self.frame_index]
                bbox_xywh, cls_conf, cls_ids = getCurrentDetection(detection_frame)
                if bbox_xywh is not None:
                    outputs = self.tracker.update(bbox_xywh, cls_conf, img)  # track
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]  # position
                        identities = outputs[:, -1]  # id
                        for i in range(len(outputs)):
                            output = outputs[i]
                            x1 = output[0]
                            y1 = output[1]
                            x2 = output[2]
                            y2 = output[3]
                            id = output[4]
                            # feature = output[5]
                            res = ''
                            res = res + str(self.frame_index) + ' ' + str(x1) + ' ' + str(y1) + ' ' + str(
                                x2) + ' ' + str(y2) + ' ' + str(
                                id) + '\n'
                            self.output_txt.write(res)


                                # print('index = ' + str(self.frame_index)
                                #       )

                                # if (int(self.frame_index) + 1) % 4 == 0:
                                #     roi = img[y1:y2, x1:x2]
                                #     filename = str(self.camera_id) + '_' + video_name + '_' + str(id) + '_' + str(self.frame_index) + '.jpg'
                                #     path = os.path.join(root_path, '/camera/imgs_test/' + str(self.camera_id))
                                #     path_filename = path + '/' + filename
                                #     cv2.imwrite(path_filename, roi)
                                #     print(path_filename + ' write done!')
                                #     self.clearFile(path)


                        img = draw_bboxes(img, bbox_xyxy, identities)
                        # print(bbox_xyxy)
            output_arr.append(img)
            self.frame_index = self.frame_index + 1
            # end = time.time()
            # print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))
        # print(len(output_arr))
        self.output_txt.close()
        return output_arr

    def clearFile(self, path):
        list_files = os.listdir(path)
        if len(list_files) > 500:
            list_files.sort()
            del_name = list_files[0]
            os.remove(path + '/' + del_name)
