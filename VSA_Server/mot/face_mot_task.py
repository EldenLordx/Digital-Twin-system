import cv2

from facedetect.mtcnn.detector import MTCNNFaceDetector
from mot.deepsort.face_deep_sort import FaceDeepSort
from config import *
from utils.util import draw_bboxes, convert_to_square
from PIL import Image
import time
import os
import numpy as np


class FaceMOTTask(object):
    def __init__(self, camera_id, gpu_id):
        self.camera_id = camera_id
        self.gpu_id = gpu_id
        self.detector = MTCNNFaceDetector()
        self.detector.init(gpu_id)
        self.tracker = FaceDeepSort(self.gpu_id)

    def save_img(self, img, bboxes, identities, img_size, landmarks):
        save_dir = os.path.join(face_detect_img_path, str(self.camera_id))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(face_detect_txt_path, str(self.camera_id), 'info.txt'), 'a') as info_file:
            for i, bbox in enumerate(bboxes):
                img_seq = int(str(time.time()).split('.')[1])
                img_name = os.path.join(save_dir, '{}.jpg'.format(img_seq))
                x1, y1, x2, y2 = convert_to_square(bbox, img_size)
                im = img[y1:y2, x1:x2]
                cv2.imwrite(img_name, im)
                print(type(bbox),type(identities[i]))
                info_file.write('{img_name} {id} {bbox[0]} {bbox[1]} {bbox[2]}'
                                ' {bbox[3]} {ld[0]:0.1f} {ld[1]:0.1f} {ld[2]:0.1f} {ld[3]:0.1f} {ld[4]:0.1f}'
                                ' {ld[5]:0.1f} {ld[6]:0.1f} {ld[7]:0.1f} {ld[8]:0.1f} {ld[9]:0.1f}\n'
                                .format(img_name=img_seq, bbox=bbox, id=identities[i], ld=landmarks[i]))
        print('save img.')

    def work(self, input_arr, videoPath):
        output_arr = []
        video_name = videoPath.split('/')[-1][:-4]
        for i in range(len(input_arr)):
            img = input_arr[i]
            img_size = np.asarray(img.shape)[0:2]
            img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if i % alternation == 0:  # 跳帧处理
                bbox_xywh, landmarks = self.detector.detect(img_pil)
                # if bbox_xywh is not None:
                if len(bbox_xywh)!=0:
                    cls_conf = [i[4] for i in bbox_xywh]
                    bbox_xywh = [[(i[0] + i[2]) / 2, (i[1] + i[3]) / 2, i[2] - i[0], i[3] - i[1]] for i in bbox_xywh]
                    outputs = self.tracker.update(bbox_xywh, cls_conf, img)
                    if len(outputs) > 0:
                        bbox_xyxy = outputs[:, :4]
                        identities = outputs[:, -1]
                        #self.save_img(img, bbox_xyxy, identities, img_size, landmarks)
                        img = draw_bboxes(img, bbox_xyxy, identities, offset=(0, 0))
            output_arr.append(img)
            # end = time.time()
            # print("time: {}s, fps: {}".format(end - start, 1 / (end - start)))
        return output_arr