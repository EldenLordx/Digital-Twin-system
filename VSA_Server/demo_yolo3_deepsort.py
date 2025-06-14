import os
import cv2

from persondetect.YOLO3 import YOLO3
from mot.deepsort.deep_sort import DeepSortTracker
from utils.util import draw_bboxes

import time

video_path = "/home/cliang/mmap/deep_sort_pytorch/test_video/"
result_path = "/home/cliang/mmap/deep_sort_pytorch/results/"
fps = 25
alternation = 1

yolo3 = YOLO3(0, "persondetect/YOLO3/cfg/yolov3-tiny.cfg", "persondetect/YOLO3/yolov3-tiny_20000.weights", "persondetect/YOLO3/cfg/coco.names", is_xywh=True)

# yolo3 = YOLO3(0, "mot/YOLO3/mobilenet_yolo_voc.cfg", "mot/YOLO3/mobilenet_yolo_voc.weights", "mot/YOLO3/cfg/coco.names", is_xywh=True)



deepsort = DeepSortTracker(0, "feature/checkpoint/ckpt.t7")
index = 0

class Detector(object):
    def __init__(self, video_name, result_file_path):
        self.result_name = video_name + "_res"
        self.result_file_path = result_file_path

        self.vdo = cv2.VideoCapture()

        self.class_names = yolo3.class_names
        self.write_video = True

    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        print(self.vdo.get(cv2.CAP_PROP_FPS))
        self.frame_index = 0
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.im_width = 960
        self.im_height = 540
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.output_path = "demo.mp4"
            self.output = cv2.VideoWriter(self.output_path, fourcc, fps, (self.im_width, self.im_height))
        return self.vdo.isOpened()

    def detect(self):
        global index
        index = index+1
        xmin, ymin, xmax, ymax = self.area
        while self.vdo.grab():
            _, ori_im = self.vdo.retrieve()
            ori_im = cv2.resize(ori_im, (self.im_width, self.im_height))
            im = ori_im[ymin:ymax, xmin:xmax, (2, 1, 0)]
            bbox_xywh, cls_conf, cls_ids = yolo3(im)
            if bbox_xywh is not None:
                mask = cls_ids == 0
                bbox_xywh = bbox_xywh[mask]
                bbox_xywh[:, 3] *= 1.2
                cls_conf = cls_conf[mask]
                outputs = deepsort.update(bbox_xywh, cls_conf, im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin))

            if self.write_video:
                self.output.write(ori_im)


if __name__ == "__main__":
    path = 'hh.mp4'
    video_name = 'V-020120'
    result_file_path = os.path.join(result_path, video_name + ".txt")
    if os.path.exists(result_file_path):
        print("this video has handle!")
    else:
        # os.mknod(result_file_path)
        start = time.time()
        det = Detector(video_name, result_file_path)
        det.open(path)
        det.detect()
        end = time.time()
        print("need time: {}s".format(end - start))
        print('fps:{}fps'.format(500.0/(end-start)))
