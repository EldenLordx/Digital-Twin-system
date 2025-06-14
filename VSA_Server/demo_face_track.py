import os
import cv2
import numpy as np
from PIL import Image
from facedetect.mtcnn.src import detector

from mot.deepsort.deep_sort import DeepSort
from utils.util import draw_bboxes,convert_to_square

import time

video_path = "/home/cliang/mmap/deep_sort_pytorch/test_video/"
result_path = "/home/cliang/mmap/deep_sort_pytorch/results/"
fps = 25
alternation = 1

#yolo3 = YOLO3("YOLO3/cfg/yolov3-tiny.cfg", "YOLO3/yolov3-tiny_20000.weights", "YOLO3/cfg/coco.names", is_xywh=True)
detector = detector()
deepsort = DeepSort(0,"deepfeature/checkpoint/ckpt.t7")
index = 0


class Detector(object):
    def __init__(self, video_name, result_file_path):
        self.result_name = video_name + "_res"
        self.result_file_path = result_file_path

        self.vdo = cv2.VideoCapture()
        self.dir = 'det_seq'
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        #self.class_names = yolo3.class_names
        self.write_video = True
        self.dic = {}
        #self.info_txt = open('info.txt','wb')
        
    def open(self, video_path):
        assert os.path.isfile(video_path), "Error: path error"
        self.vdo.open(video_path)
        self.vdo.set(1,11300)
        print(self.vdo.get(cv2.CAP_PROP_FPS))
        self.frame_index = 0
        self.im_width = int(self.vdo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.im_height = int(self.vdo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        #self.im_width = 960
        #self.im_height = 540
        self.area = 0, 0, self.im_width, self.im_height
        if self.write_video:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self.output_path = "demo.avi"
            self.output = cv2.VideoWriter(self.output_path, fourcc, fps, (self.im_width, self.im_height))
        return self.vdo.isOpened()
    
    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),self.width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),self.height-1)
        return x1,y1,x2,y2
    
    def save_sequence(self,img,bbox_xyxy,identities,img_size):
        for idx,bbox in enumerate(bbox_xyxy):
            self.dic[identities[idx]] = self.dic.get(identities[idx],0)+1
            #localtime = time.localtime()
            #strtime = time.strftime('%m%d%H%M%S',localtime)
            x1,y1,x2,y2 = convert_to_square(bbox,img_size)
            im = img[y1:y2,x1:x2]
            file_name = '{time}-{iden}-{seq}.jpg'.format(time=int(time.time()),iden=identities[idx],seq=self.dic[identities[idx]])
            cv2.imwrite(os.path.join(self.dir,file_name),im)

    def detect(self):
        global index
        index = index+1
        xmin, ymin, xmax, ymax = self.area
        i = 0
        while self.vdo.grab():
            for j in range(1):
                self.vdo.grab()
            if i<5000:
                i+=1
            else:
                break
            _, ori_im = self.vdo.retrieve()
            #ori_im = cv2.resize(ori_im, (self.im_width, self.im_height))
            #im = ori_im[ymin:ymax, xmin:xmax, (2, 1, 0)]
            im = Image.fromarray(ori_im)
            img_size = np.asarray(ori_im.shape)[0:2]
            bbox_xywh, landmarks = detector.detect_faces(im,min_face_size=30)
            if bbox_xywh is not None:
                cls_conf = [i[4] for i in bbox_xywh]
                bbox_xywh = [ [(i[0]+i[2])/2,(i[1]+i[3])/2,i[2]-i[0],i[3]-i[1]] for i in bbox_xywh]
                outputs = deepsort.update(bbox_xywh, cls_conf, ori_im)
                if len(outputs) > 0:
                    bbox_xyxy = outputs[:, :4]
                    identities = outputs[:, -1]
                    #self.save_sequence(ori_im,bbox_xyxy,identities,img_size)
                    ori_im = draw_bboxes(ori_im, bbox_xyxy, identities, offset=(xmin, ymin))
                    '''
                    cv2.imshow('',ori_im)
                    if (cv2.waitKey() & 0xFF==ord('q')):
                        cv2.destroyAllWindows()
                        break
                    '''
            if self.write_video:
                self.output.write(ori_im)


if __name__ == "__main__":
    path = '../record.mp4'
    video_name = 'V-record'
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
