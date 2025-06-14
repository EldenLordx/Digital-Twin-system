import copy
import os
import cv2
import time
from config import *
from multiprocessing import Process
from persondetect.detect_task import DetectTask
from mot.person_mot_task import PersonMOTTask
# from segmentation.HumanParsing.seg_online import SegmentationTask


#  输出图像数组、摄像头通道号、视频名称
def saveVideo(output_arr, video_name, path):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # avc1 or mp4v
    if not os.path.exists(path):
        os.makedirs(path)
    output_path = os.path.join(path,  video_name + ".mp4")
    img_height, img_width, _ = output_arr[0].shape

    output = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))
    print('save', len(output_arr))
    for img in output_arr:
        # print(img.shape)
        output.write(img)
    output.release()


def deleteOneFile(file_path, file_name):
    if os.path.exists(os.path.join(file_path, file_name)):
            os.remove(os.path.join(file_path, file_name))


class Worker(Process):
    # 摄像头ID  使用的GPU的ID  是否保存结果视频
    def __init__(self, camera_id, gpu_id, write_flag=False):
        Process.__init__(self)
        self.camera_id = camera_id
        self.gpu_id = gpu_id
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu_id)
        self.write_flag = write_flag

        self.origin_path = origin_path + str(camera_id) + '/'  # 原始视频目录
        self.face_detect_img_path = face_detect_img_path + str(camera_id) + '/'  # 行人图片存储目录
        self.person_detect_img_path = person_detect_img_path + str(camera_id) + '/'  # 行人图片存储目录

        self.face_detect_txt_path = face_detect_txt_path + str(camera_id) + '/'
        self.person_detect_txt_path = person_detect_txt_path + str(camera_id) + '/'
        self.face_mot_txt_path = face_mot_txt_path + str(camera_id) + '/'
        self.person_mot_txt_path = person_mot_txt_path + str(camera_id) + '/'

        self.face_detect_video_path = face_detect_video_path + str(camera_id) + '/'
        self.person_detect_video_path = person_detect_video_path + str(camera_id) + '/'
        self.face_mot_video_path = face_mot_video_path + str(camera_id) + '/'
        self.person_mot_video_path = person_mot_video_path + str(camera_id) + '/'
        self.segmentation_video_path = segmentation_video_path + str(camera_id) + '/'


    # 删除最早的结果视频、txt文件
    def deleteFile(self, video_list):
        # 删除最早的结果视频
        delete_video_name = video_list[0]
        deleteOneFile(self.face_detect_video_path, delete_video_name)
        deleteOneFile(self.person_detect_video_path, delete_video_name)
        deleteOneFile(self.face_mot_video_path, delete_video_name)
        deleteOneFile(self.person_mot_video_path, delete_video_name)
        deleteOneFile(self.segmentation_video_path, delete_video_name)

        deleteOneFile(self.face_detect_txt_path, delete_video_name)
        deleteOneFile(self.person_detect_txt_path, delete_video_name)
        deleteOneFile(self.face_mot_txt_path, delete_video_name)
        deleteOneFile(self.person_mot_txt_path, delete_video_name)


        # 删除最早的行人检测txt
        if os.path.exists(os.path.join(self.person_detect_txt_path, delete_video_name + ".txt")):
            os.remove(os.path.join(self.person_detect_txt_path, delete_video_name + ".txt"))

        # 删除最早的行人检测图片
        # person_imgs = os.listdir(self.person_img_path)
        # for i in range(len(person_imgs)):
        #     if person_imgs[i].__contains__(delete_video_name):
        #         os.remove(os.path.join(self.person_img_path, person_imgs[i]))



    # 拼接输入图像数组
    def getInputArray(self, videoPath):
        self.vdo.open(videoPath)
        input_arr = []
        ret, frame = self.vdo.read()

        while ret:
            # hbc0608  for batch loader
            # frame = cv2.resize(frame, (416, 416))
            # w, h, c = frame.shape
            # frame = cv2.resize(frame, (int(w//2), int(h//2)))
            # cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            input_arr.append(frame)
            ret, frame = self.vdo.read()
        return input_arr[:batch_frames]

    def run(self):
        # opencv读取视频
        self.vdo = cv2.VideoCapture()
        # 相关任务启动
        detect_task = DetectTask(self.camera_id, self.gpu_id)

        if person_mot_switch:
            person_mot_task = PersonMOTTask(self.camera_id, self.gpu_id)
        if segmentation_switch:
            seg_task = SegmentationTask()

        while True:
            origin_list = os.listdir(self.origin_path)  # 原始视频数量的控制由摄像头接入部分控制
            origin_list.sort()
            if len(origin_list) < 2:  # 没有视频文件或者当前正在生成第一个视频文件
                print("waiting")
                time.sleep(0.1)  # 避免频繁时间片轮换
                continue

            # if os.path.exists(self.person_mot_video_path):
            #     video_list = os.listdir(self.person_mot_video_path)  # 结果视频存储目录
            #     video_list.sort()
            #     if len(video_list) >= max_video:
            #         self.deleteFile(video_list)  # 达到最大数量后，删除最早的相关结果文件

            if os.path.exists(self.person_mot_txt_path):
                txt_list = os.listdir(self.person_mot_txt_path)  # 结果视频存储目录
                txt_list.sort()
                if len(txt_list) >= max_video:
                    self.deleteFile(txt_list)  # 达到最大数量后，删除最早的相关结果文件

            if os.path.exists(self.person_detect_txt_path):
                txt_list = os.listdir(self.person_detect_txt_path)  # 结果视频存储目录
                txt_list.sort()
                if len(txt_list) >= max_video:
                    self.deleteFile(txt_list)  # 达到最大数量后，删除最早的相关结果文件

            # Select the latest video
            path = origin_list[-2]
            path = os.path.join(self.origin_path, path)  # /home/mmap/cameraVideos/origin/17/xxx.mp4
            video_name = path.split('/')[-1][0:-4]

            # Determine if the current video has been processed
            # det_txt_path = os.path.join(self.detect_result_path, video_name + '_det_output.txt')
            if not os.path.exists(self.person_detect_txt_path):
                os.makedirs(self.person_detect_txt_path)

            det_txt_path = self.person_detect_txt_path + video_name + '.txt'
            det_imgs_path = self.person_detect_img_path
            if os.path.exists(det_txt_path):
                # print(det_txt_path + " this video has handle!")
                time.sleep(0.1)  # 避免频繁时间片轮换
            else:
                start = time.time()
                print("============= Getting the input array =================")
                input_arr = self.getInputArray(path)
                if person_mot_switch:
                    input_arr_person_mot = copy.deepcopy(input_arr)
                if segmentation_switch:
                    input_arr_segmentation = copy.deepcopy(input_arr)
                print("============= The input array has got! =================" )
                print("Getting the input array takes ", str(time.time() - start), " seconds!")

                start_person_detect = time.time()
                detect_task.work(input_arr, path, det_txt_path, det_imgs_path)
                end_person_detect = time.time()
                print("The person detect process takes ", str(end_person_detect - start_person_detect), " seconds!")

                if person_mot_switch:
                    start_person_mot = time.time()
                    output_arr_person_mot = person_mot_task.work(input_arr_person_mot, det_txt_path)
                    end_person_mot = time.time()
                    print("The person mot process takes ", str(end_person_mot - start_person_mot), " seconds!")
                if segmentation_switch:
                    seg_task.setVideoName(video_name)
                    output_arr_segmentation = seg_task.segmentation(input_arr_segmentation, det_txt_path)
                print("task has done!")

                if self.write_flag:
                    print("================ Writting Videos! ===========")
                    if segmentation_switch:
                        saveVideo(output_arr_segmentation,  video_name, self.segmentation_video_path)
                    if person_mot_switch:
                        saveVideo(output_arr_person_mot, video_name, self.person_mot_video_path)

                end = time.time()
                print("The entire process takes ", str(end - start), " seconds!")

                # hbc0607  ##hp
                # with open(det_txt_path, 'a+') as f:
                #    res = str(end - start) + '\n'
                #    f.write(res)


if __name__ == "__main__":
    print("workers begin working!")
    # worker1 = Worker(17, 1)
    # worker1.start()

    # worker2 = Worker(18, 0)
    # worker2.start()

    # worker3 = Worker(23, 0)
    # worker3.start()

    # worker4 = Worker(25, 0)
    # worker4.start()

    # worker5 = Worker(19, 0)
    # worker5.start()

    # worker6 = Worker(20, 0)
    # worker6.start()

    # worker7 = Worker(21, 1)
    # worker7.start()

    worker8 = Worker(24, 0)
    worker8.start()

    worker9 = Worker(25, 1)
    worker9.start()

    # worker10 = Worker(36, 1)
    # worker10.start()


    worker10 = Worker(27, 0)
    worker10.start()

    worker11 = Worker(28, 1)
    worker11.start()
