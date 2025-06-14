import cv2
from multiprocessing import Process
import os
import time
from config import *


demo_rtsp_url = "rtsp://admin:zhongxinhik123@172.16.27.113:554/ch1/main/av_stream"


class HIKCamera(Process):
    def __init__(self, rtsp_url, channelNo, img_width=1280, img_height=720):
        Process.__init__(self)
        self.rtsp_url = rtsp_url
        self.channelNo = channelNo
        self.frame_index = 0
        self.img_width = img_width
        self.img_height = img_height
        self.fps = 20
        self.file_base_path = origin_path + str(self.channelNo) + '/'
        self.fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output = cv2.VideoWriter()
        self.initOutputDir()

    def initOutputDir(self):
        if not os.path.exists(self.file_base_path):
            os.makedirs(self.file_base_path)

    def initOutputFile(self):
        video_list = os.listdir(self.file_base_path)
        video_list.sort()
        if len(video_list) >= max_video:
            delete_video_name = video_list[0]
            os.remove(os.path.join(self.file_base_path, delete_video_name))
        self.file_name = self.file_base_path + '/' + str(int(time.time())) + '.mp4'
        self.output.open(self.file_name, self.fourcc, self.fps, (self.img_width, self.img_height))

    def run(self):
        self.initOutputFile()
        self.capture = cv2.VideoCapture(self.rtsp_url)
        ret, frame = self.capture.read()
        while ret:
            if self.frame_index < batch_frames:
                self.output.write(frame)
                self.frame_index = self.frame_index + 1
            else:
                self.output.release()
                self.frame_index = 0
                self.initOutputFile()
            ret, frame = self.capture.read()


if __name__ == "__main__":
    # camera1 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.113:554/ch1/main/av_stream",
    #                        13)
    # camera1.start()
    #
    # camera2 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.114:554/ch1/main/av_stream",
    #                        14)
    # camera2.start()
    #
    # camera3 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.122:554/ch1/main/av_stream",
    #                        17)
    # camera3.start()
    #
    # camera4 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.123:554/ch1/main/av_stream",
    #                        18)
    # camera4.start()
    #
    # camera5 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.124:554/ch1/main/av_stream",
    #                        19)
    # camera5.start()
    #
    # camera6 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.125:554/ch1/main/av_stream",
    #                        20)
    # camera6.start()
    #
    # camera7 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.126:554/ch1/main/av_stream",
    #                        21)
    # camera7.start()
    #
    # camera8 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.128:554/ch1/main/av_stream",
    #                        23)
    # camera8.start()
    #
    # camera9 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.129:554/ch1/main/av_stream",
    #                        24)
    # camera9.start()
    #
    # camera10 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.130:554/ch1/main/av_stream",
    #                        25)
    # camera10.start()

    camera11 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.132:554/ch1/main/av_stream",
                           27)
    camera11.start()

    camera12 = HIKCamera("rtsp://admin:zhongxinhik123@172.16.27.133:554/ch1/main/av_stream",
                           28)
    camera12.start()


# import subprocess

# subprocess.call(['ffmpeg', '-i', 'output/24/1556333721.mp4', '-vcodec', 'libx264', 'hh.mp4'])