from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QSize, QFile, Qt, QEvent, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QBrush, QColor
from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QTreeWidgetItem
from PyQt5 import QtCore
from OfflineWindow import OfflineWindow
from visionalgmain import Ui_VisionAlgMain
import numpy as np

from config import *
import sys
import cv2
from PIL import Image, ImageDraw, ImageFont

# *** 2021.6.1 add new function to display the name result of face recongnition *** #
# *** waiting to test *** #


# from segmentation.HumanParsing.seg_offline import OffLineWindow

handling_camera = ['17', '18', '19', '20', '21', '23', '24', '25', '27', '28', '35', '36']

# drawing box data
COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]

'''
    主窗口程序 
'''


class MyMainWindow(QMainWindow, Ui_VisionAlgMain):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.initUI()
        self.initData()
        self.initVideoLab()
        self.initEvent()
        self.labelIndex = 0
        self.vdo0 = cv2.VideoCapture()
        self.vdo1 = cv2.VideoCapture()
        self.vdo2 = cv2.VideoCapture()
        self.vdo3 = cv2.VideoCapture()
        self.timer0 = QTimer(self)
        self.timer1 = QTimer(self)
        self.timer2 = QTimer(self)
        self.timer3 = QTimer(self)
        self.video_name0 = ''
        self.video_name1 = ''
        self.video_name2 = ''
        self.video_name3 = ''
        self.mot_message0 = {}
        self.mot_message1 = {}
        self.mot_message2 = {}
        self.mot_message3 = {}

    def initUI(self):
        self.move(QPoint(0, 0))
        self.setStyleSheet("QGroupBox#gboxMain{border-width:0px;}")
        file = QFile("client/icons/silvery.qss")
        if file.open(QFile.ReadOnly):
            qss = str(file.readAll(), encoding='utf-8')
            self.setStyleSheet(qss)
            file.close()
        self.setProperty("Form", True)
        self.resize(QSize(1280, 720))
        self.widget_menu.move(0, 0)
        self.widget_show.move(64, 0)
        self.widget_menu.setStyleSheet("background-color:#3C3C3C;")
        self.DVRsets_treeView.setHeaderLabels(['ChannelNo', 'Name'])
        self.DVRsets_treeView.setColumnWidth(0, 90)

    def initData(self):
        self.cameraInfo = []  # 摄像头信息  camera.txt配置
        self.VideoLab = []  # 播放label
        self.VideoLay = []  # 播放label布局
        self.tempLab = 0  # 当前选中的label
        self.windowNum = 4  # 屏幕数量
        self.maxWindowNum = 16  # 最多屏幕数量
        self.timer = []

        # self.mot_message = []       # 视频检测框信息

        self.last_file_path0 = None
        for i in range(self.maxWindowNum):
            self.timer.append(QTimer(self))  # 控制播放的QTimer
            # self.mot_message.append({})

    def initVideoLab(self):
        self.VideoLab.append(self.labVideo1)
        self.VideoLab.append(self.labVideo2)
        self.VideoLab.append(self.labVideo3)
        self.VideoLab.append(self.labVideo4)
        self.VideoLab.append(self.labVideo5)
        self.VideoLab.append(self.labVideo6)
        self.VideoLab.append(self.labVideo7)
        self.VideoLab.append(self.labVideo8)
        self.VideoLab.append(self.labVideo9)
        self.VideoLab.append(self.labVideo10)
        self.VideoLab.append(self.labVideo11)
        self.VideoLab.append(self.labVideo12)
        self.VideoLab.append(self.labVideo13)
        self.VideoLab.append(self.labVideo14)
        self.VideoLab.append(self.labVideo15)
        self.VideoLab.append(self.labVideo16)

        self.VideoLay.append(self.lay1)
        self.VideoLay.append(self.lay2)
        self.VideoLay.append(self.lay3)
        self.VideoLay.append(self.lay4)

        for i in range(16):
            self.VideoLab[i].setProperty("labVideo", True)
            self.VideoLab[i].setText("屏幕{}".format(i + 1))
        self.tempLab = self.VideoLab[0]
        self.show_video_4()  # 初始化时默认显示4个画面

    def change_window_menu(self, pos):  # 处理在视频显示区的右键菜单显示
        menu = QMenu()
        opt1 = menu.addAction("切换到1画面")
        opt2 = menu.addAction("切换到4画面")
        opt3 = menu.addAction("切换到9画面")
        opt4 = menu.addAction("切换到16画面")
        action = menu.exec_(self.gBoxMain.mapToGlobal(pos))
        if action == opt1:
            self.show_video_1()  # 显示1个摄像头
        elif action == opt2:
            self.show_video_4()  # 显示4个摄像头
        elif action == opt3:
            self.show_video_9()  # 显示9个摄像头
        elif action == opt4:
            self.show_video_16()  # 显示16个摄像头
        else:
            return

    def initEvent(self):
        self.btn_login.clicked.connect(self.login)
        self.btn_logout.clicked.connect(self.logout)
        self.btn_offline.clicked.connect(self.offlineHandle)
        self.btnMenu_PersonParse.clicked.connect(self.parseHandle)
        self.gBoxMain.setContextMenuPolicy(Qt.CustomContextMenu)  # 针对gBoxMain开放右键，gBoxMain是视频显示部分(1,4,9,16)的那部分
        self.gBoxMain.customContextMenuRequested.connect(
            self.change_window_menu)  # 当在gBoxMain区域内点击右键时，调用用户自定义的函数 custom_right_menu
        self.quitoffline = pyqtSignal(object)
        # self.btnMenu_Detect.setMouseTracking(True)
        # self.btnMenu_Detect.installEventFilter(self)
        # self.btnMenu_PersonParse.installEventFilter(self)

        # self.btnMenu_Detect.mouseMoveEvent(event)
        # self.btnMenu_Detect.clicked.connect(self.DetectFuncSelect)

    # def eventFilter(self, object, event):
    #     if event.type() == QEvent.Enter and object == self.btnMenu_Detect:
    #         self.DetectFuncSelect()
    #         # print(object.pos())
    #         # print(self.btnMenu_Detect.pos())
    #         return True
    #     elif event.type() == QEvent.Enter and object.pos() == self.btnMenu_PersonParse.pos():
    #         self.PersonParseFuncSelect()
    #         # print(object.pos())
    #         # print(self.btnMenu_Detect.pos())
    #         return True
    #     elif event.type() == QEvent.Leave:
    #         self.show()
    #     return False

    def parseHandle(self):
        parseWin = OfflineWindow(self.widget_show)
        parseWin.setAttribute(Qt.WA_DeleteOnClose)
        # self.widget_alg.hide()
        # self.widget_main.hide()
        parseWin.show()
        parseWin.move(0, 0)  # 这一行去掉了还能用，搞清楚它是干嘛的

    def login(self):
        self.root = QTreeWidgetItem(self.DVRsets_treeView)
        self.root.setText(0, 'NERCMS')
        self.root.setIcon(0, QIcon("icons/login.bmp"))

        # set child node
        self.parseCameraInfo()
        for i in range(len(self.cameraInfo)):
            child = QTreeWidgetItem(self.root)
            channelNo = self.cameraInfo[i][0]
            child.setText(0, channelNo)
            name = self.cameraInfo[i][2].strip("\n")
            child.setText(1, name)
            child.setIcon(0, QIcon("icons/camera.bmp"))
            if handling_camera.__contains__(channelNo):
                child.setBackground(0, QBrush(QColor("#32CD99")))
                child.setBackground(1, QBrush(QColor("#32CD99")))
        self.DVRsets_treeView.addTopLevelItem(self.root)
        self.DVRsets_treeView.expandAll()
        self.DVRsets_treeView.clicked.connect(self.onTreeClick)
        self.DVRsets_treeView.itemDoubleClicked.connect(self.onTreeDoubleClick)

        self.video_no = []

    def onTreeClick(self, index):
        item = self.DVRsets_treeView.currentItem()
        if item.text(0) == 'NERCMS':
            return
        channelNo = int(item.text(0))
        # print(item.text(0), " ", item.text(1))

    def onTreeDoubleClick(self, item, column):
        if item.text(0) == 'NERCMS':
            return
        channelNo = int(item.text(0))
        mot_path = os.path.join(root_path, 'camera/origin/' + str(channelNo))
        listdir = os.listdir(mot_path)
        listdir.sort()
        # filepath = os.path.join(mot_path, listdir[0])
        # filepath = os.path.join(mot_path, listdir[-1])
        filepath = os.path.join(mot_path, listdir[-3])
        print(filepath)
        # filepath = os.path.join(mot_path, '1620892963.mp4')

        # path = "/home/mmap/cameraVideos/mot_videos/"+str(channelNo)+"/0.mp4"

        if self.labelIndex == 0:
            self.last_file_path0 = filepath
            self.vdo0.open(filepath)
            self.video_name0 = filepath[-14: -4]
            self.get_mot_message0(channelNo)  # get mot box information
            self.timer0.timeout.connect(lambda: self.play0(channelNo))
            self.timer0.start(50)
        elif self.labelIndex == 1:
            self.last_file_path1 = filepath
            self.vdo1.open(filepath)
            self.video_name1 = filepath[-14: -4]
            self.get_mot_message1(channelNo)  # get mot box information
            self.timer1.timeout.connect(lambda: self.play1(channelNo))
            self.timer1.start(50)
        elif self.labelIndex == 2:
            self.last_file_path2 = filepath
            self.vdo2.open(filepath)
            self.video_name2 = filepath[-14: -4]
            self.get_mot_message2(channelNo)  # get mot box information
            self.timer2.timeout.connect(lambda: self.play2(channelNo))
            self.timer2.start(50)
        elif self.labelIndex == 3:
            self.last_file_path3 = filepath
            self.vdo3.open(filepath)
            self.video_name3 = filepath[-14: -4]
            self.get_mot_message3(channelNo)  # get mot box information
            self.timer3.timeout.connect(lambda: self.play3(channelNo))
            self.timer3.start(50)
        self.video_no.append(0)
        self.labelIndex += 1
        print(channelNo)

    def logout(self):
        self.DVRsets_treeView.clear()
        self.cameraInfo.clear()

    def iou(self, bbox1, bbox2):
        """
        Calculates the intersection-over-union of two bounding boxes.

        Args:
            bbox1 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.
            bbox2 (numpy.array, list of floats): bounding box in format x1,y1,x2,y2.

        Returns:
            int: intersection-over-onion of bbox1, bbox2
        """

        bbox1 = [float(x) for x in bbox1]
        bbox2 = [float(x) for x in bbox2]

        (x0_1, y0_1, x1_1, y1_1) = bbox1
        (x0_2, y0_2, x1_2, y1_2) = bbox2

        # get the overlap rectangle
        overlap_x0 = max(x0_1, x0_2)
        overlap_y0 = max(y0_1, y0_2)
        overlap_x1 = min(x1_1, x1_2)
        overlap_y1 = min(y1_1, y1_2)

        # check if there is an overlap
        if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
            return 0

        # if yes, calculate the ratio of the overlap to each ROI size and the unified size
        size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
        size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
        size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
        size_union = size_1 + size_2 - size_intersection

        return size_intersection / size_1

    # get mot_message
    def get_mot_message0(self, channelNo):
        self.mot_message0 = {}  #
        mot0_txt_path = os.path.join(root_path, 'camera/txt_results/person_mot' + str(
            channelNo) + '/' + self.video_name0 + '.txt')  #
        detect0_txt_path = os.path.join(face_mot_txt_path + str(
            channelNo), self.video_name0 + '.txt')  #
        frame_to_name = {}
        # for test 2021/6/7
        print(mot0_txt_path)
        frame_to_name = self.interpolation(detect0_txt_path)
        id_to_name = {}
        # self.interpolation2(mot1_txt_path, 1)
        if os.path.exists(mot0_txt_path):  ##
            # f = open(mot0_txt_path)                    ##
            # print("person mot exist and open it!")
            # lines = f.readlines()
            lines = self.interpolation2(mot0_txt_path, frame_skip)
            this_frame_num = 0
            for line in lines:
                frame_num = int(line[0])
                for n in line[1]:
                    [x1, y1, x2, y2] = [int(x) for x in n[0]]
                    bbox_array = np.array([x1, y1, x2, y2])
                    id = n[1]
                    if frame_num == this_frame_num:
                        if frame_num not in frame_to_name.keys():
                            if id in id_to_name.keys():
                                name = id_to_name[id]
                            else:
                                # name = '未识别'
                                name = str(id)
                            self.mot_message0[str(frame_num)].append([bbox_array, name])
                        else:
                            for frame in frame_to_name[frame_num]:
                                if self.iou(frame_to_name[frame_num][0][1], bbox_array) >= 0.8:
                                    name = frame[0]
                                    id_to_name[id] = name
                                    # if id not in id_to_name.keys():
                                    #     id_to_name[id] = name
                                    # if id in id_to_name.keys():
                                    #     name = id_to_name[id]
                                else:
                                    name = str(id)
                                self.mot_message0[str(frame_num)].append([bbox_array, name])
                        # name = given_name
                    else:
                        # bbox_array = np.array([x1, y1, x2, y2])
                        if frame_num not in frame_to_name.keys():
                            if id in id_to_name.keys():
                                name = id_to_name[id]
                            else:
                                name = str(id)
                        elif self.iou(frame_to_name[frame_num][0][1], bbox_array) >= 0.8:
                            name = frame_to_name[frame_num][0][0]
                            id_to_name[id] = name
                            # if id not in id_to_name.keys():
                            #     id_to_name[id] = name
                            # else:
                            #     name = id_to_name[id]
                        else:
                            name = str(id)
                        # name = given_name
                        self.mot_message0[str(frame_num)] = [[bbox_array, name]]
                        this_frame_num = frame_num
        else:
            print("person mot not found")

    def get_mot_message1(self, channelNo):
        self.mot_message1 = {}  #
        mot1_txt_path = os.path.join(root_path, 'camera/txt_results/person_mot' + str(
            channelNo) + '/' + self.video_name1 + '.txt')  #
        detect1_txt_path = os.path.join(face_mot_txt_path + str(
            channelNo), self.video_name1 + '.txt')  #
        frame_to_name = {}
        # for test 2021/6/7
        print(mot1_txt_path)
        frame_to_name = self.interpolation(detect1_txt_path)
        id_to_name = {}
        # self.interpolation2(mot1_txt_path, 1)
        if os.path.exists(mot1_txt_path):  ##
            # f = open(mot0_txt_path)                    ##
            # print("person mot exist and open it!")
            # lines = f.readlines()
            lines = self.interpolation2(mot1_txt_path, frame_skip)
            this_frame_num = 0
            for line in lines:
                frame_num = int(line[0])
                for n in line[1]:
                    [x1, y1, x2, y2] = [int(x) for x in n[0]]
                    bbox_array = np.array([x1, y1, x2, y2])
                    id = n[1]
                    if frame_num == this_frame_num:
                        if frame_num not in frame_to_name.keys():
                            if id in id_to_name.keys():
                                name = id_to_name[id]
                            else:
                                name=str(id)
                            self.mot_message1[str(frame_num)].append([bbox_array, name])
                        else:
                            for frame in frame_to_name[frame_num]:
                                if self.iou(frame_to_name[frame_num][0][1], bbox_array) >= 0.8:
                                    name = frame[0]
                                    id_to_name[id] = name
                                    # if id not in id_to_name.keys():
                                    #     id_to_name[id] = name
                                    # else:
                                    #     name = id_to_name[id]
                                else:
                                    name = str(id)
                                self.mot_message1[str(frame_num)].append([bbox_array, name])
                        # name = given_name
                    else:
                        bbox_array = np.array([x1, y1, x2, y2])
                        if frame_num not in frame_to_name.keys():
                            if id in id_to_name.keys():
                                name = id_to_name[id]
                            else:
                                name = str(id)
                        elif self.iou(frame_to_name[frame_num][0][1], bbox_array) >= 0.8:
                            name = frame_to_name[frame_num][0][0]
                            id_to_name[id] = name
                            # if id not in id_to_name.keys():
                            #     id_to_name[id] = name
                            # else:
                            #     name = id_to_name[id]
                        else:
                            name = str(id)
                        # name = given_name
                        self.mot_message1[str(frame_num)] = [[bbox_array, name]]
                        this_frame_num = frame_num
        else:
            print("person mot not found")

    def get_mot_message2(self, channelNo):
        self.mot_message2 = {}  #
        mot2_txt_path = os.path.join(root_path, 'camera/txt_results/person_mot' + str(
            channelNo) + '/' + self.video_name2 + '.txt')  #
        detect2_txt_path = os.path.join(face_mot_txt_path + str(
            channelNo), self.video_name2 + '.txt')  #
        frame_to_name = {}
        # for test 2021/6/7
        print(mot2_txt_path)
        frame_to_name = self.interpolation(detect2_txt_path)
        id_to_name = {}
        # self.interpolation2(mot1_txt_path, 1)
        if os.path.exists(mot2_txt_path):  ##
            # f = open(mot0_txt_path)                    ##
            # print("person mot exist and open it!")
            # lines = f.readlines()
            lines = self.interpolation2(mot2_txt_path, frame_skip)
            this_frame_num = 0
            for line in lines:
                frame_num = int(line[0])
                for n in line[1]:
                    [x1, y1, x2, y2] = [int(x) for x in n[0]]
                    bbox_array = np.array([x1, y1, x2, y2])
                    id = n[1]
                    if frame_num == this_frame_num:
                        if frame_num not in frame_to_name.keys():
                            if id in id_to_name.keys():
                                name = id_to_name[id]
                            else:
                                name = str(id)
                            self.mot_message2[str(frame_num)].append([bbox_array, name])
                        else:
                            for frame in frame_to_name[frame_num]:
                                if self.iou(frame_to_name[frame_num][0][1], bbox_array) >= 0.8:
                                    name = frame[0]
                                    id_to_name[id] = name
                                    # if id not in id_to_name.keys():
                                    #     id_to_name[id] = name
                                    # else:
                                    #     name = id_to_name[id]
                                else:
                                    name = str(id)
                                self.mot_message2[str(frame_num)].append([bbox_array, name])
                        # name = given_name
                    else:
                        bbox_array = np.array([x1, y1, x2, y2])
                        if frame_num not in frame_to_name.keys():
                            if id in id_to_name.keys():
                                name = id_to_name[id]
                            else:
                                name = str(id)
                        elif self.iou(frame_to_name[frame_num][0][1], bbox_array) >= 0.8:
                            name = frame_to_name[frame_num][0][0]
                            id_to_name[id] = name
                            # if id not in id_to_name.keys():
                            #     id_to_name[id] = name
                            # else:
                            #     name = id_to_name[id]
                        else:
                            name = str(id)
                        # name = given_name
                        self.mot_message2[str(frame_num)] = [[bbox_array, name]]
                        this_frame_num = frame_num
        else:
            print("person mot not found")

    def get_mot_message3(self, channelNo):
        self.mot_message3 = {}  #
        mot3_txt_path = os.path.join(root_path, 'camera/txt_results/person_mot' + str(
            channelNo) + '/' + self.video_name3 + '.txt')  #
        detect3_txt_path = os.path.join(face_mot_txt_path + str(
            channelNo), self.video_name3 + '.txt')  #
        frame_to_name = {}
        # for test 2021/6/7
        print(mot3_txt_path)
        frame_to_name = self.interpolation(detect3_txt_path)
        id_to_name = {}
        # self.interpolation2(mot1_txt_path, 1)
        if os.path.exists(mot3_txt_path):  ##
            # f = open(mot0_txt_path)                    ##
            # print("person mot exist and open it!")
            # lines = f.readlines()
            lines = self.interpolation2(mot3_txt_path, frame_skip)
            this_frame_num = 0
            for line in lines:
                frame_num = int(line[0])
                for n in line[1]:
                    [x1, y1, x2, y2] = [int(x) for x in n[0]]
                    bbox_array = np.array([x1, y1, x2, y2])
                    id = n[1]
                    if frame_num == this_frame_num:
                        if frame_num not in frame_to_name.keys():
                            if id in id_to_name.keys():
                                name = id_to_name[id]
                            else:
                                name = str(id)
                            self.mot_message3[str(frame_num)].append([bbox_array, name])
                        else:
                            for frame in frame_to_name[frame_num]:
                                if self.iou(frame_to_name[frame_num][0][1], bbox_array) >= 0.8:
                                    name = frame[0]
                                    id_to_name[id] = name
                                    # if id not in id_to_name.keys():
                                    #     id_to_name[id] = name
                                    # else:
                                    #     name = id_to_name[id]
                                else:
                                    name = str(id)
                                self.mot_message3[str(frame_num)].append([bbox_array, name])
                        # name = given_name
                    else:
                        bbox_array = np.array([x1, y1, x2, y2])
                        if frame_num not in frame_to_name.keys():
                            if id in id_to_name.keys():
                                name = id_to_name[id]
                            else:
                                name = str(id)
                        elif self.iou(frame_to_name[frame_num][0][1], bbox_array) >= 0.8:
                            name = frame_to_name[frame_num][0][0]
                            id_to_name[id] = name
                            # if id not in id_to_name.keys():
                            #     id_to_name[id] = name
                            # else:
                            #     name = id_to_name[id]
                        else:
                            name = str(id)
                        # name = given_name
                        self.mot_message3[str(frame_num)] = [[bbox_array, name]]
                        this_frame_num = frame_num
        else:
            print("person mot not found")

    # drawing frame.
    def draw_bboxes(self, img, bbox, identities=None, offset=(0, 0)):
        for i, box in enumerate(bbox):
            # print(box)
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            name = identities[i] if identities is not None else 'None'

            # get the color from the first character in name string
            c = ord(name[0])
            color = COLORS_10[c % len(COLORS_10)]

            label = '{} {}'.format("object", name)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(img, (x1 - 1, y1 - t_size[1] - 3), (x1 + t_size[0] + 2, y1 + 2), color, -1)
            # too small. bigger?
            # cv2.putText(img, label, (x1, y1), cv2.FONT_HERSHEY_PLAIN, 3, [255, 255, 255], 3)
            img = self.cvputChinese(img, label, x1, y1)
        return img

    def cvputChinese(self, img, label, x1, y1):
        fontpath = "simsun.ttc"
        # 设置字体的颜色
        b, g, r, a = 255, 255, 255, 3
        # 设置字体大小
        font = ImageFont.truetype(fontpath, 50)
        # 将numpy array的图片格式转为PIL的图片格式
        img_pil = Image.fromarray(img)
        # 创建画板
        draw = ImageDraw.Draw(img_pil)
        # 在图片上绘制中文
        draw.text((x1, y1 - 50), label, font=font, fill=(b, g, r, 10))
        # 将图片转为numpy array的数据格式
        img = np.array(img_pil)
        return img

    def play0(self, channelNo):  # drawing and showing in the same time.
        self.frame = self.vdo0.get(1)
        ret, frame = self.vdo0.read()
        '''
        mot_message = {}  # dictionary is not enough
        mot_txt_file_path = os.path.join(root_path, 'camera/txt_results/person_mot' + str(channelNo) + '/' + self.video_name0 + '.txt')
        print(mot_txt_file_path)
        print(self.frame)
        print(os.path.exists(mot_txt_file_path))
        if os.path.exists(mot_txt_file_path):
            f = open(mot_txt_file_path)
            lines = f.readlines()
            this_frame_num = 0
            for line in lines:
                frame_num, x1, y1, x2, y2, identity = line.split()
                if frame_num == this_frame_num:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num].append([bbox_array, identity])
                else:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num] = [[bbox_array, identity]]
                    this_frame_num = frame_num
        '''
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box_all = []
            id_all = []
            if self.mot_message0.__contains__(str(int(self.frame))):
                for one in self.mot_message0[str(int(self.frame))]:
                    box_all.append(one[0])
                    id_all.append(one[1])
                bbox_xyxy = np.array(box_all)
                identities = np.array(id_all)
                frame = self.draw_bboxes(frame, bbox_xyxy, identities)
                frame = cv2.resize(frame, (465, 300))
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                qimg = QtGui.QPixmap.fromImage(qimg)
                self.labVideo1.setPixmap(qimg)
                self.labVideo1.setScaledContents(True)
            else:
                frame = cv2.resize(frame, (465, 300))
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                qimg = QtGui.QPixmap.fromImage(qimg)
                self.labVideo1.setPixmap(qimg)
                self.labVideo1.setScaledContents(True)
        else:
            mot_path = os.path.join(root_path, 'camera/origin/' + str(channelNo))
            listdir = os.listdir(mot_path)
            listdir.sort()
            filepath = os.path.join(mot_path, listdir[-3])
            if filepath != self.last_file_path0:
                self.vdo0 = cv2.VideoCapture(filepath)
                self.last_file_path0 = filepath
                self.video_name0 = filepath[-14: -4]
                self.get_mot_message0(channelNo)

    def play1(self, channelNo):
        # print(self.frame)
        self.vdo1.set(1, self.frame)
        ret, frame = self.vdo1.read()
        '''
        mot_message = {}  # dictionary is not enough
        mot_txt_file_path = os.path.join(root_path, 'camera/txt_results/person_mot' + str(
            channelNo) + '/' + self.video_name1 + '.txt')
        if os.path.exists(mot_txt_file_path):
            f = open(mot_txt_file_path)
            lines = f.readlines()
            this_frame_num = 0
            for line in lines:
                frame_num, x1, y1, x2, y2, identity = line.split()
                if frame_num == this_frame_num:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num].append([bbox_array, identity])
                else:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num] = [[bbox_array, identity]]
                    this_frame_num = frame_num
        '''
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box_all = []
            id_all = []
            if self.mot_message1.__contains__(str(int(self.frame))):
                for one in self.mot_message1[str(int(self.frame))]:
                    box_all.append(one[0])
                    id_all.append(one[1])
                bbox_xyxy = np.array(box_all)
                identities = np.array(id_all)
                frame = self.draw_bboxes(frame, bbox_xyxy, identities)
                frame = cv2.resize(frame, (465, 300))
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                qimg = QtGui.QPixmap.fromImage(qimg)
                self.labVideo2.setPixmap(qimg)
                self.labVideo2.setScaledContents(True)
            else:
                frame = cv2.resize(frame, (465, 300))
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                qimg = QtGui.QPixmap.fromImage(qimg)
                self.labVideo2.setPixmap(qimg)
                self.labVideo2.setScaledContents(True)

        else:
            mot_path = os.path.join(root_path, 'camera/origin/' + str(channelNo))
            listdir = os.listdir(mot_path)
            listdir.sort()
            filepath = os.path.join(mot_path, listdir[-3])
            if filepath != self.last_file_path1:
                self.vdo1 = cv2.VideoCapture(filepath)
                self.last_file_path1 = filepath
                self.video_name1 = filepath[-14: -4]
                self.get_mot_message1(channelNo)

    def play2(self, channelNo):
        self.vdo2.set(1, self.frame)
        ret, frame = self.vdo2.read()
        '''
        mot_message = {}  # dictionary is not enough
        mot_txt_file_path = os.path.join(root_path, 'camera/txt_results/person_mot' + str(
            channelNo) + '/' + self.video_name2 + '.txt')
        if os.path.exists(mot_txt_file_path):
            f = open(mot_txt_file_path)
            lines = f.readlines()
            this_frame_num = 0
            for line in lines:
                frame_num, x1, y1, x2, y2, identity = line.split()
                if frame_num == this_frame_num:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num].append([bbox_array, identity])
                else:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num] = [[bbox_array, identity]]
                    this_frame_num = frame_num
        '''
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box_all = []
            id_all = []
            if self.mot_message2.__contains__(str(int(self.frame))):
                for one in self.mot_message2[str(int(self.frame))]:
                    box_all.append(one[0])
                    id_all.append(one[1])
                bbox_xyxy = np.array(box_all)
                identities = np.array(id_all)
                frame = self.draw_bboxes(frame, bbox_xyxy, identities)
                frame = cv2.resize(frame, (465, 300))
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                qimg = QtGui.QPixmap.fromImage(qimg)
                self.labVideo3.setPixmap(qimg)
                self.labVideo3.setScaledContents(True)
            else:
                frame = cv2.resize(frame, (465, 300))
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                qimg = QtGui.QPixmap.fromImage(qimg)
                self.labVideo3.setPixmap(qimg)
                self.labVideo3.setScaledContents(True)
        else:
            mot_path = os.path.join(root_path, 'camera/origin/' + str(channelNo))
            listdir = os.listdir(mot_path)
            listdir.sort()
            filepath = os.path.join(mot_path, listdir[-3])
            if filepath != self.last_file_path2:
                self.vdo2 = cv2.VideoCapture(filepath)
                self.last_file_path2 = filepath
                self.video_name2 = filepath[-14: -4]
                self.get_mot_message2(channelNo)

    def play3(self, channelNo):
        self.vdo3.set(1, self.frame)
        ret, frame = self.vdo3.read()
        '''
        mot_message = {}  # dictionary is not enough
        mot_txt_file_path = os.path.join(root_path, 'camera/txt_results/person_mot' + str(
            channelNo) + '/' + self.video_name3 + '.txt')
        if os.path.exists(mot_txt_file_path):
            f = open(mot_txt_file_path)
            lines = f.readlines()
            this_frame_num = 0
            for line in lines:
                frame_num, x1, y1, x2, y2, identity = line.split()
                if frame_num == this_frame_num:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num].append([bbox_array, identity])
                else:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num] = [[bbox_array, identity]]
                    this_frame_num = frame_num
        '''
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            box_all = []
            id_all = []
            if self.mot_message3.__contains__(str(int(self.frame))):
                for one in self.mot_message3[str(int(self.frame))]:
                    box_all.append(one[0])
                    id_all.append(one[1])
                bbox_xyxy = np.array(box_all)
                identities = np.array(id_all)
                frame = self.draw_bboxes(frame, bbox_xyxy, identities)
                frame = cv2.resize(frame, (465, 300))
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                qimg = QtGui.QPixmap.fromImage(qimg)
                self.labVideo4.setPixmap(qimg)
                self.labVideo4.setScaledContents(True)
            else:
                frame = cv2.resize(frame, (465, 300))
                height, width, channel = frame.shape
                bytesPerLine = 3 * width
                qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
                qimg = QtGui.QPixmap.fromImage(qimg)
                self.labVideo4.setPixmap(qimg)
                self.labVideo4.setScaledContents(True)
        else:
            mot_path = os.path.join(root_path, 'camera/origin/' + str(channelNo))
            listdir = os.listdir(mot_path)
            listdir.sort()
            filepath = os.path.join(mot_path, listdir[-3])
            if filepath != self.last_file_path3:
                self.vdo3 = cv2.VideoCapture(filepath)
                self.last_file_path3 = filepath
                self.video_name3 = filepath[-14: -4]
                self.get_mot_message3(channelNo)

    def parseCameraInfo(self):
        with open("client/ini/camera.txt", errors='ignore', encoding='utf-8') as f:
            for line in f:
                info = line.split(' ')
                self.cameraInfo.append(info)

    def offlineHandle(self):
        self.timer0.disconnect()
        self.timer1.disconnect()
        self.timer2.disconnect()
        self.timer3.disconnect()
        self.myOffline = OfflineWindow(self.widget_show)
        self.myOffline.setAttribute(Qt.WA_DeleteOnClose)
        self.widget_alg.hide()  # 这一行去掉了还能用，搞清楚它是干嘛的
        self.widget_main.hide()  # 这一行去掉了还能用，搞清楚它是干嘛的
        self.myOffline.show()
        self.myOffline.move(0, 0)  # 这一行去掉了还能用，搞清楚它是干嘛的

    def removeLayout(self):
        for i in range(4):
            self.VideoLay[0].removeWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(False)

        for i in range(4, 8):
            self.VideoLay[1].removeWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(False)

        for i in range(8, 12):
            self.VideoLay[2].removeWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(False)

        for i in range(12, 16):
            self.VideoLay[3].removeWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(False)

    def show_video_1(self):  # 显示1个摄像头
        self.removeLayout()
        self.windowNum = 1
        self.video_max = True
        self.change_video_1()

    def change_video_1(self, index=0):
        for i in range((index + 0), (index + 1)):
            self.VideoLay[0].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)

    def show_video_4(self):  # 显示4个摄像头
        self.removeLayout()
        self.windowNum = 4
        self.video_max = False
        self.change_video_4()

    def change_video_4(self, index=0):
        for i in range((index + 0), (index + 2)):
            self.VideoLay[0].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)
        for i in range((index + 2), (index + 4)):
            self.VideoLay[1].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)

    def show_video_9(self):  # 显示9个摄像头
        self.removeLayout()
        self.windowNum = 9
        self.video_max = False
        self.change_video_9()

    def change_video_9(self, index=0):
        for i in range((index + 0), (index + 3)):
            self.VideoLay[0].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)
        for i in range((index + 3), (index + 6)):
            self.VideoLay[1].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)
        for i in range((index + 6), (index + 9)):
            self.VideoLay[2].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)

    def show_video_16(self):  # 显示16个摄像头
        self.removeLayout()
        self.windowNum = 16
        self.video_max = False
        self.change_video_16()

    def change_video_16(self, index=0):
        for i in range((index + 0), (index + 4)):
            self.VideoLay[0].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)
        for i in range((index + 4), (index + 8)):
            self.VideoLay[1].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)
        for i in range((index + 8), (index + 12)):
            self.VideoLay[2].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)
        for i in range((index + 12), (index + 16)):
            self.VideoLay[3].addWidget(self.VideoLab[i])
            self.VideoLab[i].setVisible(True)

    def DetectFuncSelect(self):

        menu = QMenu()
        opt1 = menu.addAction("算法A")
        opt2 = menu.addAction("算法B")

        position = QPoint(self.btnMenu_Detect.geometry().x() + self.btnMenu_Detect.geometry().height() + self.pos().x(),
                          self.btnMenu_Detect.geometry().y() + self.btnMenu_Detect.geometry().width() / 2 + self.pos().y())
        action = menu.exec_(position)
        if action == opt1:
            self.algorithmA()
        elif action == opt2:
            self.algorithmB()

        else:
            return

    def PersonParseFuncSelect(self):

        menu = QMenu()
        opt1 = menu.addAction("算法A")
        opt2 = menu.addAction("算法B")

        position = QPoint(
            self.btnMenu_PersonParse.geometry().x() + self.btnMenu_PersonParse.geometry().height() + self.pos().x(),
            self.btnMenu_PersonParse.geometry().y() + self.btnMenu_PersonParse.geometry().width() / 2 + self.pos().y())
        action = menu.exec_(position)
        if action == opt1:
            self.algorithmA()
        elif action == opt2:
            self.algorithmB()

        else:
            return

    def algorithmA(self):
        print("AAAAAAAAAA")

    def algorithmB(self):
        print("BBBBBBBBBB")

    def interpolation(self, path):  # face detection interpolate
        detect0_txt_path = os.path.join(path)
        if os.path.exists(detect0_txt_path):
            print('face detected!')
            fdetect = open(detect0_txt_path, encoding='utf-8')
            faces = fdetect.readlines()
            finish = {}
            this_frame_num = -1
            for line in faces:
                frame_num, x1, y1, x2, y2, identity = line.split()[:6]
                # x2 = int(x1) + int(x2)
                # y2 = int(y1) + int(y2)
                if frame_num == this_frame_num:
                    bbox_array = [int(x) for x in [x1, y1, x2, y2]]
                    finish[int(frame_num)].append([identity, bbox_array])
                else:
                    bbox_array = [int(x) for x in [x1, y1, x2, y2]]
                    finish[int(frame_num)] = [[identity, bbox_array]]
                    this_frame_num = frame_num

            for i in range(len(finish)):
                if i + 1 < len(finish):
                    key1 = int(list(finish.keys())[i])
                    key2 = int(list(finish.keys())[i + 1])
                    for j in range(key1 + 1, key2):
                        finish[j] = []
                        for mess1 in finish[key1]:
                            for mess2 in finish[key2]:
                                if mess1[0] == mess2[0]:
                                    finish[j].append([mess1[0], mess1[1]])
            for key in list(finish.keys()):
                if not finish.get(key):
                    del finish[key]
            return finish
        else:
            print('face not detected!')
            return {'None': 'None'}

    def interpolation2(self, path, n):  # mot detection interpolate
        n = n + 1
        if os.path.exists(path):  ##
            mot_message = {}
            f = open(path)  ##
            lines = f.readlines()
            this_frame_num = 0
            for line in lines:
                frame_num, x1, y1, x2, y2, identity = line.split()
                if frame_num == this_frame_num:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num].append([bbox_array, identity])
                else:
                    bbox_array = np.array([x1, y1, x2, y2])
                    mot_message[frame_num] = [[bbox_array, identity]]
                    this_frame_num = frame_num

            for i in range(len(mot_message)):
                if i + 1 < len(mot_message):
                    key1 = int(list(mot_message.keys())[i])
                    key2 = int(list(mot_message.keys())[i + 1])
                    if (key1 + n) == key2:
                        for j in range(1, n):
                            mot_message[str(key1 + j)] = []
                            for mess1 in mot_message[str(key1)]:
                                for mess2 in mot_message[str(key2)]:
                                    if mess1[1] == mess2[1]:
                                        box = ['0', '0', '0', '0']
                                        for k in range(4):
                                            box[k] = str(
                                                int(float(mess1[0][k]) * (n - j) / n + float(mess2[0][k]) * j / n))
                                        mot_message[str(key1 + j)].append([box, mess1[1]])
        for key in list(mot_message.keys()):
            if not mot_message.get(key):
                del mot_message[key]
        # with open(path, 'w') as f:
        #     keys = list(int(key) for key in mot_message.keys())
        #     keys.sort()
        #     for key in keys:
        #         key = str(key)
        #         v = mot_message[key]
        #         # if v == []:
        #         #     print("null!!!")
        #         if v != []:
        #             # print(k,v)
        #             for value in v:
        #                 f.write(str(key) + " " + value[0][0] + " " + value[0][1] + " " + value[0][2] + " " + value[0][
        #                     3] + " " + value[1] + '\n')
        mot_message = sorted(mot_message.items(), key=lambda x: int(x[0]))
        return mot_message

    # def saveVideo(output_arr, video_name, path):
    #     fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # avc1 or mp4v
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     output_path = os.path.join(path, video_name + ".mp4")
    #     img_height, img_width, _ = output_arr[0].shape
    #
    #     output = cv2.VideoWriter(output_path, fourcc, fps, (img_width, img_height))
    #     print('save', len(output_arr))
    #     for img in output_arr:
    #         # print(img.shape)
    #         output.write(img)
    #     output.release()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icons/v.ico'))
    myWin = MyMainWindow()
    myWin.setWindowFlags(Qt.Window)
    myWin.show()
    # print(myWin.mapToGlobal())
    sys.exit(app.exec_())
