from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import QSize, QFile, Qt, QEvent, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import QIcon, QBrush, QColor
from PyQt5.QtWidgets import QMainWindow, QApplication, QMenu, QTreeWidgetItem
from PyQt5 import QtCore
from OfflineWindow import OfflineWindow
from visionalgmain import Ui_VisionAlgMain

from config import *
import sys
import cv2

# from segmentation.HumanParsing.seg_offline import OffLineWindow

handling_camera = ['17', '18', '19', '20', '21', '23', '24', '25', '27', '28', '35', '36']

'''
    主窗口程序  by张精制
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
        self.last_file_path0=None
        for i in range(self.maxWindowNum):
            self.timer.append(QTimer(self))  # 控制播放的QTimer

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
        parseWin = OffLineWindow(self.widget_show)
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
        mot_path = os.path.join(root_path, 'camera/video_results/person_mot' + str(channelNo))
        listdir = os.listdir(mot_path)
        listdir.sort()
        # filepath = os.path.join(mot_path, listdir[0])
        filepath = os.path.join(mot_path, listdir[-1])

        # path = "/home/mmap/cameraVideos/mot_videos/"+str(channelNo)+"/0.mp4"

        if self.labelIndex == 0:
            self.last_file_path0=filepath
            self.vdo0.open(filepath)
            self.timer0.timeout.connect(lambda: self.play0(channelNo))
            self.timer0.start(50)
        elif self.labelIndex == 1:
            self.last_file_path1 = filepath
            self.vdo1.open(filepath)
            self.timer1.timeout.connect(lambda: self.play1(channelNo))
            self.timer1.start(50)
        elif self.labelIndex == 2:
            self.last_file_path2 = filepath
            self.vdo2.open(filepath)
            self.timer2.timeout.connect(lambda: self.play2(channelNo))
            self.timer2.start(50)
        elif self.labelIndex == 3:
            self.last_file_path3 = filepath
            self.vdo3.open(filepath)
            self.timer3.timeout.connect(lambda: self.play3(channelNo))
            self.timer3.start(50)
        self.video_no.append(0)
        self.labelIndex += 1
        print(channelNo)

    def logout(self):
        self.DVRsets_treeView.clear()
        self.cameraInfo.clear()

    def play0(self, channelNo):
        # print(0)
        self.frame = self.vdo0.get(1)
        # print(self.frame)
        ret, frame = self.vdo0.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (465, 300))
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            qimg = QtGui.QPixmap.fromImage(qimg)
            self.labVideo1.setPixmap(qimg)
            self.labVideo1.setScaledContents(True)
        else:
            #self.timer0.disconnect()
            mot_path = os.path.join(root_path, 'camera/video_results/person_mot' + str(channelNo))
            listdir = os.listdir(mot_path)
            listdir.sort()
            # if self.video_no[0] + 1 < len(listdir):
            #     self.video_no[0] += 1
            # filepath = os.path.join(mot_path, listdir[self.video_no[0]])
            filepath = os.path.join(mot_path, listdir[-1])
            if filepath != self.last_file_path0:
                self.vdo0 = cv2.VideoCapture(filepath)
                self.last_file_path0 = filepath

    def play1(self, channelNo):
        # print(self.frame)
        self.vdo1.set(1, self.frame)
        ret, frame = self.vdo1.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (465, 300))
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            qimg = QtGui.QPixmap.fromImage(qimg)
            self.labVideo2.setPixmap(qimg)
            self.labVideo2.setScaledContents(True)
        else:
            mot_path = os.path.join(root_path, 'camera/video_results/person_mot' + str(channelNo))
            listdir = os.listdir(mot_path)
            listdir.sort()
            # if self.video_no[1] + 1 < len(listdir):
            #     self.video_no[1] += 1
            # filepath = os.path.join(mot_path, listdir[self.video_no[1]])
            filepath = os.path.join(mot_path, listdir[-1])

            if filepath != self.last_file_path1:
                self.vdo1 = cv2.VideoCapture(filepath)
                self.last_file_path1 = filepath
            # self.timer1.disconnect()

    def play2(self, channelNo):
        self.vdo2.set(2, self.frame)
        ret, frame = self.vdo2.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (465, 300))
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            qimg = QtGui.QPixmap.fromImage(qimg)
            self.labVideo3.setPixmap(qimg)
            self.labVideo3.setScaledContents(True)
        else:
            mot_path = os.path.join(root_path, 'camera/video_results/person_mot' + str(channelNo))
            listdir = os.listdir(mot_path)
            listdir.sort()
            filepath = os.path.join(mot_path, listdir[0])
            if filepath != self.last_file_path2:
                self.vdo2 = cv2.VideoCapture(filepath)
                self.last_file_path2 = filepath
            # self.timer2.disconnect()

    def play3(self, channelNo):
        self.vdo3.set(2, self.frame)
        ret, frame = self.vdo3.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (465, 300))
            height, width, channel = frame.shape
            bytesPerLine = 3 * width
            qimg = QtGui.QImage(frame.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            qimg = QtGui.QPixmap.fromImage(qimg)
            self.labVideo4.setPixmap(qimg)
            self.labVideo4.setScaledContents(True)
        else:
            mot_path = os.path.join(root_path, 'camera/video_results/person_mot' + str(channelNo))
            listdir = os.listdir(mot_path)
            listdir.sort()
            filepath = os.path.join(mot_path, listdir[0])
            if filepath != self.last_file_path3:
                self.vdo3 = cv2.VideoCapture(filepath)
                self.last_file_path3 = filepath
            # self.timer3.disconnect()


    def parseCameraInfo(self):
        with open("client/ini/camera.txt", errors='ignore') as f:
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


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon('icons/v.ico'))
    myWin = MyMainWindow()
    myWin.setWindowFlags(Qt.Window)
    myWin.show()
    # print(myWin.mapToGlobal())
    sys.exit(app.exec_())
