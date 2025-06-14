'''
    人脸检测基类
'''


class FaceDetector(object):
    # #  子类__init__中需调用父类构造方法，例如：super(MTCNNDetector, self).__init__(name='MTCNN')
    # def __init__(self, name):
    #     self.name = name  # 检测器名称，方便后期做一些指标统计

    # 初始化网络模型及参数
    def init(self, gpu_id):
        raise NotImplementedError

    # 人脸检测,返回bboxs,landmarks  例如：mtcnn_detector.persondetect(img)
    def detect(self, image):
        raise NotImplementedError
