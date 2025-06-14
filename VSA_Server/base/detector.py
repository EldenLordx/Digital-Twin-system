'''
    行人检测基类
'''

class Detector:
    #参数导入
    def init(self):
        raise NotImplementedError

    def detect(self, image):
        raise NotImplementedError