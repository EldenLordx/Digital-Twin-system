'''
    单目标跟踪基类
'''


class SOTTracker(object):

    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()