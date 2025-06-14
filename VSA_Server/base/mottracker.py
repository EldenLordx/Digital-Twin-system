'''
    多目标跟踪基类
'''


class MOTTracker(object):

    def init(self, gpu_id):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()