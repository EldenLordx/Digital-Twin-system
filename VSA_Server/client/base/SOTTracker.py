class Tracker(object):

    def __init__(self, name):
        self.name = name

    def init(self, image, box):
        raise NotImplementedError()

    def update(self, image):
        raise NotImplementedError()
