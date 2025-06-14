'''
图像分割基类
'''

class Segmentor:
    def init(self, gpu_id, params):
        raise NotImplementedError

    # 待处理数据载入(frmBbox_array-检测框坐标信息的array；  frame_arr-视频帧array)
    def initData(self, frame_box, frame_arr):
        raise NotImplementedError

    def predict(self, mask):
        raise NotImplementedError