'''
    特征提取基类
'''


class Extractor:
    # 参数及模型初始化
    def init(self, gpu_id, weight_file):
        raise NotImplementedError

    # 待处理数据载入(frameArr-视频帧array)
    def extract(self, img):
        raise NotImplementedError
