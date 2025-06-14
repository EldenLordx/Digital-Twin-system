from sklearn import linear_model
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from utils import draw_utils

import matplotlib.pyplot as pl
import sklearn.metrics as sm
import numpy as np
import csv
import math
import os
import time
import pandas as pd


# def csv2numpy(path_csv):
#     # 从文本文件中提取数据并转为numpy Array格式
#     csv_file = pd.read_csv(path_csv)
#     content = []  # 用来存储整个文件的数据，存成一个列表，列表的每一个元素又是一个列表，表示的是文件的某一行
#     for index, list in enumerate(csv_file):
#         if index == 0 or list[0] == '':
#             continue
#         list_item = []
#         for index_sub, item in enumerate(list):
#             list_item.append(int(item))
#         content.append(list_item)
#     content = np.array(content)
#     return content

def csv2numpy(path_csv):
    # 从文本文件中提取数据并转为numpy Array格式
    csv_file = csv.reader(open(path_csv, encoding='ISO-8859-1'))
    content = []  # 用来存储整个文件的数据，存成一个列表，列表的每一个元素又是一个列表，表示的是文件的某一行
    for index, list in enumerate(csv_file):
        if index == 0 or list[0] == '':
            continue
        list_item = []
        for index_sub, item in enumerate(list):
            list_item.append(int(item))
        content.append(list_item)
    content = np.array(content)
    return content
