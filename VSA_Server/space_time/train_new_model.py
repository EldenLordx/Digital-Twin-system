import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
import csv

degrees = [0] * 37
degrees[17] = [3, 1]
degrees[18] = [1, 1]
degrees[19] = [1, 1]
degrees[20] = [12, 1]
degrees[21] = [1, 1]
degrees[23] = [8, 1]
degrees[24] = [1, 1]
degrees[25] = [3, 1]
degrees[27] = [2, 8]
degrees[28] = [9, 3]
degrees[35] = [1, 1]
degrees[36] = [1, 1]

def load_data(cam_num):
  file=csv.reader(open('./train/'+str(cam_num)+'.csv','r'))
  X,Y=[],[]
  for line in file:
    # print(line)
    if file.line_num == 1:
      continue
    x1,x2,y1,y2=[int(x) for x in line]
    X.append([x1,x2])
    Y.append([y1, y2])
  trainx = np.array([[degrees[cam_num][0], x1, x2, y1 + (np.random.random(1) - 0.5)] for [x1,x2],[y1,y2] in zip(X, Y)])
  trainy = np.array([[degrees[cam_num][1], x1, x2, y2 + (np.random.random(1) - 0.5)] for [x1, x2], [y1, y2] in zip(X, Y)])
  return trainx,trainy

from sklearn import linear_model
# model_LinearRegression = linear_model.LinearRegression()
###########2.回归部分##########
def train_save_method():
  Xmodel=linear_model.LinearRegression()
  Ymodel=linear_model.LinearRegression()
  xmodel = Xmodel.fit(x_trainx,y_trainx)
  joblib.dump(xmodel, './model/'+str(cam_num)+'_x.model')
  ymodel = Ymodel.fit(x_trainy, y_trainy)
  joblib.dump(ymodel, './model/' + str(cam_num) + '_y.model')

cam_num=35
trainx,trainy = load_data(cam_num)
x_trainx, y_trainx = trainx[:,:3], trainx[:,3] #数据前两列是x1,x2 第三列是y,这里的y有随机噪声
x_trainy, y_trainy = trainy[:,:3], trainy[:,3] #数据前两列是x1,x2 第三列是y,这里的y有随机噪声
train_save_method()






# ###########3.具体方法选择##########
# ####3.1决策树回归####
# from sklearn import tree
# model_DecisionTreeRegressor = tree.DecisionTreeRegressor()
# ####3.2线性回归####
# from sklearn import linear_model
# model_LinearRegression = linear_model.LinearRegression()
# ####3.3SVM回归####
# from sklearn import svm
# model_SVR = svm.SVR()
# ####3.4KNN回归####
# from sklearn import neighbors
# model_KNeighborsRegressor = neighbors.KNeighborsRegressor()
# ####3.5随机森林回归####
# from sklearn import ensemble
# model_RandomForestRegressor = ensemble.RandomForestRegressor(n_estimators=20)#这里使用20个决策树
# ####3.6Adaboost回归####
# from sklearn import ensemble
# model_AdaBoostRegressor = ensemble.AdaBoostRegressor(n_estimators=50)#这里使用50个决策树
# ####3.7GBRT回归####
# from sklearn import ensemble
# model_GradientBoostingRegressor = ensemble.GradientBoostingRegressor(n_estimators=100)#这里使用100个决策树
# ####3.8Bagging回归####
# from sklearn.ensemble import BaggingRegressor
# model_BaggingRegressor = BaggingRegressor()
# ####3.9ExtraTree极端随机树回归####
# from sklearn.tree import ExtraTreeRegressor
# model_ExtraTreeRegressor = ExtraTreeRegressor()
###########4.具体方法调用部分##########
# try_different_method(model_DecisionTreeRegressor)

