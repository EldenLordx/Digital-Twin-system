import numpy as np
from method.utils import *
import sys
sys.path.append('/home/cliang/mmap/VSA_Server')
from config import *
def parse_total_viper(total,probe_id,biaozhu):
    print('total: ',total)
    listtemp = total.split("*")
    newlist=[]
    temp = []
    # 获取解析后的标注信息，每五个字符串分成一组标注信息
    for i in range(len(listtemp)):
        temp.append(listtemp[i])
        if((i+1)%5==0):
           for j in range(len(newlist)):
               if temp[0]==newlist[j][0]:
                   del newlist[j]
                   break
           newlist.append(temp)
           temp = []
    old_bzlist=newlist
   # 加载已有标注信息
   #  bzScore = np.load('static/result/biaozhu/' + str(probe_id) + 'bzresult' + '.npy')
    bzScore = np.load(biaozhu)
    bzScore = bzScore.tolist()
    # 获取相同标注信息索引号delindex
    delindex=[]
    for i in range(len(newlist)):
        index = (newlist[i][0])
        torso_value = (newlist[i][1])
        leg_value = (newlist[i][2])
        for j in range(len(bzScore)):
            if ((index == (bzScore[j][0])) and (torso_value == bzScore[j][1]) and (leg_value == bzScore[j][2])):
                delindex.append(i)
                break

    listResult=[]
    if (len(delindex) > 0):
        for i in range(len(newlist)):
             if i not in delindex:
                listResult.append(newlist[i])
        newlist=listResult
    # 获取当前查找图片的标注信息
    if(len(newlist)>0):
        for i in range(len(newlist)):
            for j in range(len(bzScore)):
                if(newlist[i][0]==bzScore[j][0]):
                    del bzScore[j]
                    break;
            bzScore.append(newlist[i])
    np.save(biaozhu,bzScore)
    return newlist,old_bzlist

def parse_totalv2(total, probe_id, biaozhu):
    listtemp = total.split("*")
    newlist = []
    temp = []
    # 获取解析后的标注信息
    for i in range(len(listtemp)):
        temp.append(listtemp[i])
        if ((i + 1) % 3 == 0):
            for j in range(len(newlist)):
                if temp[0] == newlist[j][0]:
                    del newlist[j]
                    break
            newlist.append(temp)
            temp = []
    # 将newlist中第一项中的23_1555320665_2_21555321931005转换为id
    featuresList = os.listdir(reid_root+'static/Pic/video_feature')
    for i in range(len(newlist)):
        imgname = newlist[i][0]
        Id = featuresList.index(imgname + '.npy')
        newlist[i][0]=Id+1

    old_bzlist = newlist
    # 加载已有标注信息
    #  bzScore = np.load('static/result/biaozhu/' + str(probe_id) + 'bzresult' + '.npy')
    bzScore = np.load(biaozhu)
    bzScore = bzScore.tolist()
    # 获取相同标注信息索引号
    delindex = []
    for i in range(len(newlist)):
        index = (newlist[i][0])
        torso_value = (newlist[i][1])
        leg_value = (newlist[i][2])
        for j in range(len(bzScore)):
            if ((index == (bzScore[j][0])) and (torso_value == bzScore[j][1]) and (leg_value == bzScore[j][2])):
                delindex.append(i)
                break
    listResult = []
    if (len(delindex) > 0):
        for i in range(len(newlist)):
            if i not in delindex:
                listResult.append(newlist[i])
        newlist = listResult
    # 获取当前查找图片的标注信息
    if (len(newlist) > 0):
        for i in range(len(newlist)):
            for j in range(len(bzScore)):
                if (newlist[i][0] == bzScore[j][0]):
                    del bzScore[j]
                    break;
            bzScore.append(newlist[i])
    np.save(biaozhu, bzScore)
    return newlist, old_bzlist