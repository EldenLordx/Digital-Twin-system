import numpy as np
import scipy.io
import math
import datetime
from django.shortcuts import render
import time

from skimage.feature import local_binary_pattern
import cv2
import os
def parse_query(query_time, probe_id,probe_name, total,sign,username,checkStr,torsoScore,legScore,path,checkvalue):

    name = scipy.io.loadmat('progal_name_tab.mat')
    torso = scipy.io.loadmat('g2g_dist_torso.mat')
    leg = scipy.io.loadmat('g2g_dist_leg.mat')
    ScoreRec = scipy.io.loadmat(path+'scoreRec.mat')
    torso_dist = torso['g2g_dist_torso']
    leg_dist = leg['g2g_dist_leg']
    dist=ScoreRec['p2g_dist']
    galleryname = name['progal_name_tab']
    int_id=int(probe_id)
    galleryname = galleryname.tolist()
    nametab = []
    for i in range(0, len(galleryname)):
        nametab.append(galleryname[i][0])
    Wu = W_affinity(torso_dist)
    Wl = W_affinity(leg_dist)

    # 查询
    if total is None:
        # 判断交互模式_距离矩阵部分
        if (checkvalue > 0):
            ScoreRec = scipy.io.loadmat('scoreRec.mat')
            dist2 = ScoreRec['p2g_dist']
            f = dist2[probe_id - 1]

        else:
            f = dist[probe_id - 1]


    else:
        newList = parse_total(total)
        # 打印输出结果
        resultprint(query_time, probe_id, probe_name, newList, nametab, sign, username)

        # 更新torsoScore和legScore
        for i in range(0, len(newList)):
            torsoScore[probe_id - 1].append(newList[i][1])
            legScore[probe_id - 1].append(newList[i][2])
        np.save(path + "torsoScore.npy", torsoScore)
        np.save(path + "legScore.npy", legScore)


        # 判断交互模式_交互部分
        if(checkvalue>0):
            newList2 = merge(probe_id)
        else:
            newList2 = newList

        new_score = []
        for i in range(0, len(newList2)):
            line = int(newList2[i][0])
            score1 = Wu[:, line - 1]
            score2 = Wl[:, line - 1]
            score = newList2[i][1] * score1 + newList2[i][2] * score2
            new_score.append(score)
        new_score = np.sum(new_score, axis=0)
        # 判断交互模式_距离矩阵部分
        if (checkvalue > 0):
            ScoreRec = scipy.io.loadmat('scoreRec.mat')
            dist2 = ScoreRec['p2g_dist']
            f=dist2[probe_id - 1]

        else:
            f = dist[probe_id - 1]

        f = f + new_score
        fmax, fmin = f.max(axis=0), f.min(axis=0)
        f = (f - fmin) / (fmax - fmin)


        # 不同用户模式结果保存
        if (checkvalue > 0):
            ScoreRec = scipy.io.loadmat('scoreRec.mat')
            dist2 = ScoreRec['p2g_dist']
            dist2[probe_id - 1] = f

            scipy.io.savemat('scoreRec.mat', {'p2g_dist': dist2, })
        else:
            dist[probe_id - 1] = f
            scipy.io.savemat(path + 'scoreRec.mat', {'p2g_dist': dist, })

    sortname = []
    sortnameV2 = []
    sortnumber = np.argsort(-f)
    galleryid = sortnumber + 1
    galleryid = galleryid.tolist()

    for i in range(0, len(torso_dist)):
        if (galleryid[i] == probe_id):
            rank = i + 1
            print("Rank:", rank)
    for i in range(0, len(f)):
        sortname.append(nametab[sortnumber[i]])
    for i in range(0, len(f)):
        sortnameV2.append(sortname[i][0])
    f = f.tolist()
    return f, sortnameV2, galleryid, rank,torsoScore,legScore

def W_affinity(W):
    Wmax, Wmin = W.max(axis=1), W.min(axis=1)
    W = (W - Wmin) / (Wmax - Wmin)
    W_temp = 1 - W
    W_temp = (W_temp + np.transpose(W_temp)) / 2
    W = W_temp - np.eye(54)
    # W=np.exp(-W)
    return W

def resultprint(query_time,probe_id, probe_name,newList,nametab,sign,username):
    # viper_bzresult = []            # 单个feedback_details
    path = 'viper_bzresult/' + username + '/'
    feedback_details_all = np.load(path + "feedback_details_all.npy")
    feedback_details_all = feedback_details_all.tolist()
    feedback_details=feedback_details_all[probe_id-1]

    Inid=int(probe_id)
    # global feedback_details
    beta=0.05
    theta=0.25
    # sign=int(sign)
    # if(sign==1):
    #     feedback_details=[]

    result2=[]             # probe信息
    torso_array = []       # 所有标注值的上半身置信度
    leg_array = []         # 所有标注值的下半身置信度
    torso_sgn = []
    leg_sgn = []
    result3=[]
    # 读取mat文件
    body_rect = scipy.io.loadmat('body_rect.mat')
    rect = body_rect['body_rectV2']
    rect = rect.tolist()
    rectangle = []
    # rectangle获取了54张gallery位置框信息--rectangle
    for i in range(0, 54):
        rectangle.append(rect[0][i])
    for i in range(len(newList)):
        source = ['U', 'U']  # M代表标注，U代表未定义
        mark_flag = ['N', 'N']  # 对上下半身，Y代表已标记，N代表未标记
        birth_run = np.array([0,0]) # 上下半身若被标记，则显示该半身查询的次数query_time
        birth_run = birth_run.reshape(len(birth_run), 1)

        box_rect = []  # 上下半身位置框信息
        body_part = np.array([0, 0])  # 若上半身被标记第一个元素为1，若下半身被标记第二个元素为2，否则为0
        body_part = body_part.reshape(len(body_part), 1)

        box_type = np.array([0, 0])   # 两个元素，若被标记置信度为正则为1，标记为负则为-1
        box_type = box_type.reshape(len(box_type), 1)

        box_conf=[0,0]
        cur_pos=[0,0]
        # last_update_time参数设置
        import time
        update_time = time.localtime()
        year = (update_time.tm_year)
        mon = (update_time.tm_mon)
        mday = (update_time.tm_mday)
        hour = (update_time.tm_hour)
        min = (update_time.tm_min)
        sec = (update_time.tm_sec)
        thetime=[]
        last_update_time = []
        last_update_time1 = []
        last_update_time.append(0)
        last_update_time.append(0)
        last_update_time.append(0)
        last_update_time.append(0)
        last_update_time.append(0)
        last_update_time.append(0)
        last_update_time1.append(year)
        last_update_time1.append(mon)
        last_update_time1.append(mday)
        last_update_time1.append(hour)
        last_update_time1.append(min)
        last_update_time1.append(sec)
        time1 = []
        time2 = []
        time3 = []
        last_update_time = list(map(float, last_update_time))
        last_update_time1 = list(map(float, last_update_time1))

        time1.append(last_update_time1)
        time1.append(last_update_time)
        time2.append(last_update_time)
        time2.append(last_update_time1)
        time3.append(last_update_time1)
        time3.append(last_update_time1)
        #operator
        operator = np.empty((2, 1), dtype=object)
        operator0 = np.empty((2, 1), dtype=object)
        operator0[0, 0] = username
        operator0[1, 0] = 'default'
        operator1 = np.empty((2, 1), dtype=object)
        operator1[0, 0] = 'default'
        operator1[1, 0] = username
        operator2 = np.empty((2, 1), dtype=object)
        operator2[0, 0] = username
        operator2[1, 0] = username

        index=newList[i][0] # galleryid
        box_rect=rectangle[index-1]
        torso_array.append(newList[i][1])
        leg_array.append(newList[i][2])
        if (torso_array[i] > beta):
            source[0]='M'
            mark_flag[0]='Y'
            birth_run[0][0]=1
            body_part[0]=1
            box_type[0]=1
            cur_pos[0]=(torso_array[i])
            box_conf[0]=(torso_array[i]-beta)/(1-beta)

        elif(torso_array[i] < -1*beta):
            source[0] = 'M'
            mark_flag[0] = 'Y'
            birth_run[0][0] = 1
            body_part[0] = 1
            box_type[0] = -1
            cur_pos[0] = torso_array[i]
            box_conf[0] = -1*(torso_array[i] + beta) / (1 - beta)

        elif(abs(torso_array[i])<=beta):
            source[0] = 'U'
            mark_flag[0] = 'N'
            birth_run[0][0] = 0
            cur_pos[0] = torso_array[i]
            box_conf[0] = 0

        #下半身
        if (leg_array[i] > beta):
            source[1]='M'
            mark_flag[1]='Y'
            birth_run[1][0]=1
            body_part[1]=2
            box_type[1]=1
            cur_pos[1]=leg_array[i]
            box_conf[1]=(leg_array[i]-beta)/(1-beta)
        elif(leg_array[i] < -1*beta):
            source[1] = 'M'
            mark_flag[1] = 'Y'
            birth_run[1][0] = 1
            body_part[1] = 2
            box_type[1] = -1
            cur_pos[1] = leg_array[i]
            box_conf[1] = -1*(leg_array[i] + beta) / (1 - beta)
        elif(abs(leg_array[i])<=beta):
            source[1] = 'U'
            mark_flag[1] = 'N'
            birth_run[1][0] = 0
            cur_pos[1] = leg_array[i]
            box_conf[1] = 0

        if(torso_array[i]!=0 and leg_array[i]==0 ):
            thetime=time1
            operator=operator0
        if(torso_array[i]==0 and leg_array[i]!=0):
            thetime = time2
            operator=operator1
        if (torso_array[i] != 0 and leg_array[i] != 0):
            thetime=time3
            operator=operator2
        cur_pos = np.array(cur_pos)  # 标注的置信度的值
        cur_pos = cur_pos.reshape(len(cur_pos), 1)

        box_conf = np.array(box_conf)
        box_conf = box_conf.reshape(len(box_conf), 1)


        # body_part = list(map(float, body_part))
        birth_run=birth_run.astype(np.float)
        box_rect = box_rect.astype(np.float)
        body_part = body_part.astype(np.float)
        box_type = box_type.astype(np.float)
        box_conf = box_conf.astype(np.float)
        cur_pos = cur_pos.astype(np.float)
        probe_id = float(probe_id)
        dict1 = {
                'source':source,
                'mark_flag':mark_flag,
                'birth_run':birth_run,
                'box_rect':box_rect,
                'body_part':body_part,
                'box_type':box_type,
                'box_conf':box_conf,
                'cur_pos':cur_pos,
                'last_update_time':thetime,
                'gallery_name':nametab[newList[i][0]-1][0],
                'operator': operator,
            }
        feedback_details.append(dict1)

    feedback_details_all[Inid - 1]=feedback_details

    np.save(path + 'feedback_details_all.npy', feedback_details_all)
    dict2={
        'probe_id':probe_id,
        'probe_name':probe_name,
    }

    path='viper_bzresult/'+username+'/'
    mkdir(path)

    dict3={
         'feedback_details': feedback_details,
        'probe_info': dict2,
     }

    scipy.io.savemat(path+probe_name+'.mat',
                     {'feedback_info':dict3,

                            })

def todrawrectangle(probe_id,galleryid,torso,leg,username):
    import matplotlib.pyplot as plt
    import cv2
    # 读取mat文件
    body_rect = scipy.io.loadmat('body_rect.mat')
    galid2name = scipy.io.loadmat('progal_name_tab.mat')
    rect = body_rect['body_rectV2']
    rect = rect.tolist()
    rectangle = []
    galleryid2name = galid2name['progal_name_tab']
    galleryid2name = galleryid2name.tolist()

    #rectangle获取了316张gallery位置框信息--rectangle
    for i in range(0, 54):
        rectangle.append(rect[0][i])

    #galleryname获取被标注图片的图片名
    galleryname=[]
    index=int(galleryid)
    galleryname=galleryid2name[index-1][0][0]

    if(int(galleryname)<10):
        galleryname="0"+galleryname
    # fname = 'D:/SPF20180925/static/Pic/gallery/' + galleryname + '.jpg'
    fname = './static/Pic/gallery/' + galleryname + '.jpg'
    path='./static/Pic/returnimage/'+username +'/'+ probe_id+'/'
    mkdir(path)
    img = cv2.imread(fname)

    if (float(torso) > 0):
        cv2.rectangle(img, (rectangle[index-1][0][0], rectangle[index-1][0][1]),(rectangle[index-1][0][0] + rectangle[index-1][0][2], rectangle[index-1][0][1] + rectangle[index-1][0][3]), (0, 255, 0),1)
    if (float(torso) < 0):
        cv2.rectangle(img, (rectangle[index-1][0][0], rectangle[index-1][0][1]),(rectangle[index-1][0][0] + rectangle[index-1][0][2], rectangle[index-1][0][1] + rectangle[index-1][0][3]), (0, 0, 255),1)
    if (float(leg) > 0):
        cv2.rectangle(img, (rectangle[index-1][1][0], rectangle[index-1][1][1]),(rectangle[index-1][1][0] + rectangle[index-1][1][2], rectangle[index-1][1][1] + rectangle[index-1][1][3]), (0, 255, 0),1)
    if (float(leg) < 0):
        cv2.rectangle(img, (rectangle[index-1][1][0], rectangle[index-1][1][1]),(rectangle[index-1][1][0] + rectangle[index-1][1][2], rectangle[index-1][1][1] + rectangle[index-1][1][3]), (0, 0, 255),1)
    cv2.imwrite(path+galleryname + '.jpg', img)
    return galleryname


def mkdir(path):
    # 引入模块
    import os

    # 去除首位空格
    path = path.strip()
    # 去除尾部 \ 符号
    path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(path)
        print(path + ' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        print(path + ' 目录已存在')
        return False

def mkdir_load(path,probe_id):
    # 引入模块
    import os
    #
    # # 去除首位空格
    # path = path.strip()
    # # 去除尾部 \ 符号
    # path = path.rstrip("\\")

    # 判断路径是否存在
    # 存在     True
    # 不存在   False
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数

        result = [[] for i in range(29)]
        np.save(path, result)
        print(' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

def parse_total(total):
    listtemp = total.split("*")
    newlist=[]
    temp = []
    for i in range(len(listtemp)):
        temp.append(listtemp[i])
        if((i+1)%3==0):
            newlist.append(temp)
            temp = []
    return newlist

def average(seq):
 return float(sum(seq)) / len(seq)

def merge(probeid):
    index = probeid - 1
    users = os.listdir('./viper_bzresult')
    list_checkStr = []
    list_torsoScore = []
    list_legScore = []
    usernumber = len(users)
    for i in range(usernumber):
        path_checkStr = os.path.join('viper_bzresult', users[i], 'checkStr.npy')
        path_torsoScore = os.path.join('viper_bzresult', users[i], 'torsoScore.npy')
        path_legScore = os.path.join('viper_bzresult', users[i], 'legScore.npy')
        list_checkStr.append(path_checkStr)
        list_torsoScore.append(path_torsoScore)
        list_legScore.append(path_legScore)
        # 二维数组
    checkInfo = []
    torsoInfo = []
    legInfo = []
    for i in range(len(list_checkStr)):
        probe_list_checkStr = np.load(list_checkStr[i])
        probe_list_checkStr[index] = list(map(int, probe_list_checkStr[index]))
        checkInfo.append(probe_list_checkStr[index])

        probe_list_torsoScore = np.load(list_torsoScore[i])
        torsoInfo.append(probe_list_torsoScore[index])

        probe_list_legScore = np.load(list_legScore[i])
        legInfo.append(probe_list_legScore[index])

    # 生成拼接数组
    checkInfo_all = []
    for i in range(len(checkInfo)):
        checkInfo_all.extend(checkInfo[i])

    myList = []
    for i in range(len(checkInfo)):
        for j in range(len(checkInfo[i])):
            listtemp = []
            listtemp.append(checkInfo[i][j])
            listtemp.append(torsoInfo[i][j])
            listtemp.append(legInfo[i][j])
            myList.append(listtemp)

    answer = []
    answerlist = []
    idlist = []
    for i in range(len(checkInfo_all)):
        if (checkInfo_all.count(checkInfo_all[i]) > 1):
            id = checkInfo_all[i]
            idlist.append(id)
    length = int(len(idlist) / 2)
    idlist = idlist[0:length]

    # 找出相同项answerlist
    for j in range(len(idlist)):
        answer = [i for i, x in enumerate(checkInfo_all) if x == idlist[j]]
        answerlist.append(answer)

    torso = []
    leg = []
    upsum = [[] for i in range(len(answerlist))]
    downsum = [[] for i in range(len(answerlist))]
    for i in range(len(answerlist)):
        for j in range(len(answerlist[i])):
            newid = answerlist[i][j]
            upsum[i].append(myList[newid][1])
            downsum[i].append(myList[newid][2])

    torso = [[] for i in range(len(answerlist))]
    leg = [[] for i in range(len(answerlist))]

    for i in range(len(upsum)):
        torso[i].append(average(upsum[i]))
        leg[i].append(average(downsum[i]))

    allList = []
    if (len(idlist) == 0):
        allList = myList
    else:
        for i in range(len(myList)):
            for j in range(len(idlist)):
                if (myList[i][0] != idlist[j]):
                    allList.append(myList[i])
        for i in range(len(answerlist)):
            allList.append([idlist[i], torso[i][0], leg[i][0]])
    return allList


def LBP_features(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp_img = local_binary_pattern(im, n_points, radius)
    lbp = lbp_img.reshape(lbp_img.size, order="C")
    lbp = lbp.astype(np.float64)
    # lbp = img2cols(lbp_img)
    return lbp
def img2cols(img):
    img = img.reshape(img.size, order="C")
    img = img.astype(np.float64)
    return img

def EuclideanDistances(A, B):
    BT = B.transpose()
    # vecProd = A * BT
    vecProd = np.dot(A,BT)
    # print(vecProd)
    SqA =  A**2
    # print(SqA)
    sumSqA = np.matrix(np.sum(SqA, axis=1))
    sumSqAEx = np.tile(sumSqA.transpose(), (1, vecProd.shape[1]))
    # print(sumSqAEx)

    SqB = B**2
    sumSqB = np.sum(SqB, axis=1)
    sumSqBEx = np.tile(sumSqB, (vecProd.shape[0], 1))
    SqED = sumSqBEx + sumSqAEx - 2*vecProd
    SqED[SqED<0]=0.0
    ED = np.sqrt(SqED)
    return ED






