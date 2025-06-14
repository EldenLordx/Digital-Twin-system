import numpy as np
import scipy.io,sys
from skimage.feature import local_binary_pattern
from decimal import Decimal
sys.path.append('/home/cliang/mmap/VSA_Server')
from config import *
from io import BytesIO
import base64
import matplotlib.pyplot as plt
import os
import cv2
from pylab import *
from collections import defaultdict
import time
from config import *

# 生成相应的结果文件夹
def mkdir(path):


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

# 生成相应的结果文件
def mkdir_load(path):
    # 引入模块
    import os
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

# 计算两个向量间的欧氏距离
def calEuclideanDistance(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist

# 解析total成标注信息数组
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


def viper_biaozhu(path):
    # 引入模块
    import os
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        result = []
        np.save(path, result)
        print(' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

def viper_mkdir_load(path,gallerynumber):
    # 引入模块
    import os
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        result = [[] for i in range(gallerynumber)]
        np.save(path, result)
        print(' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

def video_mAP_load(path):
    # 引入模块
    import os
    isExists = os.path.exists(path)

    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        result = []
        np.save(path, result)
        print('123456')
        print(' 创建成功')
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

def biaozhuId(probe_id,username):

    gallerypath = reid_root+'static/Pic/gallery/'
    bz_imglist = sorted(os.listdir(gallerypath))

    biaozhupath=reid_root+'static/Pic/returnimage/'+username+'/'+dataset+'/'+probe_id+'/'
    isExists = os.path.exists(biaozhupath)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        bzimgname = []
        bzimgId = []
        return bzimgname,bzimgId
    else:

        bzlist=os.listdir(biaozhupath)

        bzimgname=[]
        bzimgId=[]
        for i in range(len(bzlist)):
            bzimgname.append(bzlist[i])
            imgid=bz_imglist.index(bzlist[i])+1
            bzimgId.append(imgid)
    return bzimgname,bzimgId

# 提取LBP特征
def LBP_features(im):
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    radius = 3
    n_points = 8 * radius
    lbp_img = local_binary_pattern(im, n_points, radius)
    lbp = lbp_img.reshape(lbp_img.size, order="C")
    lbp = lbp.astype(np.float64)
    # lbp = img2cols(lbp_img)
    return lbp

# min_Max归一化，减平均
def min_Max(Initialdis):
    Dis=[]
    for x in Initialdis:
        x = float(x - np.mean(Initialdis)) / (np.max(Initialdis) - np.min(Initialdis))
        Dis.append(x)
    return Dis

# 归一化，减最小值
def norm(Initialdis):
    Dis=[]
    for x in Initialdis:
        x = float(x - np.min(Initialdis)) / (np.max(Initialdis) - np.min(Initialdis))
        Dis.append(x)
    return Dis

# 显示得分，保留三位小数
def showScore(score):
    result=[]
    for x in score:
        result.append(format(x,'.3f'))
    return result

# 绘制排序图
def qu2rank(ranklist):
    plt.switch_backend('agg')
    x = [i+1 for i in range(len(ranklist))]
    y=ranklist
    plt.figure(figsize=(2, 1.3))
    plt.xlim((0,len(ranklist)+1))
    plt.ylim((1,int(1.5*max(ranklist))))
    plt.xlabel('querytimes',fontsize=6)
    plt.ylabel('Rank',fontsize=5)
    plt.title('querytimes-rank',fontsize=6)
    my_x_ticks = np.arange(0, len(ranklist)+1,1)
    # my_y_ticks = np.arange(1, max(ranklist))
    plt.xticks(my_x_ticks,fontsize=5)
    plt.yticks(fontsize=5)
    plt.plot(x, y, color='r',markerfacecolor='blue',marker='o',markersize='2')
    for a,b in zip(x, y):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=6)
    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = str(base64.encodebytes(sio.getvalue()).decode()).replace('\n' , '')
    qu2ranksrc='data:image/png;base64,' + data
    plt.close()
    return qu2ranksrc

# 绘制mAP图
def qu2mAP(mAP):
    plt.switch_backend('agg')
    x = [i+1 for i in range(len(mAP))]
    y=mAP
    plt.figure(figsize=(2, 1.6))
    plt.xlim((0,len(mAP)+1))
    plt.ylim((0, 1))
    plt.xlabel('querytimes',fontsize=6)
    plt.ylabel('mAP',fontsize=5)
    plt.title('querytimes-mAP',fontsize=6)
    my_x_ticks = np.arange(0, len(mAP)+1,1)
    # my_y_ticks = np.arange(1, max(ranklist))
    plt.xticks(my_x_ticks,fontsize=5)
    plt.yticks(fontsize=5)
    plt.plot(x, y, color='r',markerfacecolor='blue',marker='o',markersize='2')
    for a,b in zip(x, y):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=4.5)
    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = str(base64.encodebytes(sio.getvalue()).decode()).replace('\n' , '')
    qu2mAP='data:image/png;base64,' + data
    plt.close()
    return qu2mAP

# 计算cmc用到的函数，获取特殊样本
def _unique_sample(ids_dict, num):
    mask = np.zeros(num, dtype=np.bool)
    for _, indices in ids_dict.items():
        i = np.random.choice(indices)
        mask[i] = True
    return mask

# 计算cmc的值
def getcmcresult(score, query_ids, gallery_ids,
                query_cams, gallery_cams, topk,
                separate_camera_set=False,
                single_gallery_shot=False,
                first_match_break=False):
    score = np.array(score)
    m, n = score.shape

    # Fill up default values
    if query_ids is None:
        query_ids = np.arange(m)
    if gallery_ids is None:
        gallery_ids = np.arange(n)
    if query_cams is None:
        query_cams = np.zeros(m).astype(np.int32)
    if gallery_cams is None:
        gallery_cams = np.ones(n).astype(np.int32)
    # Ensure numpy array
    query_ids = np.asarray(query_ids)
    gallery_ids = np.asarray(gallery_ids)
    query_cams = np.asarray(query_cams)
    gallery_cams = np.asarray(gallery_cams)
    # Sort and find correct matches
    indices = np.argsort(-score, axis=1)

    matches = (gallery_ids[indices] == query_ids[:, np.newaxis])
    # Compute CMC for each query
    ret = np.zeros(topk)
    num_valid_queries = 0
    for i in range(m):
        # Filter out the same id and same camera
        valid = ((gallery_ids[indices[i]] != query_ids[i]) |
                 (gallery_cams[indices[i]] != query_cams[i]))
        if separate_camera_set:
            # Filter out samples from same camera
            valid &= (gallery_cams[indices[i]] != query_cams[i])

        if not np.any(matches[i, valid]):
            continue
        if single_gallery_shot:
            repeat = 10
            gids = gallery_ids[indices[i][valid]]
            inds = np.where(valid)[0]
            ids_dict = defaultdict(list)
            for j, x in zip(inds, gids):
                ids_dict[x].append(j)
        else:
            repeat = 1
        for _ in range(repeat):
            if single_gallery_shot:
                # Randomly choose one instance for each id
                sampled = (valid & _unique_sample(ids_dict, len(valid)))
                index = np.nonzero(matches[i, sampled])[0]
            else:
                index = np.nonzero(matches[i, valid])[0]
            delta = 1. / (len(index) * repeat)
            for j, k in enumerate(index):
                if k - j >= topk: break
                if first_match_break:
                    ret[k - j] += 1
                    break
                ret[k - j] += delta
        num_valid_queries += 1
    if num_valid_queries == 0:
        raise RuntimeError("No valid query")
    return ret.cumsum() / num_valid_queries

# 绘制第一次查询的cmc图
def drawcmc1(cmcresult,gallerynumber):
    plt.switch_backend('agg')
    x = [i+1 for i in range(len(cmcresult))]
    y=cmcresult
    xlim((1, gallerynumber))
    ylim((0, 1))
    plt.figure(figsize=(2, 1.3))
    plt.xlabel('Rank',fontsize=6)
    plt.ylabel('Identification Rate',fontsize=5)
    plt.title('CMC',fontsize=6)
    my_x_ticks = np.arange(0, gallerynumber, 100)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.xticks(my_x_ticks,fontsize=5)
    plt.yticks(my_y_ticks,fontsize=5)

    currentCMC=plt.plot(x, y, color='red', label='currentCMC')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 4,
             }
    plt.legend(currentCMC, prop=font1)

    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = str(base64.encodebytes(sio.getvalue()).decode()).replace('\n', '')
    cmcsrc = 'data:image/png;base64,' + (data)
    plt.close()
    return cmcsrc

# 绘制多次查询的cmc图
def drawcmc2(cmcresult1,cmcresult2,gallerynumber):
    plt.switch_backend('agg')
    x = [i+1 for i in range(len(cmcresult1))]
    y1=cmcresult1
    y2=cmcresult2
    xlim((1, gallerynumber))
    ylim((0, 1))
    plt.figure(figsize=(2, 1.3))
    plt.xlabel('Rank',fontsize=5)
    plt.ylabel('Identification Rate',fontsize=5)
    plt.title('CMC',fontsize=6)
    my_x_ticks = np.arange(0, gallerynumber, 100)
    my_y_ticks = np.arange(0, 1, 0.1)
    plt.xticks(my_x_ticks,fontsize=5)
    plt.yticks(my_y_ticks,fontsize=5)
    LastCMC,=plt.plot(x, y1, color='blue', label='LastCMC')
    currentCMC,=plt.plot(x, y2, color='red', label='currentCMC')
    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 4,
             }
    plt.legend([LastCMC,currentCMC], prop=font1)

    sio = BytesIO()
    plt.savefig(sio, format='png')
    data = str(base64.encodebytes(sio.getvalue()).decode()).replace('\n', '')
    cmcsrc = 'data:image/png;base64,' + (data)
    plt.close()
    return cmcsrc

def parse_total2(total,probe_id):
    listtemp = total.split("*")
    newlist = []
    temp = []
    for i in range(len(listtemp)):
        temp.append(listtemp[i])
        if ((i + 1) % 3 == 0):
            newlist.append(temp)
            temp = []
    bzScore = np.load(reid_root+'static/Pic/biaozhu/' + str(probe_id) + 'bzresult' + '.npy')
    bzScore=bzScore.tolist()
    for i in range(len(newlist)):
        index=int(newlist[i][0])
        torso_value=float(newlist[i][1])
        leg_value=float(newlist[i][2])
        for j in range(len(bzScore)):
            if((index==bzScore[j][0]) and (torso_value==bzScore[j][0]) and(leg_value==bzScore[j][1])):
                del newlist[i]
    return newlist

# 将opencv画框后的图片保存为base64格式，便于在前端实时更新
def image_to_base64(image_np,filetype):
    image = cv2.imencode(filetype, image_np)[1]
    image_code = str(base64.b64encode(image))[2:-1].replace('\n' , '')
    image_code='data:image/bmp;base64,' + image_code
    return image_code

# 更新每次排序得分
def parse_bzscore(Score,probe_id):

    result=[]
    for i in range(len(Score)):
        result.append(Score[i])
    np.save(reid_root+'static/Pic/tjInfo/' + str(probe_id) + 'tongji' + '.npy',result)

    return result


# 加载统计信息，解析出相应的标注信息，最终显示在前端页面
def parse_tongjiinfo(tongjinifo):
    gallerynumber=0
    posnumber=0
    negnumber=0
    torsonumber=0
    legnumber=0
    for i in range(len(tongjinifo)):
        torsovalue=float(tongjinifo[i][1])
        legvalue=float(tongjinifo[i][2])
        # 计算gallerynumber
        if(torsovalue>0):
            gallerynumber=gallerynumber+1
            posnumber=posnumber+1
            torsonumber=torsonumber+1
        if(torsovalue<0):
            gallerynumber = gallerynumber + 1
            negnumber = negnumber + 1
            torsonumber = torsonumber + 1
        if(legvalue>0):
            gallerynumber = gallerynumber + 1
            posnumber = posnumber + 1
            legnumber = legnumber + 1
        if (legvalue < 0):
            gallerynumber = gallerynumber + 1
            negnumber = negnumber + 1
            legnumber = legnumber + 1

    return gallerynumber, posnumber,negnumber,torsonumber,legnumber


# 在获取中心视频人物中的probe时，通过与跟踪结果匹配最大IOU来获取正确的查询图片
def IOU( box1, box2 ):
    """
    :param box1:[x1,y1,x2,y2] 左上角的坐标与右下角的坐标
    :param box2:[x1,y1,x2,y2]
    :return: iou_ratio--交并比
    """
    width1 = abs(box1[2] - box1[0])
    height1 = abs(box1[1] - box1[3]) # 这里y1-y2是因为一般情况y1>y2，为了方便采用绝对值
    width2 = abs(box2[2] - box2[0])
    height2 = abs(box2[1] - box2[3])
    x_max = max(box1[0],box1[2],box2[0],box2[2])
    y_max = max(box1[1],box1[3],box2[1],box2[3])
    x_min = min(box1[0],box1[2],box2[0],box2[2])
    y_min = min(box1[1],box1[3],box2[1],box2[3])
    iou_width = x_min + width1 + width2 - x_max
    iou_height = y_min + height1 + height2 - y_max
    if iou_width <= 0 or iou_height <= 0:
        iou_ratio = 0
    else:
        iou_area = iou_width * iou_height # 交集的面积
        box1_area = width1 * height1
        box2_area = width2 * height2
        iou_ratio = iou_area / (box1_area + box2_area - iou_area) # 并集的面积
    return iou_ratio
# box1 = [1,3,4,1]
# box2 = [2,4,5,2]

# 计算当前查询图片的ap值
def get_ap(good_image, junk, index):
    ap = 0
    old_recall = 0
    old_precision = 1.0
    intersect_size = 0
    n_junk = 0
    ngood = len(good_image)
    if(ngood==0):
        ap=0
    else:
        good_now = 0
        j = 0
        for i in range(index.shape[0]):
            flag = False
            if index[i] in junk:
                n_junk += 1
                continue
            if index[i] in good_image:
                good_now += 1
                flag = True
            if flag:
                intersect_size = intersect_size + 1
            recall = intersect_size/ngood
            precision = intersect_size/(j + 1)
            ap = ap + (recall - old_recall)*((old_precision+precision)/2)
            old_recall = recall
            old_precision = precision
            j = j + 1
            if good_now == ngood:
                return ap
    return ap

# 这里是计算mAP用到的函数
def get_rank(good_image, junk, index):
    rank = index.shape[0]
    n_junk = 0
    for i in range(index.shape[0]):
        if index[i] in junk:
            n_junk += 1
            continue
        if index[i] in good_image:
            return i - n_junk
# 这里是计算mAP用到的函数
def get_good_junk(q, label_gallery, label_query, cam_gallery, cam_query, junk_0):
    q_label = label_query[q]
    q_cam = cam_query[q]
    pos = label_gallery == q_label
    pos_2 = cam_gallery != q_cam
    good_image = np.argwhere(np.logical_and(pos, pos_2))
    # print('good_image: ',good_image)
    if (good_image.shape[0] > 1):
        good_image = good_image.squeeze()
    # else:
    #     good_image = good_image[0]
    elif(len(good_image)>0):
        good_image = good_image[0]


    pos_3 = cam_gallery == q_cam
    junk = np.argwhere(np.logical_and(pos, pos_3)).squeeze()
    junk = np.append(junk_0, junk)
    return good_image, junk

# 这里是计算mAP用到的函数，计算最终mAP
def evaluate(distance, label_gallery, label_query, cam_gallery, cam_query):
    mAP = []
    junk_0 = np.argwhere(label_gallery == "-1").squeeze()
    ranks = np.zeros(distance.shape[1])

    for q in range(distance.shape[0]):
        # score = distance[q]
        score = distance.tolist()[q]

        good_image, junk = get_good_junk(q, label_gallery, label_query, cam_gallery, cam_query, junk_0)
        print('good_image: ',good_image)
        print('junk: ',junk)
        index = np.argsort(-(np.array(score)))
        # index = np.argsort(-score)
        rank = get_rank(good_image, junk, index)

        ap = get_ap(good_image, junk, index)

        mAP.append(ap)
        ranks[rank] += 1
    last_rank = 0
    mAP = np.mean(mAP)
    # mAP=round(mAP,4)
    for i in range(distance.shape[1]):
        last_rank += ranks[i]
    return  mAP

# 通过不同大小范围，不同大小的标注值代表不同线条粗细
def valueto(value):
    # alpha表示线条粗细程度
    if(abs(value)<=0.25):
        alpha=1
    if(abs(value) > 0.25 and abs(value) <= 0.5):
        alpha=2
    if (abs(value) > 0.5 and abs(value) <= 0.75):
        alpha = 4
    if (abs(value) > 0.75 and abs(value) <=1):
        alpha = 5

    return alpha


def effectiveBz(query_time,viper_tongji,probe_id,bzinfo,username):
    # 根据当前信息获取历史所有有效标注结果
    # 用一个npy文件保存

    path = viper_tongji + str(probe_id) + 'tongji' + '.npy'
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        result = []
        np.save(path, result)
    tongjiinfo = np.load(viper_tongji + str(probe_id) + 'tongji' + '.npy')
    tongjiinfo = tongjiinfo.tolist()

    temp = []
    for i in range(len(bzinfo)):
        idindex = int(bzinfo[i][0])
        for j in range(len(tongjiinfo)):
            idindex2 = int(tongjiinfo[j][0])
            if (idindex == idindex2):
                temp.append(j)


    # 如果存在重复的标注
    delinfo=[]
    if (len(temp) > 0):
        for i in range(len(temp)):
            delinfo.append(tongjiinfo[temp[i]])
    if(len(delinfo)>0):
        for i in range(len(delinfo)):
            tongjiinfo.remove(delinfo[i])


    # 获取统计信息
    for i in range(len(bzinfo)):
        idindex = int(bzinfo[i][0])
        bzinfo[i].append(query_time)
        bzinfo[i].append(username)
        tongjiinfo.append(bzinfo[i])
    tempbzinfo =tongjiinfo
    np.save(viper_tongji + str(probe_id) + 'tongji' + '.npy', tongjiinfo)
    newbzinfo=[]
    for i in range(len(tempbzinfo)):
        show=[]
        show.append((tempbzinfo[i][0]))
        show.append((tempbzinfo[i][1]))
        show.append((tempbzinfo[i][2]))
        newbzinfo.append(show)

    return newbzinfo,tongjiinfo



def effectiveBz_video(query_time,video_tongji,probe_id,bzinfo,username):
    # 根据当前信息获取历史所有有效标注结果
    # 用一个npy文件保存
    path = video_tongji + str(probe_id) + 'tongji' + '.npy'
    isExists = os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        result = []
        np.save(path, result)
    tongjiinfo = np.load(video_tongji + str(probe_id) + 'tongji' + '.npy')
    tongjiinfo = tongjiinfo.tolist()

    temp = []
    for i in range(len(bzinfo)):
        idindex = (bzinfo[i][0])
        for j in range(len(tongjiinfo)):
            idindex2 = (tongjiinfo[j][0])
            if (idindex == idindex2):
                temp.append(j)

    # 如果存在重复的标注
    if (len(temp) > 0):
        for i in range(len(temp)):
            del tongjiinfo[temp[i]]
    # 获取统计信息
    for i in range(len(bzinfo)):

        # bzinfo[i].append(query_time)
        # bzinfo[i].append(username)
        tongjiinfo.append(bzinfo[i])
    tempbzinfo =tongjiinfo
    np.save(video_tongji + str(probe_id) + 'tongji' + '.npy', tongjiinfo)

    newbzinfo=[]
    for i in range(len(tempbzinfo)):
        show=[]
        show.append(str(tempbzinfo[i][0]))
        show.append(float(tempbzinfo[i][1]))
        show.append(float(tempbzinfo[i][2]))
        newbzinfo.append(show)

    return newbzinfo


        # galnumber, posnumber, negnumber, torsonumber, legnumber = method.parse_tongjiinfo(tongjiinfo)
        # tongjilist = []
        # tongjilist.append(galnumber)
        # tongjilist.append(posnumber)
        # tongjilist.append(negnumber)
        # tongjilist.append(torsonumber)
        # tongjilist.append(legnumber)

def parsenewlist(bzinfo,tlist,llist):
    alpha = 0.6
    gallerynumber=len(tlist)
    bzscore = np.zeros(gallerynumber)
    for i in range(len(bzinfo)):
        Id = int(bzinfo[i][0])
        torso_value = float(bzinfo[i][1])
        leg_value = float(bzinfo[i][2])
        bzscore = bzscore + alpha * torso_value * np.array(tlist[Id - 1]) + (1 - alpha) * leg_value * np.array(llist[Id - 1])
    return bzscore

def parsenewlist_video(bzinfo,featuresList,tlist,llist):
    alpha = 0.6
    bzscore = np.zeros(29)
    bzscore_temp = np.zeros(29)
    for i in range(len(bzinfo)):
        for j in range(len(bzinfo[i])):
            Id= bzinfo[i][0]
            # imgname = bzinfo[i][0]
            # Id = featuresList.index(imgname + '.npy')
            torso_value = float(bzinfo[i][1])
            leg_value = float(bzinfo[i][2])
            bzscore = bzscore + alpha * torso_value * np.array(tlist[Id]) + (1 - alpha) * leg_value * np.array(
                llist[Id])
    return bzscore

def restart_remove(path):
    isExists = os.path.exists(path)
    if isExists:
        os.remove(path)
    return 0

def getmAP_Score(basepath):
    Scores = np.zeros((1, 29))
    Scorelist=os.listdir(basepath)
    for i in range(len(Scorelist)):
        tempScore=np.load(basepath+Scorelist[i])
        tempScore = mat(tempScore)
        Scores = np.row_stack((Scores, tempScore))
    # 删除第一行全0的向量
    Scores = np.delete(Scores, 0, axis=0)
    return Scores

def get_label_query(ItemId,username):
    video_mAP_ItemId = reid_root+'static/video_result/mAP/ItemId/' + username + '/'
    ItemId_path = video_mAP_ItemId +  'ID' + '.npy'
    # check ID.npy is exit
    isExists = os.path.exists(ItemId_path)
    if not isExists:
        idlist=[]
    else:
        idlist = np.load(ItemId_path)
        idlist=idlist.tolist()
    temp=[]
    temp.append(int(ItemId))
    idlist.append(temp)
    np.save(ItemId_path, idlist)
    return idlist

def get_cam_query(Id,username):
    video_mAP_camQuery = reid_root+'static/video_result/mAP/cam_query/' + username + '/'
    camQuery_path = video_mAP_camQuery +  'ID' + '.npy'
    # check ID.npy is exit
    isExists = os.path.exists(camQuery_path)
    if not isExists:
        idlist=[]
    else:
        idlist = np.load(camQuery_path)
        idlist=idlist.tolist()
    temp=[]
    temp.append(Id)
    idlist.append(temp)
    np.save(camQuery_path, idlist)
    return idlist


def make_log_txt(username,probe_name,bzinfo):
    gallerypath = reid_root+'static/Pic/gallery/'
    namelist = sorted(os.listdir(gallerypath))

    # namelist.sort(key=lambda x: int(x[0:3]))
    with open(reid_root+"static/result/log.txt", "a") as f:
        currentTime=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        f.write('###################################################\n')
        f.write('current time: '+currentTime + '\n')
        f.write('username: '+username+'\n')
        f.write('Probe: '+probe_name+'\n')
        for i in range(len(bzinfo)):
            index=int(bzinfo[i][0])
            torso=str(bzinfo[i][1])
            leg = str(bzinfo[i][2])
            galleryname=namelist[index-1].split('.')[0]
            f.write('sampleImage: '+galleryname+'  torso： '+torso+'  leg：'+leg+'\n')

# 这里是PCM14算法中，构建正负样本对
def bzinfo2bzset(info):
    box_type_tab,gallery_id_tab = get_box_type(info)

    feedback_set=[]
    for i in range(len(box_type_tab)):
        pos_ix_tmp = []
        neg_ix_tmp = []
        for j in range(len(box_type_tab[i])):

            pos_ix_tmp = [m+1 for m, x in enumerate(box_type_tab[i]) if x > 0]
            neg_ix_tmp = [m+1 for m, x in enumerate(box_type_tab[i]) if x < 0]
            # if(box_type_tab[i][j]>0):
            #     pos_ix_tmp.append(box_type_tab[i].index(box_type_tab[i][j])+1)
            # if(box_type_tab[i][j]<0):
            #     neg_ix_tmp.append(box_type_tab[i].index(box_type_tab[i][j])+1)
        # add probe into the feedback_set as positive
        pos_ix_tmp.append(0)
        # add last ranking galleries into the feedback_set as negative
        neg_ix_tmp.append(-1)
        pos_num = len(pos_ix_tmp);
        neg_num = len(neg_ix_tmp);

        if(pos_num>0 and neg_num>0):
            # get all possible pos - neg pairs
            temp=np.zeros((3,pos_num*neg_num))
            for m in range(pos_num):
                for n in range(neg_num):
                    if(pos_ix_tmp[m]==0):
                        pos_ix=0
                    else:
                        pos_ix=gallery_id_tab[pos_ix_tmp[m]-1]
                    if(neg_ix_tmp[n]==-1):
                        neg_ix=-1
                    else:
                        neg_ix = gallery_id_tab[neg_ix_tmp[n]-1]
                    temp[:,n+m*neg_num]=[i+1,pos_ix,neg_ix]

        feedback_set.append(temp)

    feedback_set_torso=np.array(feedback_set[0])


    feedback_set_leg = np.array(feedback_set[1])

    feedback_set = np.concatenate((feedback_set_torso,feedback_set_leg),axis=1)
    return feedback_set


def get_box_type(newList):
    beta = 0.05
    torso_array = []  # 所有标注值的上半身置信度
    leg_array = []  # 所有标注值的下半身置信度
    gallery_id_tab=[]
    box_type_tab=[[] for i in range(2)]
    for i in range(len(newList)):
        box_type = [0,0]  # 两个元素，若被标记置信度为正则为1，标记为负则为-1
        index = int(newList[i][0] ) # galleryid
        gallery_id_tab.append(index)
        torso_array.append(float(newList[i][1]))
        leg_array.append(float(newList[i][2]))
        if (torso_array[i] > beta):
            box_type_tab[0].append(1)
        elif (torso_array[i] < -1 * beta):
            box_type_tab[0].append(-1)
        else:
            box_type_tab[0].append(0)
        # 下半身
        if (leg_array[i] > beta):
            box_type_tab[1].append(1)
        elif (leg_array[i] < -1 * beta):
            box_type_tab[1].append(-1)
        else:
            box_type_tab[1].append(0)
    return box_type_tab,gallery_id_tab


# pcm14核心，matlab版本转换，这里需要使用pcm14.mat信息
def pcm14_core(info,IniScore,probe_id):
    # 将上一次得分中倒数21个样本均值作为伪负样本
    reid_score=IniScore.copy()
    index = np.argsort(-reid_score)
    neg_ix_group=(index[-21:])
    # 构建正负样本对
    feedback_set=bzinfo2bzset(info)
    tau=0.1
    pcm14 = scipy.io.loadmat(reid_root+'static/data/Init/pcm14.mat')
    g2g_dist=pcm14['g2g_dist']
    p2g_dist = pcm14['p2g_dist']
    g2p_dist = pcm14['g2p_dist']
    a,gallery_num,part_num=g2g_dist.shape
    if(len(feedback_set)==0):
        return IniScore
    g2g_sim=np.zeros((gallery_num,gallery_num,part_num))
    p2g_sim=np.zeros((1,gallery_num,part_num))
    for k in range((part_num)):
        g2g_dist_sym = 0.5 * (g2g_dist[:,:, k]+g2g_dist[:,:, k]);
        g2g_sim[:,:, k] = np.exp(-1 * g2g_dist_sym);
        p2g_dist_sym = 0.5 * (p2g_dist[probe_id-1,:, k]+g2p_dist[:, probe_id-1, k]);
        p2g_sim[0,:, k] = np.exp(-1 * p2g_dist_sym);

    feedback_pair_num=feedback_set.shape[1]

    for i in range(gallery_num):
        for j in range(feedback_pair_num):
            k = int(feedback_set[0, j]);
            pos_ix = int(feedback_set[1, j]);
            neg_ix = int(feedback_set[2, j]);

            if(pos_ix>0):
                pos_sim_score=g2g_sim[pos_ix-1, i, k-1]

            elif(pos_ix==0):
               # treat 0 as virtual positive(probe)
                pos_sim_score = p2g_sim[0, i, k-1]

            if (neg_ix > 0):
                neg_sim_score = g2g_sim[neg_ix - 1, i , k - 1]
            elif (neg_ix == -1):
                # treat - 1 as virtual negative(a group of most dissimilar galleries)
                neg_sim_score_temp = g2g_sim[neg_ix_group, i, k-1]
                neg_sim_score=(sum(neg_sim_score_temp) / len(neg_sim_score_temp))

            if (abs(pos_sim_score - neg_sim_score) > tau):
                reid_score[i] = reid_score[i] + pos_sim_score - neg_sim_score

    reid_score = np.array(min_Max(reid_score))
    return reid_score

# 中心视频PCM14
def pcm14_core_video(p2g_dist,info,IniScore,probe_id):
    # 将上一次得分中倒数21个样本均值作为伪负样本
    reid_score=IniScore.copy()
    index = np.argsort(-reid_score)
    neg_ix_group=(index[-5:])
    feedback_set=bzinfo2bzset(info)

    tau=0.01
    beta=10
    pcm14 = scipy.io.loadmat(reid_root+'static/data/Init/g2g_dist.mat')
    g2g_dist=pcm14['g2g_dist']
    # p2g_dist = pcm14['p2g_dist']
    # g2p_dist = pcm14['g2p_dist']
    a,gallery_num,part_num=g2g_dist.shape

    if(len(feedback_set)==0):
        return IniScore
    g2g_sim=np.zeros((gallery_num,gallery_num,part_num))
    p2g_sim=np.zeros((1,gallery_num,part_num))
    p2g_dist = np.array(p2g_dist)
    for k in range((part_num)):
        g2g_dist_sym = g2g_dist[:,:, k]
        g2g_sim[:,:, k] = np.exp(-1 * g2g_dist_sym)
    p2g_sim= np.exp(-p2g_dist)
    feedback_pair_num=feedback_set.shape[1]

    for i in range(gallery_num):
        for j in range(feedback_pair_num):
            k = int(feedback_set[0, j]);
            pos_ix = int(feedback_set[1, j]);
            neg_ix = int(feedback_set[2, j]);

            if(pos_ix>0):
                pos_sim_score=g2g_sim[pos_ix-1, i, k-1]

            elif(pos_ix==0):
               # treat 0 as virtual positive(probe)
                pos_sim_score = p2g_sim[0]

            if (neg_ix > 0):
                neg_sim_score = g2g_sim[neg_ix - 1, i , k - 1]
            elif (neg_ix == -1):
                # treat - 1 as virtual negative(a group of most dissimilar galleries)
                neg_sim_score_temp = g2g_sim[neg_ix_group, i, k-1]
                neg_sim_score=(sum(neg_sim_score_temp) / len(neg_sim_score_temp))

            if (abs(pos_sim_score - neg_sim_score) > tau):
                reid_score[i] = reid_score[i] + (pos_sim_score - neg_sim_score)*beta

    reid_score = np.array(min_Max(reid_score))
    return reid_score

# 将上一步图像分割的预处理的得到的几个.npy文件加载进去，运行pcm14算法
def pcm14_core_new(info,IniScore,probe_id):
    # 将上一次得分中倒数21个样本均值作为伪负样本
    reid_score=IniScore.copy()
    index = np.argsort(-reid_score)
    neg_ix_group=(index[-21:])
    feedback_set=bzinfo2bzset(info)
    tau=0.1
    g2g_tlist=np.load(reid_root+'static/data/data_result/'+dataset+'/' +'g2g_torso_dist.npy')
    g2g_llist=np.load(reid_root+'static/data/data_result/'+dataset+'/' +'g2g_leg_dist.npy')
    g2g_dist = np.zeros((len(g2g_tlist), len(g2g_llist), 2))
    g2g_dist[:, :, 0] = g2g_tlist
    g2g_dist[:, :, 1] = g2g_llist
    p2g_dist=np.load(reid_root+'static/data/data_result/'+dataset+'/p2g_dist.npy')
    gallery_num,gallery_num=g2g_tlist.shape
    part_num=2
    if(len(feedback_set)==0):
        return IniScore
    g2g_sim=np.zeros((gallery_num,gallery_num,part_num))
    p2g_sim=np.zeros((1,gallery_num,part_num))
    for k in range((part_num)):
        g2g_dist_sym = 0.5 * (g2g_dist[:,:,k]+g2g_dist[:,:,k]);
        g2g_sim[:,:,k] = np.exp(-1 * g2g_dist_sym);
    p2g_sim= np.exp(-p2g_dist)
    feedback_pair_num=feedback_set.shape[1]
    for i in range(gallery_num):
        for j in range(feedback_pair_num):
            k = int(feedback_set[0, j]);
            pos_ix = int(feedback_set[1, j]);
            neg_ix = int(feedback_set[2, j]);
            if(pos_ix>0):
                pos_sim_score=g2g_sim[pos_ix-1, i,k-1]
            elif(pos_ix==0):
               # treat 0 as virtual positive(probe)
                pos_sim_score = p2g_sim[0, i]
            if (neg_ix > 0):
                neg_sim_score = g2g_sim[neg_ix - 1, i, k - 1]
            elif (neg_ix == -1):
                # treat - 1 as virtual negative(a group of most dissimilar galleries)
                neg_sim_score_temp = g2g_sim[neg_ix_group, i,k-1]
                neg_sim_score=(sum(neg_sim_score_temp) / len(neg_sim_score_temp))
            if (abs(pos_sim_score - neg_sim_score) > tau):
                reid_score[i] = reid_score[i] + pos_sim_score - neg_sim_score
    reid_score = np.array(min_Max(reid_score))
    return reid_score


def fromname2id(name):
    name2id_dict = {
        '23_1555320665_1': 1,
        '15_1555320537_4': 1,
        '25_1555320537_1': 1,
        '18_1555320599_1': 1,
        '15_1555320537_2': 1,
        '23_1555320602_3': 1,
        '25_1555320721_1': 1,
        '23_1555320602_2': 1,
        '25_1555320537_2': 1,

        '23_1555320267_1': 2,

        '18_1555320660_1': 3,
        '15_1555320537_1': 3,
        '23_1555320665_3': 3,
        '15_1555320667_4': 3,
        '15_1555320605_1': 3,
        '23_1555320665_5': 3,

        '23_1555320602_1': 4,
        '23_1555320537_5': 4,

        '23_1555320537_1': 5,

        '23_1555320665_2': 6,
        '23_1555320537_2': 6,

        '23_1555320267_2': 7,

        '23_1555320080_1': 8,
        '23_1555320080_2': 8,

        '23_1555320537_4': 9,

        '15_1555320667_3': 10,

        '15_1555320537_5': 11,

        '15_1555320667_2': 12,

        '15_1555320667_1': 13,

    }
    name = srcList[i].split('_')
    galleryItem = name[0] + '_' + name[1] + '_' + name[2]
    for key in name2id_dict.keys():
        if (galleryItem == key):
            getItemId = name2id_dict[key]
    return getItemId

# 保存.mat结果
def resultprint(save_fedback_details , probe_id, probe_name,newList,username,dataset):

    feedback_details_all = np.load(save_fedback_details)
    feedback_details_all = feedback_details_all.tolist()
    feedback_details=feedback_details_all[probe_id-1]
    Inid=int(probe_id)
    # global feedback_details
    beta=0.05
    theta=0.25
    result2=[]             # probe信息
    torso_array = []       # 所有标注值的上半身置信度
    leg_array = []         # 所有标注值的下半身置信度
    torso_sgn = []
    leg_sgn = []
    result3=[]

    rectangle=np.load(reid_root+'static/data/data_result/'+dataset+'/'+'body_div.npy')
    gallerypath = reid_root+'static/Pic/gallery/'
    namelist = sorted(os.listdir(gallerypath))

    # namelist.sort(key=lambda x: int(x[0:3]))

    for i in range(len(newList)):
        if(newList[i][1]=='0'and newList[i][2]=='0'):
            continue
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

        index=int(newList[i][0]) # galleryid
        box_rect=rectangle[index-1]
        torso_array.append(float(newList[i][1]))
        leg_array.append(float(newList[i][2]))
        if ((torso_array[i]) > beta):
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
                'gallery_name':namelist[int(newList[i][0])-1].split('.')[0],
                'operator': operator,
            }
        feedback_details.append(dict1)
    feedback_details_all[Inid - 1]=feedback_details

    np.save(save_fedback_details , feedback_details_all)
    dict2={
        'probe_id':probe_id,
        'probe_name':probe_name,
    }


    # mkdir(path)

    dict3={
         'feedback_details': feedback_details,
        'probe_info': dict2,
     }
    savepath=reid_root+'static/result/mat_result/' + username + '/'+dataset+'/'
    scipy.io.savemat(savepath+probe_name+'.mat',
                     {'feedback_info':dict3,

                            })

def delbz(info,probe_id,username,dataset):
    # info = ['265', '0', '0']
    imgpath = reid_root+'static/Pic/gallery/'
    namelist = sorted(os.listdir(imgpath))
    filetype = '.' + os.listdir(reid_root+'static/Pic/gallery/')[0].split('.')[1]
    # namelist.sort(key=lambda x: int(x[0:3]))
    # get biaozhu_result
    path = reid_root+'static/result/fed_details/'
    feedback_details_all = np.load(path + username+'/'+dataset +'/'+ "feedback_details_all.npy")
    feedback_details_all = feedback_details_all.tolist()
    feedback_details = feedback_details_all[probe_id-1]
    index_list = []
    if(len(feedback_details)>0):
        for i in range(len(feedback_details)):
            galleryname = feedback_details[i]['gallery_name'] + filetype
            index = namelist.index(galleryname) + 1
            index_list.append(index)
        bzid = int(info[0])
        if (bzid in index_list):
            delId = (index_list.index(bzid))
            del feedback_details[delId]
            # del_feedback_details=feedback_details[delId]
            #
            # feedback_details.remove(del_feedback_details)
        feedback_details_all[probe_id-1]=feedback_details
        np.save(path + username+'/'+dataset +'/'+ "feedback_details_all.npy",feedback_details_all)
    return feedback_details

def updatemat(info,probe_id,username,dataset):
    # info = ['265', '0', '0']
    imgpath = reid_root+'static/Pic/gallery/'
    namelist = sorted(os.listdir(imgpath))
    filetype = '.' + os.listdir(reid_root+'static/Pic/gallery/')[0].split('.')[1]
    # namelist.sort(key=lambda x: int(x[0:3]))
    # get biaozhu_result
    path = reid_root+'static/result/fed_details/'
    feedback_details_all = np.load(path + username+'/'+dataset +'/'+ "feedback_details_all.npy")
    feedback_details_all = feedback_details_all.tolist()
    feedback_details = feedback_details_all[probe_id-1]
    index_list = []
    if(len(feedback_details)>0):
        for i in range(len(feedback_details)):
            galleryname = feedback_details[i]['gallery_name'] + filetype
            index = namelist.index(galleryname) + 1
            index_list.append(index)
        bzid = int(info[0])
        if (bzid in index_list):
            delId = (index_list.index(bzid))
            del feedback_details[delId]
            # del_feedback_details=feedback_details[delId]
            #
            # feedback_details.remove(del_feedback_details)
        feedback_details_all[probe_id-1]=feedback_details
        np.save(path + username+'/'+dataset +'/'+ "feedback_details_all.npy",feedback_details_all)
    return feedback_details