# _*_ coding:UTF-8 _*_
from django.shortcuts import render
import hashlib

# Create your views here.
from django.http import HttpResponse
import json
from SPF.settings import BASE_DIR
import os
import base64
import numpy as np
from . import utils
import cv2
# from method.feature_extractor import Extractor


reid_score1=[]
reid_score2=[]

def index(request):
    return render(request, "video.html", )

def base(request):
    return render(request, "base.html", )

def load(request):
    img_list = []
    # task_path = 'D:\\task\\26-Oct-2018-v2\\26-Oct-2018-o'
    task_path = os.path.join(BASE_DIR, 'static/Pic/probe/')
    task_mat_path = os.listdir(task_path)
    # probe_path = os.path.join(BASE_DIR, 'static/Pic/probe/')  # probe图片所在路径
    base_path = '/static/Pic/probe/'  # 前端发送请求时的路径
    # 遍历
    # probe_list = os.listdir(probe_path)
    for i in range(len(task_mat_path)):
        img_path = task_mat_path[i][:-4]
        img_path = img_path+'.jpg'
        img_list.append(base_path + img_path)
    username = request.session['user_name']
    path = 'viper_bzresult/' + username + '/'
    utils.mkdir_load(path,username)

    return HttpResponse(json.dumps(img_list))

def show(request):
    gallery_show = []
    # 遍历
    gallerytempsrc=request.POST.get("gallerytemp_src")
    galfeaname = gallerytempsrc + '.npy'
    featuresList = os.listdir('/home/mmap/huliang/pre_feature/')
    index = featuresList.index(galfeaname)

    probe_src = request.POST.get("probe_src")
    probe_src = probe_src[1:]
    split_probesrc = probe_src.split('/')
    probe_id = (split_probesrc[3][0:-4])
    showlist=np.load('static/bzresult/'+probe_id+'.npy')
    showlist=showlist.tolist()
    if(len(showlist[index])>=2):
        torso=showlist[index][-2]
        leg = showlist[index][-1]
    else:
        torso=0
        leg=0
    print(torso,leg)
    dicdata = {}
    dicdata["torso"] = torso
    dicdata["leg"] = leg
    return HttpResponse(json.dumps(dicdata))

def query(request):
    import time
    import scipy.io
    print('start running')
    startTime = time.clock()
    # probe_feature
    probe_src=request.POST.get("probe_src")
    gallerytempsrc = request.POST.get("gallerytemp_src")
    query_time = int(request.POST.get("query_time"))

    savenumber=int(request.POST.get("savenumber"))
    probe_src = request.POST.get("probe_src")
    torso = float(request.POST.get("torso"))
    print(torso)
    probe_src=probe_src[1:]
    split_probesrc=probe_src.split('/')
    probe_id=int(split_probesrc[3][0:-4])
    featuresList = os.listdir('/home/mmap/huliang/pre_feature/')

    # g2gdist_torso,g2gdist_leg
    torsolist = scipy.io.loadmat('/home/mmap/huliang/dist_torso.mat')
    leglist = scipy.io.loadmat('/home/mmap/huliang/dist_leg.mat')
    tlist = torsolist['dist']
    llist = leglist['dist']
    tlist=np.exp(-tlist)
    llist=np.exp(-llist)
    # bzscore
    # total = request.POST.get("total[]")
    # bzinfo=method.parse_total(total)
    # bzscore=np.zeros(29)
    # for i in range(len(bzinfo)):
    #     for j in range(bzinfo[i]):
    #         imgname=bzinfo[i][0]
    #         Id=featuresList.index(imgname+'.npy')
    #         torso_value=float(bzinfo[i][1])
    #         leg_value=float(bzinfo[i][2])
    #         bzscore=bzscore+torso_value*np.array(tlist[Id])+leg_value*np.array(llist[Id])

    # biaozhuresult
    biaozhupath='/home/mmap/huliang/web/supervisevideo new/static/bzresult/'
    bzpath=biaozhupath+str(probe_id)+'.npy'
    utils.mkdir_load(bzpath,probe_id)

    # probeFeature
    img=cv2.imread(probe_src)
    extractor = Extractor("../checkpoint/ckpt.t7", use_cuda=True)
    probe_feature = extractor(img)[0]

    # galleryfeature
    galleryFeature = []

    for i in range(len(featuresList)):
        feature = np.load('/home/mmap/huliang/pre_feature/' + featuresList[i])
        galleryFeature.append(feature)
    galleryFeature = np.array(galleryFeature)
    gallerynumber = galleryFeature.shape[0]

    # distance
    Dis = []
    Distemp=[]
    Matrix=[]
    for i in range((gallerynumber)):
        distance = calEuclideanDistance(probe_feature, galleryFeature[i])
        Dis.append(distance)
    # for i in range((gallerynumber)):
    #     temp = []
    #     for j in range(gallerynumber):
    #         distance = calEuclideanDistance(galleryFeature[i], galleryFeature[j])
    #         temp.append(distance)
    #     Matrix.append(temp)
    # np.save('static/Pic/score.npy',Matrix)

    LastScore=[]
    Dis=np.array(Dis)

    if(query_time==1):
        Inscore_path=biaozhupath+str(probe_id)+'Score'+'.npy'
        # check IniScore is exit
        isExists = os.path.exists(Inscore_path)
        if not isExists:
            IniScore=np.exp(-Dis)
        else:
            IniScore=np.load(biaozhupath+str(probe_id)+'Score'+'.npy')

        np.save(biaozhupath+str(probe_id)+'Score'+'.npy',IniScore)
    if(query_time > 1):
        IniScore=np.load(biaozhupath+str(probe_id)+'Score'+'.npy')

    # Matrix=np.array(Matrix)
    if (query_time > 1):

        total = request.POST.get("total")
        bzinfo = utils.parse_total(total)
        bzscore = np.zeros(29)
        for i in range(len(bzinfo)):
            for j in range(len(bzinfo[i])):
                imgname = bzinfo[i][0]
                Id = featuresList.index(imgname + '.npy')
                torso_value = float(bzinfo[i][1])
                leg_value = float(bzinfo[i][2])
                bzscore = bzscore + torso_value * np.array(tlist[Id]) + leg_value * np.array(llist[Id])
        LastScore=IniScore+bzscore
        np.save(biaozhupath+str(probe_id)+'Score'+'.npy',LastScore)
        print(LastScore)
    # if(savenumber>=query_time):
    #     if(probe_id==1):
    #         if (query_time == 1):
    #             Distemp = Dis
    #         elif (query_time == 2):
    #
    #             Dis[1] = Dis[1] - 0.02
    #             Dis[9] = Dis[9] - 0.03
    #             # Dis[20] = Dis[20] - 0.01
    #             # Dis[23] = Dis[23] - 0.04
    #             # Dis[11] = Dis[11] - 0.01
    #             # Dis[18] = Dis[18] - 0.02
    #             Dis[6] = Dis[6] - 0.05
    #             # Dis[28] = Dis[28] - 0.03
    #
    #             Dis[13] = Dis[13] - 0.02
    #             Dis[14] = Dis[14] - 0.01
    #             Dis[0] = Dis[0] - 0.01
    #
    #             Distemp = Dis
    #         elif (query_time == 3):
    #             # Dis[1] = Dis[1] - 0.03
    #             Dis[9] = Dis[9] - 0.02
    #             Dis[20] = Dis[20] - 0.01
    #             # Dis[23] = Dis[23] - 0.06
    #             # # Dis[11] = Dis[11] - 0.018
    #             # Dis[18] = Dis[18] - 0.02
    #             # Dis[6] = Dis[6] - 0.02
    #             # Dis[28] = Dis[28] - 0.06
    #
    #             Dis[13] = Dis[13] - 0.01
    #             Dis[14] = Dis[14] - 0.02
    #             Dis[0] = Dis[0] - 0.02
    #             Distemp = Dis
    #         else:
    #             Dis[1] = Dis[1] - 0.04
    #             Dis[9] = Dis[9] - 0.021
    #             Dis[20] = Dis[20] - 0.053
    #             Dis[23] = Dis[23] - 0.062
    #             Dis[11] = Dis[11] - 0.019
    #             Dis[18] = Dis[18] - 0.048
    #             Dis[6] = Dis[6] - 0.08
    #             Dis[28] = Dis[28] - 0.09
    #             Distemp = Dis
    #     if(probe_id==2):
    #         if (query_time == 1):
    #             Distemp = Dis
    #         elif (query_time == 2):
    #             # Dis[1] = Dis[1] - 0.02
    #             # Dis[9] = Dis[9] - 0.02
    #             Dis[20] = Dis[20] - 0.01
    #             Dis[23] = Dis[23] - 0.04
    #             # Dis[11] = Dis[11] - 0.01
    #             # Dis[18] = Dis[18] - 0.02
    #             # Dis[6] = Dis[6] - 0.05
    #             Dis[28] = Dis[28] - 0.05
    #
    #             Dis[13] = Dis[13] - 0.02
    #             Dis[14] = Dis[14] - 0.01
    #             Dis[0] = Dis[0] - 0.01
    #             # Dis[15] = Dis[15] - 0.06
    #             # Dis[12] = Dis[12] - 0.01
    #             # Dis[5] = Dis[5] - 0.02
    #
    #             Distemp = Dis
    #         elif (query_time == 3):
    #             Dis[1] = Dis[1] - 0.05
    #             Dis[9] = Dis[9] - 0.02
    #             Dis[20] = Dis[20] - 0.05
    #             Dis[23] = Dis[23] - 0.06
    #             Dis[11] = Dis[11] - 0.018
    #             # Dis[18] = Dis[18] - 0.04
    #             # Dis[6] = Dis[6] - 0.06
    #             # Dis[28] = Dis[28] - 0.06
    #
    #             Dis[13] = Dis[13] - 0.03
    #             Dis[14] = Dis[14] - 0.02
    #             Dis[0] = Dis[0] - 0.06
    #             Dis[15] = Dis[15] - 0.07
    #             Dis[12] = Dis[12] - 0.018
    #             # Dis[5] = Dis[5] - 0.04
    #             Distemp = Dis
    #         else:
    #             Dis[13] = Dis[13] - 0.08
    #             Dis[14] = Dis[14] - 0.025
    #             Dis[0] = Dis[0] - 0.053
    #             Dis[15] = Dis[15] - 0.062
    #             Dis[12] = Dis[12] - 0.019
    #             Dis[5] = Dis[5] - 0.048
    #
    #             Distemp = Dis
    #
    #     if(probe_id == 3):
    #         if (query_time == 1):
    #             Distemp = Dis
    #         elif (query_time == 2):
    #
    #             # Dis[1] = Dis[1] - 0.02
    #             # Dis[9] = Dis[9] - 0.03
    #             # Dis[20] = Dis[20] - 0.01
    #             # Dis[23] = Dis[23] - 0.04
    #             Dis[11] = Dis[11] - 0.01
    #             # Dis[18] = Dis[18] - 0.02
    #             Dis[6] = Dis[6] - 0.05
    #             # Dis[28] = Dis[28] - 0.03
    #
    #             Dis[13] = Dis[13] - 0.02
    #             Dis[14] = Dis[14] - 0.01
    #             Dis[0] = Dis[0] - 0.01
    #             Dis[7]=Dis[7]-0.03
    #             Distemp = Dis
    #         elif (query_time == 3):
    #             # Dis[1] = Dis[1] - 0.03
    #             # Dis[9] = Dis[9] - 0.03
    #             # Dis[20] = Dis[20] - 0.01
    #             # Dis[23] = Dis[23] - 0.06
    #             # # Dis[11] = Dis[11] - 0.018
    #             Dis[18] = Dis[18] - 0.02
    #             Dis[6] = Dis[6] - 0.02
    #             Dis[28] = Dis[28] - 0.06
    #
    #             Dis[13] = Dis[13] - 0.01
    #             Dis[14] = Dis[14] - 0.02
    #             Dis[0] = Dis[0] - 0.02
    #             Distemp = Dis
    #         else:
    #             Dis[1] = Dis[1] - 0.04
    #             Dis[9] = Dis[9] - 0.021
    #             Dis[20] = Dis[20] - 0.053
    #             Dis[23] = Dis[23] - 0.062
    #             Dis[11] = Dis[11] - 0.019
    #             Dis[18] = Dis[18] - 0.048
    #             Dis[6] = Dis[6] - 0.08
    #             Dis[28] = Dis[28] - 0.09
    #             Distemp = Dis
    #     if(probe_id == 4):
    #         if (query_time == 1):
    #             Distemp = Dis
    #         elif (query_time == 2):
    #             # Dis[1] = Dis[1] - 0.02
    #             # Dis[8] = Dis[8] - 0.02
    #             Dis[20] = Dis[20] - 0.01
    #             Dis[23] = Dis[23] - 0.04
    #             # Dis[10] = Dis[10] - 0.01
    #             # Dis[18] = Dis[18] - 0.02
    #             # Dis[6] = Dis[6] - 0.05
    #             # Dis[28] = Dis[28] - 0.05
    #
    #             Dis[13] = Dis[13] - 0.02
    #             Dis[3] = Dis[3] - 0.01
    #             Dis[0] = Dis[0] - 0.01
    #             # Dis[15] = Dis[15] - 0.06
    #             # Dis[12] = Dis[12] - 0.01
    #             # Dis[5] = Dis[5] - 0.02
    #
    #             Distemp = Dis
    #         elif (query_time == 3):
    #             Dis[7] = Dis[7] - 0.05
    #             Dis[9] = Dis[9] - 0.02
    #             Dis[2] = Dis[2] - 0.05
    #             Dis[23] = Dis[23] - 0.06
    #             Dis[1] = Dis[1] - 0.018
    #             # Dis[18] = Dis[18] - 0.04
    #             # Dis[6] = Dis[6] - 0.06
    #             # Dis[28] = Dis[28] - 0.06
    #
    #             Dis[13] = Dis[13] - 0.03
    #             Dis[14] = Dis[14] - 0.02
    #             # Dis[0] = Dis[0] - 0.06
    #             # Dis[15] = Dis[15] - 0.07
    #             # Dis[12] = Dis[12] - 0.018
    #             # Dis[5] = Dis[5] - 0.04
    #             Distemp = Dis
    #         else:
    #             Dis[13] = Dis[13] - 0.08
    #             Dis[14] = Dis[14] - 0.025
    #             Dis[0] = Dis[0] - 0.053
    #             # Dis[15] = Dis[15] - 0.062
    #             # Dis[12] = Dis[12] - 0.019
    #             # Dis[5] = Dis[5] - 0.048
    #
    #             Distemp = Dis
    #     if(probe_id == 5):
    #         if (query_time == 1):
    #             Distemp = Dis
    #         elif (query_time == 2):
    #
    #             # Dis[1] = Dis[1] - 0.02
    #             # Dis[9] = Dis[9] - 0.03
    #             # Dis[20] = Dis[20] - 0.01
    #             # Dis[23] = Dis[23] - 0.04
    #             Dis[11] = Dis[11] - 0.01
    #             # Dis[18] = Dis[18] - 0.02
    #             Dis[6] = Dis[6] - 0.05
    #             # Dis[28] = Dis[28] - 0.03
    #
    #             Dis[13] = Dis[13] - 0.02
    #             Dis[14] = Dis[14] - 0.01
    #             Dis[0] = Dis[0] - 0.01
    #             Dis[7] = Dis[7] - 0.03
    #             Distemp = Dis
    #         elif (query_time == 3):
    #             # Dis[1] = Dis[1] - 0.03
    #             # Dis[9] = Dis[9] - 0.03
    #             # Dis[20] = Dis[20] - 0.01
    #             # Dis[23] = Dis[23] - 0.06
    #             # # Dis[11] = Dis[11] - 0.018
    #             Dis[18] = Dis[18] - 0.02
    #             Dis[6] = Dis[6] - 0.02
    #             Dis[28] = Dis[28] - 0.06
    #
    #             Dis[13] = Dis[13] - 0.01
    #             Dis[14] = Dis[14] - 0.02
    #             Dis[0] = Dis[0] - 0.02
    #             Distemp = Dis
    #         else:
    #             Dis[1] = Dis[1] - 0.04
    #             Dis[9] = Dis[9] - 0.021
    #             Dis[20] = Dis[20] - 0.053
    #             Dis[23] = Dis[23] - 0.062
    #             Dis[11] = Dis[11] - 0.019
    #             Dis[18] = Dis[18] - 0.048
    #             Dis[6] = Dis[6] - 0.08
    #             Dis[28] = Dis[28] - 0.09
    #             Distemp = Dis
    #     if(probe_id == 6):
    #         if (query_time == 1):
    #             Distemp = Dis
    #         elif (query_time == 2):
    #             # Dis[1] = Dis[1] - 0.02
    #             # Dis[8] = Dis[8] - 0.02
    #             Dis[20] = Dis[20] - 0.01
    #             Dis[23] = Dis[23] - 0.04
    #             # Dis[10] = Dis[10] - 0.01
    #             # Dis[18] = Dis[18] - 0.02
    #             # Dis[6] = Dis[6] - 0.05
    #             # Dis[28] = Dis[28] - 0.05
    #
    #             Dis[13] = Dis[13] - 0.02
    #             Dis[3] = Dis[3] - 0.01
    #             Dis[0] = Dis[0] - 0.01
    #             # Dis[15] = Dis[15] - 0.06
    #             # Dis[12] = Dis[12] - 0.01
    #             # Dis[5] = Dis[5] - 0.02
    #
    #             Distemp = Dis
    #         elif (query_time == 3):
    #             Dis[7] = Dis[7] - 0.05
    #             Dis[9] = Dis[9] - 0.02
    #             Dis[2] = Dis[2] - 0.05
    #             Dis[23] = Dis[23] - 0.06
    #             Dis[1] = Dis[1] - 0.018
    #             # Dis[18] = Dis[18] - 0.04
    #             # Dis[6] = Dis[6] - 0.06
    #             # Dis[28] = Dis[28] - 0.06
    #
    #             Dis[13] = Dis[13] - 0.03
    #             Dis[14] = Dis[14] - 0.02
    #             # Dis[0] = Dis[0] - 0.06
    #             # Dis[15] = Dis[15] - 0.07
    #             # Dis[12] = Dis[12] - 0.018
    #             # Dis[5] = Dis[5] - 0.04
    #             Distemp = Dis
    #         else:
    #             Dis[13] = Dis[13] - 0.08
    #             Dis[14] = Dis[14] - 0.025
    #             Dis[0] = Dis[0] - 0.053
    #             # Dis[15] = Dis[15] - 0.062
    #             # Dis[12] = Dis[12] - 0.019
    #             # Dis[5] = Dis[5] - 0.048
    #
    #             Distemp = Dis
    #     if(probe_id>4):
    #         Distemp = Dis

    # distance_temp = np.argsort(Distemp)
    if(query_time==1):
        distance_temp = np.argsort(-IniScore)
    if (query_time > 1):
        distance_temp = np.argsort(-LastScore)

    distance = distance_temp

    basepath = '/home/mmap/work/RealTimeCamera/results/'
    srcList = []
    giflist=[]
    for i in range(len(distance)):
        numberv2=distance[i]
        namelistv2=featuresList[numberv2][0:-4]
        srcList.append(namelistv2)

    # for i in range(len(distance)):
    #     number = distance[i]
    #     print(featuresList[number])
    #     gallerycam, gallid = name2id(featuresList[number])
    #     for j in range(len(featuresList)):
    #         feacam, feaid = name2id(featuresList[j])
    #         if ((gallerycam == feacam) and (gallid == feaid)):
    #             featurename = featuresList[j]
    #             imgname = featurename[0:-4] + '.jpg'
    #             srcList.append(imgname)
    # query_time = int(request.POST.get("query_time"))
    # print("query_time: ", query_time)

    # make gif

    # for i in range(len(srcList)):
    #
    #     namelist=srcList[i]
    #     gifname=namelist
    #     namelist = namelist.split('_')
    #     txtpath = basepath + namelist[0] + '/' + namelist[1] + '_res.txt'
    #     giflist = []
    #     with open(txtpath, 'r') as f:
    #         for line in f.readlines():
    #             olist = line.strip().split(' ')
    #             if (len(olist) != 1):
    #                 id = olist[5]
    #                 if (id == namelist[2]):
    #                     giflist.append(olist[6])
    #                     # srcList.append(olist[6])
    #     gifpath='static/Pic/gif/'
    #
    #     giflistpath=tianjia(giflist)
    #     if(len(giflistpath)>500):
    #         giflistpath=giflistpath[0:200]
    #
    #     creat_gif(giflistpath, gifpath+gifname+'.gif', duration=0.05)

    endTime = time.clock()
    time = float(endTime - startTime)
    dicdata = {}
    dicdata["src"] = srcList
    dicdata["time"] = time
    dicdata["probeid"] = probe_id
    return HttpResponse(json.dumps(dicdata))

def save(request):
    torso=request.POST.get("torso")
    leg = request.POST.get("leg")
    galleytempsrc = (request.POST.get("gallerytemp_src"))
    probe_src = (request.POST.get("probe_src"))
    probe_src = probe_src[1:]
    split_probesrc = probe_src.split('/')
    probe_id = (split_probesrc[3][0:-4])

    featuresList = os.listdir('/home/mmap/huliang/pre_feature/')
    bzindex=featuresList.index(galleytempsrc+'.npy')

    Mscore=np.load('static/bzresult/'+probe_id+'.npy')
    Mscore=Mscore.tolist()
    Mscore[bzindex].append(torso)
    Mscore[bzindex].append(leg)
    np.save('static/bzresult/'+probe_id+'.npy',Mscore)

    # galleryname=method.todrawrectangle(probe_id,galleryid,torso,leg,username)

    a=0

    return HttpResponse(json.dumps(0))

# def showFeedback(request):
#
#     username = request.session['user_name']
#     path = 'viper_bzresult/' + username + '/'
#     checkStr = np.load(path + "checkStr.npy")
#     checkStr = checkStr.tolist()
#     torsoScore = np.load(path + "torsoScore.npy")
#     torsoScore = torsoScore.tolist()
#     legScore = np.load(path + "legScore.npy")
#     legScore = legScore.tolist()
#
#     dicdata={}
#
#     dicdata["checkStr"] = checkStr
#     dicdata["torsoScore"] = torsoScore
#     dicdata["legScore"] = legScore
#     dicdata["username"] = username
#     return HttpResponse(json.dumps(dicdata))

def camera(request):
    value = int(request.POST.get("value"))
    video_list = []
    # task_path = 'D:\\task\\26-Oct-2018-v2\\26-Oct-2018-o'
    task_path = os.path.join(BASE_DIR, '/home/mmap/work/RealTimeCamera/videos/')
    video_path = os.listdir(task_path)
    video_cam_number=video_path[value-1]

    return_path = '/home/mmap/work/RealTimeCamera/videos/'+video_cam_number+'/'
    video_path_list=os.listdir(return_path)

    # 遍历
    for i in range(len(video_path_list)):
        videoCam_path = video_path_list[i]
        img_path = '/'+video_cam_number+'/'+videoCam_path
        video_list.append(img_path)
    video_list.sort()
    return HttpResponse(json.dumps(video_list[-3]))


def capture(request):

    srcTemp = request.POST.get("base64")
    srcTemp = srcTemp.split(',')
    src = srcTemp[1]
    imagedata = base64.b64decode(src)

    savePic_path='static/Pic/savePic'
    len_savePic=len(os.listdir(savePic_path))
    imgname=str(len_savePic+1)
    file = open('static/Pic/savePic/'+imgname+'.jpg', "wb")

    # file = open('static/Pic/savePic/1.jpg', "wb")
    file.write(imagedata)
    file.close()
    a = 0

    return HttpResponse(json.dumps(a))

def xianshi(request):
    x = (request.POST.get("x"))
    y = (request.POST.get("y"))
    w = (request.POST.get("w"))
    h = (request.POST.get("h"))

    x = int(x[:-2])
    y = int(y[:-2])
    w = int(w[:-2])
    h = int(h[:-2])
    # 先读取相应的视频帧图片
    framecapture = 'static/Pic/savePic'
    framelist = (os.listdir(framecapture))

    len_framelist = len(framelist)

    # framename=framelist[len_framelist-1]
    framename=str(len_framelist)+'.jpg'

    # 保存probe名字
    cropProbe = 'static/Pic/cropProbe'
    len_savePic = len(os.listdir(cropProbe))
    cropProname = str(len_savePic + 1)

    img = cv2.imread('static/Pic/savePic/'+framename)

    crop = img[y:(h + y), x:(w + x)]
    probesrc = 'static/Pic/cropProbe/'+cropProname+'.jpg'
    cv2.imwrite(probesrc, crop)
    return HttpResponse(json.dumps(probesrc))


def calEuclideanDistance(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist



