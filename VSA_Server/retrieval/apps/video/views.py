# _*_ coding:UTF-8 _*_
import base64
import json
import os
import numpy as np
from method.getDic import caldis
from django.http import HttpResponse
from django.shortcuts import render, redirect
import scipy.io
from method import utils
from method.utils import  *
sys.path.append('/home/cliang/mmap/VSA_Server')
from config import *
import cv2
# from method.feature_extractor import Extractor
from method.getInitialScore import getInitialScore
from method.getLastScore  import getLastScore
from method.utils import  min_Max
from Initial.settings import BASE_DIR
name2id_dict={
    '23_1555320665_1':1,
    '15_1555320537_4':1,
    '25_1555320537_1':1,
    '18_1555320599_1':1,
    '15_1555320537_2':1,
    '23_1555320602_3':1,
    '25_1555320721_1':1,
    '23_1555320602_2':1,
    '25_1555320537_2':1,

    '23_1555320267_1':2,

    '18_1555320660_1':3,
    '15_1555320537_1':3,
    '23_1555320665_3':3,
    '15_1555320667_4':3,
    '15_1555320605_1':3,
    '23_1555320665_5':3,

    '23_1555320602_1':4,
    '23_1555320537_5':4,

    '23_1555320537_1':5,

    '23_1555320665_2':6,
    '23_1555320537_2':6,

    '23_1555320267_2':7,

    '23_1555320080_1':8,
    '23_1555320080_2':8,

    '23_1555320537_4':9,

    '15_1555320667_3':10,

    '15_1555320537_5':11,

    '15_1555320667_2':12,

    '15_1555320667_1':13,

}

def index(request):
    return render(request, "video.html")

def cameraPath(request):
    value = int(request.POST.get("value"))
    video_list = []
    task_path = reid_root+'static/info/videos/'
    video_path = os.listdir(task_path)
    video_cam_number=video_path[value-1]
    return_path = reid_root+'static/info/videos/'+video_cam_number+'/'
    video_path_list=os.listdir(return_path)
    # 遍历
    for i in range(len(video_path_list)):
        videoCam_path = video_path_list[i]
        img_path = '/'+video_cam_number+'/'+videoCam_path
        video_list.append(img_path)
    video_list.sort()
    videos_return='static/info/videos'+video_list[-3]
    # videos_return='/'+video_cam_number+'/'+video_list[-3]
    videonames=video_list[-3]
    dict={}
    dict['videonames']=videonames
    dict['videos_return'] = videos_return
    print(videos_return)
    return HttpResponse(json.dumps(dict))

def query(request):
    import time
    print('start running')
    startTime = time.clock()
    query_time = int(request.POST.get("query_time"))
    total = request.POST.get("total")
    ItemId = request.POST.get("ItemId")
    username = request.session['user_name']
    videonames = request.POST.get("videonames")

    # probe_id
    probe_src=request.POST.get("probe_src")
    print(probe_src)
    # probe_src=probe_src[1:]
    split_probesrc=probe_src.split('/')
    probe_id=int(split_probesrc[3][0:-4])

    print('probe_id: ',probe_id)
    featuresList = os.listdir(reid_root+'static/Pic/video_feature')
    # g2gdist_torso,g2gdist_leg
    torsolist = scipy.io.loadmat(reid_root+'static/data/Init/'+'dist_torso.mat')
    leglist = scipy.io.loadmat(reid_root+'static/data/Init/'+'dist_leg.mat')
    tlist = torsolist['dist']
    llist = leglist['dist']
    tlist=np.exp(-tlist)
    llist=np.exp(-llist)
    gallerynumber=tlist.shape[0]
    # biaozhuresult
    # biaozhuresult
    biaozhupath = reid_root+'static/video_result/viper_bzresult/' + username + '/'
    # 记录每一个probe的查询次数和当前排名rank值
    video_biaozhu = reid_root+'static/video_result/biaozhu/' + username + '/'
    video_score = reid_root+'static/video_result/score/' + username + '/'
    video_tongji =reid_root+ 'static/video_result/tongji/' + username + '/'
    # mAP相关计算
    video_mAPscore=reid_root+'static/video_result/mAP/score/' + username + '/'
    video_mAP_ItemId=reid_root+'static/video_result/mAP/ItemId/' + username + '/'
    video_mAP_camQuery = reid_root+'static/video_result/mAP/cam_query/' + username + '/'
    video_mAP_mAP =reid_root+ 'static/video_result/mAP/mAP/' + username + '/'
    print(username)
    utils.mkdir(biaozhupath)
    utils.mkdir(video_biaozhu)
    utils.mkdir(video_score)
    utils.mkdir(video_tongji)
    utils.mkdir(video_mAPscore)
    utils.mkdir(video_mAP_ItemId)
    utils.mkdir(video_mAP_camQuery)
    utils.mkdir(video_mAP_mAP)
    bzpath = biaozhupath + str(probe_id) + '.npy'
    biaozhu = video_biaozhu + str(probe_id) + 'bzresult' + '.npy'
    video_show_mAP=video_mAP_mAP+'mAP'+'.npy'
    utils.viper_mkdir_load(bzpath,gallerynumber)
    utils.viper_biaozhu(biaozhu)
    utils.video_mAP_load(video_show_mAP)

    # probeFeature
    path=reid_root+probe_src
    print(path)
    img=cv2.imread(reid_root+probe_src)
    print(img)
    img = cv2.resize(img, (60, 110), interpolation=cv2.INTER_CUBIC)
    probe_feature=utils.LBP_features(img)
    probe_feature = np.mat(probe_feature)
    # galleryfeature
    galleryFeature = []
    for i in range(len(featuresList)):
        feature = np.load(reid_root+'static/Pic/video_feature/'+featuresList[i])
        galleryFeature.append(feature)
    galleryFeature = np.array(galleryFeature)
    gallerynumber = galleryFeature.shape[0]
    # getDis
    Dis=caldis(probe_feature,galleryFeature,gallerynumber)
    Dis = norm(Dis)
    p2g_dist = Dis.copy()
    Dis = np.array(Dis)

    # getInitialScore
    IniScore,Dis2Score=getInitialScore(query_time,biaozhupath,probe_id,Dis)
    print('IniScore: ',IniScore)
    np.save(video_mAPscore+ str(probe_id)+'.npy', IniScore)
    # argsort(score)
    if (query_time == 1):
        sortindex = np.argsort(-IniScore)
        ScoreMat=IniScore
        sortIniScore = []
        for i in range(len(sortindex)):
            sortIniScore.append(format(IniScore[sortindex[i]], '.3f'))

    if (query_time > 1):
        LastScore,bzinfo = getLastScore(p2g_dist,total, featuresList,IniScore, tlist, llist, biaozhupath, probe_id,biaozhu,query_time,Dis2Score,video_tongji,username)
        np.save(video_mAPscore+ str(probe_id)  + '.npy', LastScore)

        sortindex = np.argsort(-LastScore)
        ScoreMat=LastScore
        # sortIniScore和sortLastScore用于前端展示保留三位小数
        sortIniScore = []
        for i in range(len(IniScore)):
            sortIniScore.append(format(IniScore[sortindex[i]], '.3f'))
        sortLastScore = []
        for i in range(len(LastScore)):
            sortLastScore.append(format(LastScore[sortindex[i]], '.3f'))
    distance = sortindex

    # getSrcList
    srcList = []
    for i in range(len(distance)):
        numberv2=distance[i]
        namelistv2=featuresList[numberv2][0:-4]
        srcList.append(namelistv2)
    galItemId=[]
    for i in range(len(srcList)):
        name = srcList[i].split('_')
        galleryItem = name[0] + '_' + name[1] + '_' + name[2]
        for key in name2id_dict.keys():
            if (galleryItem == key):
                getItemId = name2id_dict[key]
                galItemId.append(getItemId)


    # 获取label_query,label_gallery,cam_gallery,cam_gallery计算mAP
    label_gallery = []
    label_query = get_label_query(ItemId,username)
    tempname = videonames.split('/')
    cam_query_id=int(tempname[1])
    cam_query=get_cam_query(cam_query_id,username)
    # cam_query = [cam_query_id]
    cam_gallery = []
    featuresList = os.listdir(reid_root+'static/Pic/video_feature')
    for i in range(len(featuresList)):
        label_query_name = featuresList[i].split('_')
        cam_gallery.append(int(label_query_name[0]))
        label_queryItem = label_query_name[0] + '_' + label_query_name[1] + '_' + label_query_name[2]
        for key in name2id_dict.keys():
            if (label_queryItem == key):
                label_gallery.append(name2id_dict[key])

    label_gallery=np.array(label_gallery).reshape([-1,1])
    label_query = np.array(label_query).reshape([-1,1])
    cam_gallery = np.array(cam_gallery).reshape([-1,1])
    cam_query = np.array(cam_query).reshape([-1,1])
    # ScoreMat=ScoreMat.reshape([1,-1])
    Score=getmAP_Score(video_mAPscore)
    print('mAPScore: ',Score.shape)
    mAP=evaluate(Score, label_gallery, label_query, cam_gallery, cam_query)
    showmAP=round(mAP,4)

    mAPpath=video_mAP_mAP  + 'mAP' + '.npy'
    mAPlist=np.load(mAPpath).tolist()

    mAPlist.append(showmAP)
    np.save(mAPpath,mAPlist)
    data = utils.qu2mAP(mAPlist)
    print('mAP: ',mAP)

    endTime = time.clock()
    time = float(endTime - startTime)

    if(query_time==1):
        dicdata = {}
        dicdata["src"] = srcList
        dicdata["galItemId"] = galItemId
        dicdata["time"] = time
        dicdata["probeid"] = probe_id
        dicdata["IniScore"] = sortIniScore
        dicdata["showmAP"] = showmAP
        dicdata["ItemId"] = ItemId
        dicdata["data"] = data
    if(query_time>1):
        dicdata = {}
        dicdata["src"] = srcList
        dicdata["galItemId"] = galItemId
        dicdata["time"] = time
        dicdata["probeid"] = probe_id
        dicdata["IniScore"] = sortIniScore
        dicdata["LastScore"] = sortLastScore
        dicdata["showmAP"] = showmAP
        dicdata["data"] = data
    return HttpResponse(json.dumps(dicdata))

def capture(request):
    srcTemp = request.POST.get("base64")

    srcTemp = srcTemp.split(',')
    src = srcTemp[1]
    imagedata = base64.b64decode(src)

    savePic_path=reid_root+'static/Pic/savePic'
    len_savePic=len(os.listdir(savePic_path))
    imgname=str(len_savePic+1)
    file = open(reid_root+'static/Pic/savePic/'+imgname+'.jpg', "wb")

    # file = open('static/Pic/savePic/1.jpg', "wb")
    file.write(imagedata)
    file.close()
    a = 0

    return HttpResponse(json.dumps(a))

def xianshi(request):
    username = request.session['user_name']
    x = (request.POST.get("x"))
    y = (request.POST.get("y"))
    w = (request.POST.get("w"))
    h = (request.POST.get("h"))
    x = int(x[:-2])
    y = int(y[:-2])
    w = int(w[:-2])
    h = int(h[:-2])
    # 先读取相应的视频帧图片
    framecapture = reid_root+'static/Pic/savePic'
    framelist = (os.listdir(framecapture))

    len_framelist = len(framelist)

    # framename=framelist[len_framelist-1]
    framename=str(len_framelist)+'.jpg'

    # 保存probe名字
    cropProbe = reid_root+'static/Pic/cropProbe'
    len_savePic = len(os.listdir(cropProbe))
    cropProname = str(len_savePic + 1)
    img = cv2.imread(reid_root+'static/Pic/savePic/'+framename)
    crop = img[y:(h + y), x:(w + x)]
    probesrc = 'static/Pic/cropProbe/'+cropProname+'.jpg'
    nercms_probesrc = '/probe/' + cropProname + '.jpg'
    cv2.imwrite(reid_root+'static/Pic/cropProbe/'+cropProname+'.jpg', crop)

    # 获取被截取的查询图像的行人Id
    videonames = (request.POST.get("videonames"))
    print(videonames)
    frameDisplay = int(request.POST.get("frameDisplay"))
    startframe=frameDisplay-30
    endframe=frameDisplay+30
    print('startframe: ' ,startframe)
    print('endframe: ' ,endframe)
    probe_x = int(x * 1.5)
    probe_y = int(y * 1.5)
    probe_w = int(w * 1.5)
    probe_h = int(h * 1.5)
    box1=[probe_x,probe_y,probe_x+probe_w,probe_y+probe_h]
    print('box1: ',box1)
    # 获取当前视频对应的txt跟踪结果
    txtpath=reid_root+'static/info/results'+videonames[0:-4]+'_res.txt'
    f=open(txtpath,'r')
    for line in f.readlines():
       line=line.split(' ')
       frame=int(line[0])

       if(frame>startframe and frame<endframe):
           box2 = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
           print(box2)
           if(utils.IOU(box1,box2)>0.5):
               showname=line[6]
               print(line[6])
               print(utils.IOU(box1,box2))
               picname=line[6].split('_')
               pipeiItem=picname[0]+'_'+picname[1]+'_'+picname[2]
               for key in name2id_dict.keys():
                   if(pipeiItem==key):
                       ItemId=name2id_dict[key]
                       print("probeId: ",ItemId)
               break
       else:
           continue

    # 每次点击end时候，令showmAP=[]
    video_mAP_mAP = reid_root+'static/video_result/mAP/mAP/' + username + '/'
    utils.mkdir(video_mAP_mAP)
    mAPpath = video_mAP_mAP + '/' + 'mAP' + '.npy'
    mAPlist=[]
    np.save(mAPpath,mAPlist)
    dicdata={}
    dicdata["probesrc"] = probesrc
    dicdata["ItemId"] = ItemId
    dicdata["showname"] = showname
    dicdata["nercms_probesrc"] = nercms_probesrc

    return HttpResponse(json.dumps(dicdata))

def show(request):
    gallery_show = []
    # 遍历
    username = request.session['user_name']
    gallerytempsrc=request.POST.get("gallerytemp_src")
    galfeaname = gallerytempsrc + '.npy'
    featuresList = os.listdir(reid_root+'static/Pic/video_feature/')
    index = featuresList.index(galfeaname)

    probe_src = request.POST.get("probe_src")
    probe_src = probe_src[1:]
    split_probesrc = probe_src.split('/')
    probe_id = (split_probesrc[3][0:-4])

    showlist=np.load(reid_root+'static/video_result/viper_bzresult/' + username + '/'+probe_id+'.npy')
    showlist=showlist.tolist()
    if(len(showlist[index])>=2):
        torso=showlist[index][-2]
        leg = showlist[index][-1]
    else:
        torso=0
        leg=0
    dicdata = {}
    dicdata["torso"] = torso
    dicdata["leg"] = leg
    return HttpResponse(json.dumps(dicdata))

def save(request):
    torso=request.POST.get("torso")
    leg = request.POST.get("leg")
    galleytempsrc = (request.POST.get("gallerytemp_src"))
    probe_src = (request.POST.get("probe_src"))
    username = request.session['user_name']

    probe_src = probe_src[1:]
    split_probesrc = probe_src.split('/')
    probe_id = (split_probesrc[3][0:-4])
    featuresList = os.listdir(reid_root+'static/Pic/video_feature/')
    bzindex=featuresList.index(galleytempsrc+'.npy')
    biaozhupath = reid_root+'static/video_result/viper_bzresult/' + username + '/'
    Mscore=np.load(biaozhupath+probe_id+'.npy')
    Mscore=Mscore.tolist()
    Mscore[bzindex].append(torso)
    Mscore[bzindex].append(leg)
    np.save(biaozhupath+probe_id+'.npy',Mscore)
    return HttpResponse(json.dumps(featuresList))

def pipeiTxt(x,y,w,h,videonames,frameDisplay):
    # 获取被截取的查询图像的行人Id
    x = int(x[:-2])
    y = int(y[:-2])
    w = int(w[:-2])
    h = int(h[:-2])
    startframe = frameDisplay - 30
    endframe = frameDisplay + 30
    print('startframe: ', startframe)
    print('endframe: ', endframe)
    probe_x = int(x * 1.5)
    probe_y = int(y * 1.5)
    probe_w = int(w * 1.5)
    probe_h = int(h * 1.5)
    box1 = [probe_x, probe_y, probe_x + probe_w, probe_y + probe_h]
    print('box1: ', box1)
    # 获取当前视频对应的txt跟踪结果
    txtpath = reid_root+'static/info/results' + videonames[0:-4] + '_res.txt'
    f = open(txtpath, 'r')
    for line in f.readlines():
        line = line.split(' ')
        frame = int(line[0])
        if (frame > startframe and frame < endframe):
            box2 = [int(line[1]), int(line[2]), int(line[3]), int(line[4])]
            if (utils.IOU(box1, box2) > 0.5):
                picname = line[6].split('_')
                pipeiItem = picname[0] + '_' + picname[1] + '_' + picname[2]
                for key in name2id_dict.keys():
                    if (pipeiItem == key):
                        ItemId = name2id_dict[key]
                        print("probeId: ", ItemId)
                break
        else:
            continue
    return ItemId




