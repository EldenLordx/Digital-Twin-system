# _*_ coding:UTF-8 _*_
from django.shortcuts import render, redirect
from django.db import models
from django import forms
import hashlib
# from database.models import User
from django.contrib import auth
import base64
from .forms import UserForm,RegisterForm
# Create your views here.
from django.http import HttpResponse
import json
import cv2
from Initial.settings import BASE_DIR
import os,stat,sys
import datetime
import time
import numpy as np
sys.path.append('/home/cliang/mmap/VSA_Server')
from config import *
from method import utils
from method.getInitialScore import getInitialScore
from method.getLastScore  import getLastScore_viper,getLastScore_new
from method import utils
from PIL import Image, ImageDraw
from method.utils import resultprint
import scipy.io
from scipy.spatial.distance import pdist
from django.db.models import Q
from Initial.views import  *
from method.utils import delbz,updatemat
from config import *

# 登录函数
def login(request):
    if request.session.get('is_login', None):
        return redirect('/dataset_singleshot')
    if request.method == "POST":
        login_form = UserForm(request.POST)
        message = "请检查填写的内容！"
        if login_form.is_valid():
            username = login_form.cleaned_data['username']
            password = login_form.cleaned_data['password']
            auth_obj = auth.authenticate(request, username=username, password=password)
            if auth_obj:
                # 需要auth验证cookie
                auth.login(request, auth_obj)
            # 与数据库中的用户进行比对:
            try:
                user = models.User.objects.get(name=username)
                if user.password == password:  #和数据库内的值进行比对
                    request.session['is_login'] = True
                    request.session['user_id'] = user.id
                    request.session['user_name'] = user.name
                    return redirect('/dataset_singleshot/')
                else:
                    message = "密码不正确！"
            except:
                message = "用户不存在！"
        return render(request, 'dataset_singleshot/login.html', locals())

    login_form = UserForm()
    return render(request, 'dataset_singleshot/login.html', locals())

# 登出函数
def logout(request):
    if not request.session.get('is_login', None):
        return redirect('/login/')
    request.session.flush()
    return redirect('/login/')

# 基本主页，各功能网页以此为模板进行扩展
def base(request):
    return render(request, "base.html", )

# 注册函数
def register(request):
    # if request.session.get('is_login', None):
    #     # 登录状态不允许注册。你可以修改这条原则！
    #     return redirect("/index/")
    if request.method == "POST":
        register_form = RegisterForm(request.POST)
        message = "请检查填写的内容！"
        if register_form.is_valid():  # 获取数据
            username = register_form.cleaned_data['username']
            password1 = register_form.cleaned_data['password1']
            password2 = register_form.cleaned_data['password2']
            email = register_form.cleaned_data['email']
            sex = register_form.cleaned_data['sex']

            if password1 != password2:  # 判断两次密码是否相同
                message = "两次输入的密码不同！"
                return render(request, 'register.html', locals())
            else:
                same_name_user = User.objects.filter(name=username)
                if same_name_user:  # 用户名唯一
                    message = '用户已经存在，请重新选择用户名！'
                    return render(request, 'register.html', locals())
                same_email_user = User.objects.filter(email=email)
                if same_email_user:  # 邮箱地址唯一
                    message = '该邮箱地址已被注册，请使用别的邮箱！'
                    return render(request, 'register.html', locals())

                # 当一切都OK的情况下，创建新用户

                new_user = User.objects.create()
                new_user.name = username
                new_user.password = password1
                new_user.email = email
                new_user.sex = sex
                new_user.save()
                return redirect('/login/')  # 自动跳转到登录页面
    register_form = RegisterForm()
    return render(request, 'register.html', locals())

# 首页
def index(request):
    return render(request, "dataset_singleshot.html", )

# 加载并显示所有probe图像
def showPreview(request):
    probe_name = request.POST.get("probe_name")
    mylist = sorted(os.listdir(reid_root+'static/Pic/probe/'))
    # filetype表示数据集图片的类型，如.jpg,.bmp
    filetype='.'+mylist[0].split('.')[1]
    # mylist.sort(key=lambda x: int(x[0:3]))

    result = []
    for i in range(len(mylist)):
        result.append(mylist[i].split('.')[0])
    index = result.index(probe_name)
    probe_id = index + 1
    # 加载新的图片时，清空之前的标注结果
    username = request.session['user_name']
    viper_biaozhu = reid_root+'static/result/biaozhu/' + username + '/'+dataset+'/'
    temp_biaozhu = viper_biaozhu + str(probe_id) + 'bzresult.npy'

    viper_score = reid_root+'static/result/score/' + username + '/'+dataset+'/'
    temp_score = viper_score + 'score1.npy'

    viper_tongji = reid_root+'static/result/tongji/' + username + '/'+dataset+'/'
    temp_tongji = viper_tongji + str(probe_id) + 'tongji' + '.npy'

    viper_rank = reid_root+'static/result/viper_rank/' + username + '/'+dataset+'/'
    temp_rank = viper_rank + str(probe_id) + 'rank' + '.npy'

    viper_bzresult = reid_root+'static/result/viper_bzresult/' + username + '/'+dataset+'/'
    temp_bzresult1 = viper_bzresult + str(probe_id) + '.npy'
    temp_bzresult2 = viper_bzresult + str(probe_id) + 'Score' + '.npy'
    returnimage =reid_root+ 'static/Pic/returnimage/' + username + '/'+dataset+'/' + str(probe_id) + '/'
    isExists = os.path.exists(returnimage)
    if isExists:
        allimage = os.listdir(returnimage)
        for i in range(len(allimage)):
            utils.restart_remove(returnimage + allimage[i])

    # 每次加载图片时，清除当前图片之前所存的信息
    utils.restart_remove(temp_biaozhu)
    utils.restart_remove(temp_score)
    utils.restart_remove(temp_tongji)
    utils.restart_remove(temp_rank)
    utils.restart_remove(temp_bzresult1)
    utils.restart_remove(temp_bzresult2)

    dicdata = {}
    dicdata["probe_id"] = probe_id
    dicdata["filetype"] = filetype
    dicdata["dataset"] =dataset
    return HttpResponse(json.dumps(dicdata))

def load(request):
    img_list = []
    probe_list=np.load('imglist.npy')
    base_path = reid_root+'/static/Pic/probe/'  # 前端发送请求时的路径
    # 遍历
    for i in range(len(probe_list)):
        img_path = probe_list[i]
        img_list.append(base_path + img_path)

    return HttpResponse(json.dumps(img_list))

# 查询功能函数
def query(request):
    import time
    print('start running')
    username = request.session['user_name']
    startTime = time.clock()
    query_time = int(request.POST.get("query_time"))
    probe_name = request.POST.get("probe_name")
    probe_id = int(request.POST.get("probe_id"))
    total = request.POST.get("total")
    # 获取g2gdist_torso,g2gdist_leg两个矩阵
    tlist=np.load(reid_root+'static/data/data_result/'+dataset+'/' + 'g2g_torso_dist.npy')
    llist=np.load(reid_root+'static/data/data_result/'+dataset+'/' + 'g2g_leg_dist.npy')
    # 将距离转换为得分矩阵
    tlist = np.exp(-tlist)
    llist = np.exp(-llist)

    # 获取probe-gallery距离矩阵
    # p2gList = scipy.io.loadmat('p2g_dist.mat')
    # p2g_Dis = p2gList['p2g_dist']
    p2g_Dis=np.load(reid_root+'static/data/data_result/'+dataset+'/' + 'p2g_dist.npy')
    # 获取gallerynumber
    probenumber=p2g_Dis.shape[0]
    gallerynumber = p2g_Dis.shape[1]
    scoremat=np.exp(-p2g_Dis)
    Score_temp=scoremat[probe_id-1]
    Dis=-1*np.log(Score_temp)

    # 标注结果
    biaozhupath = reid_root+'static/result/viper_bzresult/' + username + '/'+dataset+'/'
    # 记录每一个probe的查询次数，当前排名rank值，统计信息等
    viper_rank = reid_root+'static/result/viper_rank/' + username + '/'+dataset+'/'
    viper_biaozhu = reid_root+'static/result/biaozhu/' + username + '/'+dataset+'/'
    viper_score = reid_root+'static/result/score/' + username + '/'+dataset+'/'
    viper_tongji =reid_root+'static/result/tongji/' + username + '/'+dataset+'/'
    mat_result=reid_root+'static/result/mat_result/' + username + '/'+dataset+'/'
    feedback_tails=reid_root+'static/result/fed_details/' + username + '/'+dataset+'/'

    utils.mkdir(biaozhupath)
    utils.mkdir(viper_rank)
    utils.mkdir(viper_biaozhu)
    utils.mkdir(viper_score)
    utils.mkdir(viper_tongji)
    utils.mkdir(mat_result)
    utils.mkdir(feedback_tails)

    bzpath = biaozhupath + str(probe_id) + '.npy'
    rankpath = viper_rank + str(probe_id) + 'rank' + '.npy'
    biaozhu = viper_biaozhu + str(probe_id) + 'bzresult' + '.npy'
    save_fedback_details=feedback_tails+  'feedback_details_all.npy'
    utils.viper_mkdir_load(bzpath,gallerynumber)
    utils.viper_mkdir_load(rankpath,gallerynumber)
    utils.viper_mkdir_load(save_fedback_details, gallerynumber)
    utils.viper_biaozhu(biaozhu)
    # 获取初始得分
    IniScore,Dis2Score= getInitialScore(query_time, biaozhupath, probe_id, Dis)
    # 将得分从大到小排序
    if(query_time==1):
        sortindex = np.argsort(-IniScore)
        sortIniScore = []
        for i in range(len(sortindex)):
            sortIniScore.append(format(IniScore[sortindex[i]],'.3f'))

    if(query_time > 1):
        # 获取当前得分
        # 防止没有标注值点击query的情况
        # if(len(total)==0):
        #     LastScore=IniScore
        LastScore,bzinfo,tongjiinfo = getLastScore_new(total, IniScore,  biaozhupath, probe_id,biaozhu,query_time,Dis2Score,viper_tongji,username)

        # 以下是关于.mat结果的保存
        if(len(bzinfo)>0):
            for i in range(len(bzinfo)):
                info=bzinfo[i]
                # 如果存在删除标注，则删除对应的.mat信息
                if((info[1]=='0')and int(info[2]=='0') ):
                    delbz(bzinfo[i], probe_id, username,dataset)
                else:
                # 如果同一个标注样本进行修改，则更新相应.mat信息
                    updatemat(bzinfo[i], probe_id, username,dataset)
        # resultprint()函数将feedback_details.npy转换成相应格式的.mat
        resultprint(save_fedback_details, probe_id, probe_name, bzinfo, username,dataset)

        sortindex = np.argsort(-LastScore)
        # 这里的sortIniScore,sortLastScore为显示在前端的排序得分，保留三位小数
        sortIniScore = []
        for i in range(len(IniScore)):
            sortIniScore.append(format(IniScore[sortindex[i]], '.3f'))
        sortLastScore = []
        for i in range(len(LastScore)):
            sortLastScore.append(format(LastScore[sortindex[i]], '.3f'))
    #  生成CMC曲线
    if(query_time==1):
        score=np.exp(-p2g_Dis)

        scorepath=reid_root+'static/result/score/'+username+'/'+dataset+'/'
        np.save(scorepath+'score1.npy',score)
        # 通过getcmcresult函数计算出cmc的值
        cmcresult=utils.getcmcresult(score, query_ids=list(range(1, probenumber+1)), gallery_ids=list(range(1, gallerynumber+1)),
                         query_cams=np.ones((probenumber)), gallery_cams=2 * np.ones((gallerynumber)), topk=gallerynumber,
                         separate_camera_set=False,
                         single_gallery_shot=False,
                         first_match_break=False)
        # 根据相应cmc的值绘制cmc曲线
        cmcsrc=utils.drawcmc1(cmcresult,gallerynumber)
    if(query_time>1):
        # 标注之前的得分
        scorepath = reid_root+'static/result/score/' + username + '/'+dataset+'/'
        newscore=np.load(scorepath+'score1.npy')
        # 通过getcmcresult函数计算出cmc的值
        cmcresult1 = utils.getcmcresult(newscore, query_ids=list(range(1, probenumber+1)), gallery_ids=list(range(1, gallerynumber+1)),
                                        query_cams=np.ones((probenumber)), gallery_cams=2 * np.ones((gallerynumber)), topk=gallerynumber,
                                        separate_camera_set=False,
                                        single_gallery_shot=False,
                                        first_match_break=False)

        # 标注之后的得分
        newscore[probe_id-1]=LastScore
        np.save(scorepath+'score1.npy',newscore)
        outputScore=np.load(scorepath+'score1.npy')

        cmcresult2 = utils.getcmcresult(outputScore, query_ids=list(range(1, probenumber+1)), gallery_ids=list(range(1, gallerynumber+1)),
                                   query_cams=np.ones((probenumber)), gallery_cams=2 * np.ones((gallerynumber)), topk=gallerynumber,
                                   separate_camera_set=False,
                                   single_gallery_shot=False,
                                   first_match_break=False)
        # 根据相应cmc的值绘制cmc曲线
        cmcsrc = utils.drawcmc2(cmcresult1,cmcresult2,gallerynumber)
    # getSrcList表示每一次查询排序后，得分从大到小的图片名，这是一个数组
    galleryid=sortindex+1
    groundTruth_id = np.load(reid_root+'static/data/data_result/'+dataset+'/' + 'picIdindex.npy')[probe_id - 1][1]
    for i in range(0, len(galleryid)):
        # 找出相应的groundTruth
        if (galleryid[i] == groundTruth_id):
            rank = i + 1
            print("Rank:", rank)
    if (query_time > 1):
        # 解析统计信息，统计正负样本个数，上下半身个数等
        galnumber, posnumber, negnumber, torsonumber, legnumber = utils.parse_tongjiinfo(tongjiinfo)
        tongjilist = []
        tongjilist.append(galnumber)
        tongjilist.append(posnumber)
        tongjilist.append(negnumber)
        tongjilist.append(torsonumber)
        tongjilist.append(legnumber)
    query2rank = np.load(rankpath)
    query2rank = query2rank.tolist()
    query2rank[probe_id-1].append(rank)
    np.save(rankpath, query2rank)

    # 绘制query2rank曲线
    ranklist=np.load(rankpath)
    showranklist=ranklist[probe_id-1]
    data=utils.qu2rank(showranklist)

    gallerypath=reid_root+'static/Pic/gallery/'
    namelist = sorted(os.listdir(gallerypath))

    # namelist.sort(key=lambda x: int(x[0:3]))
    srcList = []
    for i in range(len(namelist)):

        srcList.append(namelist[sortindex[i]])

    endTime = time.clock()
    time = float(endTime - startTime)
    galleryid = galleryid.tolist()
    # 找出哪些图片已经被标注，在标注保存后将它显示出来
    bzimgname,bzimgId=utils.biaozhuId(str(probe_id),username)
    if(query_time==1):
        dicdata = {}
        dicdata["src"] = srcList
        dicdata["probeid"] = probe_id
        dicdata['Rank'] = rank
        dicdata['Time'] = time
        dicdata["garid"] = galleryid
        dicdata["bzimgname"] = bzimgname
        dicdata["bzimgId"] = bzimgId
        dicdata["IniScore"] = sortIniScore
        dicdata["data"] = data
        dicdata["cmcsrc"]=cmcsrc
        dicdata["username"] = username
        dicdata["groundTruth_id"] = int(groundTruth_id)
    else:
        # 生成log.txt
        if (len(bzinfo) > 0):
            utils.make_log_txt(username, probe_name,  bzinfo)
        dicdata = {}
        dicdata["src"] = srcList
        dicdata["probeid"] = probe_id
        dicdata['Rank'] = rank
        dicdata['Time'] = time
        dicdata["garid"] = galleryid
        dicdata["bzimgname"] = bzimgname
        dicdata["bzimgId"] = bzimgId
        dicdata["IniScore"] = sortIniScore
        dicdata["LastScore"] = sortLastScore
        dicdata["data"] = data
        dicdata["cmcsrc"]=cmcsrc
        dicdata["tongjiinfo"]=tongjiinfo
        dicdata["tongjilist"] = tongjilist
        dicdata["username"] = username
        dicdata["groundTruth_id"] = int(groundTruth_id)

    return HttpResponse(json.dumps(dicdata))



def show(request):
    # show函数主要是为了显示点击标注图片，呈现当前标注样本的置信度

    # 遍历
    gallery_id=request.POST.get("galleryid")
    probe_id=request.POST.get("probe_id")
    gallery_name=request.POST.get("galleryname")
    probe_name = request.POST.get("probe_name")
    username = request.session['user_name']
    filetype='.'+os.listdir(reid_root+'static/Pic/gallery/')[0].split('.')[1]

    # rectangle为gallery图片集中每张图片的上下半身位置信息框
    rectangle = np.load(reid_root+'static/data/data_result/'+dataset+'/'+'body_div.npy')
    galleryList=sorted(os.listdir(reid_root+'static/Pic/gallery/'))
    # index表示该标注图片的galleryid号
    index=galleryList.index(gallery_name+filetype)+1

    # 获取被标注图片的上半身框的信息
    torso_bodyrec=[]
    torso_bodyrec.append(int(rectangle[index-1][0][0]))
    torso_bodyrec.append(int(rectangle[index-1][0][1]))
    torso_bodyrec.append(int(rectangle[index-1][0][2]))
    torso_bodyrec.append(int(rectangle[index-1][0][3]))

    # 获取被标注图片的下半身框的信息
    leg_bodyrec=[]
    leg_bodyrec.append(int(rectangle[index - 1][1][0]))
    leg_bodyrec.append(int(rectangle[index - 1][1][1]))
    leg_bodyrec.append(int(rectangle[index - 1][1][2]))
    leg_bodyrec.append(int(rectangle[index - 1][1][3]))
    biaozhupath = reid_root+'static/result/viper_bzresult/' + username + '/'+dataset+'/'
    showlist=np.load(biaozhupath+probe_id+'.npy')
    showlist=showlist.tolist()
    # 每次显示该标注图片的最新标注信息，即该数组的最后两位
    if(len(showlist[index-1])>=2):
        torso=showlist[index-1][-2]
        leg = showlist[index-1][-1]
    else:
        torso=0
        leg=0

    # 找出标注图片
    biaozhupath = reid_root+'static/Pic/returnimage/' + username + '/'+dataset+'/' + probe_id + '/'
    isExists = os.path.exists(biaozhupath)
    if not isExists:
        bzimgname=[]
    else:
        bzimgname=os.listdir(biaozhupath)
    gallery_name = gallery_name.split('?')
    gallery_name = gallery_name[0]
    gallery_name = gallery_name.split('.')
    gallery_name = gallery_name[0]

    gallerytuname=gallery_name+filetype

    # 如果该标注图片已经被标注，则在显示页面呈现出相应的红绿颜色框
    if(gallerytuname in bzimgname):
        imgsign=True
    else:
        imgsign=False

    dicdata = {}
    dicdata["torso"] = torso
    dicdata["leg"] = leg
    dicdata["torso_bodyrec"] = torso_bodyrec
    dicdata["leg_bodyrec"] = leg_bodyrec
    dicdata["imgsign"] = imgsign
    dicdata["galleryList"] = galleryList


    return HttpResponse(json.dumps(dicdata))

def save(request):
    # 保存每一次的上下半身标注结果
    torso=request.POST.get("torso")
    leg = request.POST.get("leg")
    galleyid= request.POST.get("galleryid")

    probe_id = request.POST.get("probe_id")
    gallerytemp_src=request.POST.get("gallerytemp_src")

    username = request.session['user_name']
    bzindex=int(galleyid)

    #Mscore用做显示上下半身的值
    biaozhupath = reid_root+'static/result/viper_bzresult/' + username + '/'+dataset+'/'
    Mscore=np.load(biaozhupath+probe_id+'.npy')
    Mscore=Mscore.tolist()
    Mscore[bzindex-1].append(torso)
    Mscore[bzindex-1].append(leg)
    np.save(biaozhupath+probe_id+'.npy',Mscore)
    bzSrc ,base64_data= todrawrectangle(probe_id, galleyid, gallerytemp_src, torso, leg,username)
    dicdata = {}
    dicdata["bzSrc"] = bzSrc
    dicdata["base64_data"] = base64_data
    dicdata["username"] = username
    return HttpResponse(json.dumps(dicdata))

def todrawrectangle(probe_id,galleryid,gallerytemp_src,torso,leg,username):
    # 给标注图片画框，通过标注置信度的正负性和大小显示出相应的颜色矩形框
    rectangle=np.load(reid_root+'static/data/data_result/'+dataset+'/'+'body_div.npy')
    filetype = '.' + os.listdir(reid_root+'static/Pic/gallery/')[0].split('.')[1]
    #galleryname获取被标注图片的图片名
    fname = reid_root+'static/Pic/gallery/' + gallerytemp_src + filetype
    path=reid_root+'static/Pic/returnimage/'+username+'/'+dataset+'/' + probe_id+'/'
    index=int(galleryid)
    utils.mkdir(path)
    img = cv2.imread(fname)
    img = cv2.resize(img, (64, 128))
    # 给不同标注值分配不同的粗细的线条宽度
    alpha1=utils.valueto(float(torso))
    alpha2 = utils.valueto(float(leg))
    if (float(torso) > 0):
        cv2.rectangle(img, (rectangle[index-1][0][0], rectangle[index-1][0][1]),(rectangle[index-1][0][0] + rectangle[index-1][0][2], rectangle[index-1][0][1] + rectangle[index-1][0][3]), (0, 255, 0),alpha1)
    if (float(torso) < 0):
        cv2.rectangle(img, (rectangle[index-1][0][0], rectangle[index-1][0][1]),(rectangle[index-1][0][0] + rectangle[index-1][0][2], rectangle[index-1][0][1] + rectangle[index-1][0][3]), (0, 0, 255),alpha1)
    if (float(leg) > 0):
        cv2.rectangle(img, (rectangle[index-1][1][0], rectangle[index-1][1][1]),(rectangle[index-1][1][0] + rectangle[index-1][1][2], rectangle[index-1][1][1] + rectangle[index-1][1][3]), (0, 255, 0),alpha2)
    if (float(leg) < 0):
        cv2.rectangle(img, (rectangle[index-1][1][0], rectangle[index-1][1][1]),(rectangle[index-1][1][0] + rectangle[index-1][1][2], rectangle[index-1][1][1] + rectangle[index-1][1][3]), (0, 0, 255),alpha2)
    cv2.imwrite(path+gallerytemp_src + filetype, img)
    base64_data=utils.image_to_base64(img,filetype)
    bzSrc = path + gallerytemp_src + filetype
    return bzSrc,base64_data

def parse_Bar(request):
    # 拖动滑动条，不同的上下半身标注值，呈现出不同颜色，不同线条粗细的位置框
    torso = request.POST.get("torso",0)
    leg = request.POST.get("leg",0)
    galleyid = request.POST.get("galleryid")
    probe_id = request.POST.get("probe_id")
    gallerytemp_src = request.POST.get("gallerytemp_src")
    username = request.session['user_name']
    bzindex = int(galleyid)

    bzSrc, base64_data = Bar_rectangle(probe_id, galleyid, gallerytemp_src, torso, leg, username)
    dicdata = {}
    dicdata["bzSrc"] = bzSrc
    dicdata["base64_data"] = base64_data
    dicdata["username"] = username
    return HttpResponse(json.dumps(dicdata))

def Bar_rectangle(probe_id,galleryid,gallerytemp_src,torso,leg,username):

    rectangle=np.load(reid_root+'static/data/data_result/'+dataset+'/'+'body_div.npy')
    # filetype为数据集图片名的类型（后缀）
    filetype='.'+os.listdir(reid_root+'static/Pic/gallery/')[0].split('.')[1]

    #galleryname获取被标注图片的图片名
    fname = reid_root+'static/Pic/gallery/' + gallerytemp_src + filetype
    path=reid_root+'static/Pic/returnimage/'+username+'/'+dataset+'/' + probe_id+'/'
    index=int(galleryid)
    utils.mkdir(path)
    img = cv2.imread(fname)
    img=cv2.resize(img, (64, 128))
    # 给不同标注值分配不同的粗细的线条宽度

    alpha1 = utils.valueto(float(torso))
    alpha2 = utils.valueto(float(leg))

    tor_x1=int((rectangle[index-1][0][0]))
    tor_y1=int((rectangle[index-1][0][1]))
    tor_x2=int((rectangle[index-1][0][0] + rectangle[index-1][0][2]))
    tor_y2=int((rectangle[index-1][0][1] + rectangle[index-1][0][3]))

    leg_x1 = int((rectangle[index - 1][1][0]))
    leg_y1 = int((rectangle[index - 1][1][1]))
    leg_x2 = int((rectangle[index - 1][1][0] + rectangle[index - 1][1][2]))
    leg_y2 = int((rectangle[index - 1][1][1] + rectangle[index - 1][1][3]))
    if (float(torso) > 0):
        cv2.rectangle(img, (tor_x1, tor_y1),(tor_x2, tor_y2), (0, 255, 0),alpha1)
    if (float(torso) < 0):
        cv2.rectangle(img, (tor_x1, tor_y1),(tor_x2, tor_y2), (0, 0, 255),alpha1)
    if (float(leg) > 0):
        cv2.rectangle(img, (leg_x1, leg_y1),(leg_x2, leg_y2), (0, 255, 0),alpha2)
    if (float(leg) < 0):
        cv2.rectangle(img, (leg_x1, leg_y1),(leg_x2, leg_y2), (0, 0, 255),alpha2)
    # 将生成的图片转换为base64格式，便于在前端页面实时改变更新
    base64_data=utils.image_to_base64(img,filetype)
    bzSrc = path + gallerytemp_src + filetype
    return bzSrc,base64_data


def zero_process(request):
    # 点击zero按钮实现，标注图片框的清除
    torso = request.POST.get("torso", 0)
    leg = request.POST.get("leg", 0)
    galleyid = request.POST.get("galleryid")
    probe_id = request.POST.get("probe_id")
    gallerytemp_src = request.POST.get("gallerytemp_src")
    username = request.session['user_name']

    bzSrc, base64_data = Bar_rectangle(probe_id, galleyid, gallerytemp_src, torso, leg, username)
    dicdata = {}
    dicdata["bzSrc"] = bzSrc
    dicdata["base64_data"] = base64_data
    dicdata["username"] = username
    return HttpResponse(json.dumps(dicdata))


def changeRec(request):
    # find biaozhuImage
    probe_id=request.POST.get("probe_id")
    gallery_name = request.POST.get("galleryname")
    username = request.POST.get("username")
    filetype='.'+os.listdir(reid_root+'static/Pic/gallery/')[0].split('.')[1]
    biaozhupath = reid_root+'static/Pic/returnimage/' + username + '/' +dataset+'/'+ probe_id + '/'
    isExists = os.path.exists(biaozhupath)
    if not isExists:
        bzimgname = []
    else:
        bzimgname = os.listdir(biaozhupath)
    gallery_name = gallery_name.split('?')
    gallery_name = gallery_name[0]
    gallery_name = gallery_name.split('.')
    gallery_name = gallery_name[0]

    gallerytuname = gallery_name + filetype


    if (gallerytuname in bzimgname):
        imgsign = True
    else:
        imgsign = False

    dicdata = {}
    dicdata["imgsign"] = imgsign
    return HttpResponse(json.dumps(dicdata))

def restart(request):
    # restart函数删除当前查询图片的所有保存信息，回到最开始的加载图片那一步
    username = request.session['user_name']
    probe_id = request.POST.get("probe_id")
    probe_name = request.POST.get("probe_name")
    viper_biaozhu = reid_root+'static/result/biaozhu/' + username + '/'+dataset+'/'
    temp_biaozhu = viper_biaozhu + str(probe_id)  + 'bzresult.npy'

    viper_score = reid_root+'static/result/score/' + username + '/'+dataset+'/'
    temp_score=viper_score+'score1.npy'

    viper_tongji = reid_root+'static/result/tongji/' + username + '/'+dataset+'/'
    temp_tongji=viper_tongji+str(probe_id) + 'tongji' + '.npy'

    viper_rank =reid_root+ 'static/result/viper_rank/' + username + '/'+dataset+'/'
    temp_rank = viper_rank + str(probe_id) + 'rank' + '.npy'

    viper_bzresult = reid_root+'static/result/viper_bzresult/' + username + '/'+dataset+'/'
    temp_bzresult1=viper_bzresult+str(probe_id) + '.npy'
    temp_bzresult2=viper_bzresult+str(probe_id) + 'Score' + '.npy'
    returnimage=reid_root+'static/Pic/returnimage/'+username+'/'+dataset+'/'+ str(probe_id)+'/'

    matpath=reid_root+'static/result/mat_result/' + username + '/'+dataset+'/'
    temp_mat=matpath+probe_name+'.mat'

    feedback_tails = reid_root+'static/result/fed_details/' + username + '/'+dataset+'/'
    save_fedback_details=feedback_tails+  'feedback_details_all.npy'


    # chmod，由于ubuntu删除信息需要权限，这里修改权限以便删除
    os.chmod(temp_biaozhu, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(temp_score, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(temp_tongji, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(temp_rank, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(temp_bzresult1, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(temp_bzresult2, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(temp_mat, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    os.chmod(save_fedback_details, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)

    # remove
    isExists = os.path.exists(returnimage)
    if isExists:
        allimage = os.listdir(returnimage)
        for i in range(len(allimage)):
            utils.restart_remove(returnimage + allimage[i])
    utils.restart_remove(temp_biaozhu)
    utils.restart_remove(temp_score)
    utils.restart_remove(temp_tongji)
    utils.restart_remove(temp_rank)
    utils.restart_remove(temp_bzresult1)
    utils.restart_remove(temp_bzresult2)
    utils.restart_remove(temp_mat)
    utils.restart_remove(save_fedback_details)
    return HttpResponse(json.dumps(0))

# 刷新功能
def refresh(request):
    a=1
    return HttpResponse(json.dumps(a))

