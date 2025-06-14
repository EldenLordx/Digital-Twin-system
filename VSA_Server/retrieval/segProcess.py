import os,stat,sys
import cv2
import numpy as np
import shutil
# from DeepLabV3plus.test_new import segmentation
sys.path.append('/home/cliang/mmap/VSA_Server')
from config import *
from segmentation.DeepLabV3plus import test_reid


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

def grayHist(imgpath):
    img = cv2.imread(imgpath)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    # 将列向量转换为行向量
    hist = hist.transpose()
    return hist

def calEuclideanDistance(vec1,vec2):
    dist = np.sqrt(np.sum(np.square(vec1 - vec2)))
    return dist


def extFea(imgpath):
    namelist = os.listdir(imgpath)
    featuresList = os.listdir(reid_root+'static/Pic/video_feature/')
    totalfea = np.zeros((1, 256))
    for i in range(len(featuresList)):
        name = featuresList[i].split('_')
        effectName = name[0] + "_" + name[1] + "_" + name[2] + "_"
        feature = np.zeros((1, 256))
        for j in range(len(namelist)):
            othername = namelist[j].split('_')
            nowname = othername[0] + '_' + othername[1] + '_' + othername[2] + '_'
            if (effectName == nowname):
                fea = grayHist(imgpath + namelist[j])
                feature = np.concatenate((feature, fea), axis=0)

                feature = feature.max(axis=0)
                feature = np.mat(feature)
        totalfea = np.concatenate((totalfea, feature), axis=0)

    totalfea = np.delete(totalfea, 0, axis=0)
    return totalfea


def extFeaimg(imgpath):
    namelist = sorted(os.listdir(imgpath))

    # namelist.sort(key=lambda x: int(x[0:3]))
    feature = np.zeros((1, 256))
    for i in range(len(namelist)):
        fea = grayHist(imgpath + namelist[i])
        feature = np.concatenate((feature, fea), axis=0)
    feature = np.delete(feature, 0, axis=0)
    return feature


def calg2gdist(fea):

    gallerynum=fea.shape[0]
    Dis = np.zeros((gallerynum,gallerynum))
    for i in range((gallerynum)):
        for j in range((gallerynum)):
            distance = calEuclideanDistance(fea[i], fea[j])
            Dis[i][j]=distance
    print(Dis)
    Dismax, Dismin = Dis.max(axis=0), Dis.min(axis=1)
    Dis = (Dis - Dismin) / (Dismax - Dismin)
    print(Dis)
    return Dis


def caldist(fea1,fea2):
    # number of probe
    pronum = fea1.shape[0]
    # number of gallery
    galnumber=fea2.shape[0]

    Dis = np.zeros((pronum, galnumber))
    for i in range((pronum)):
        for j in range((galnumber)):
            distance = calEuclideanDistance(fea1[i], fea2[j])
            Dis[i][j] = distance

    Dismax, Dismin = Dis.max(axis=0), Dis.min(axis=0)
    Dis = (Dis - Dismin) / (Dismax - Dismin)
    return Dis


def segResult_process(indexMapPath,imgpath):
    # 输入是索引图路径和原始图片路径

    # 索引图路径
    # indexMapPath='E:\PedParsing\D-segResult\indexMap\\'
    # 原始图片路径
    # imgpath='E:\PedParsing\img_resize\\'
    # 保存上下部件图片路径
    # torsopath='E:\PedParsing\\result\\torso\\'
    # legpath = 'E:\PedParsing\\result\\leg\\'
    torsopath = seg_root+'DeepLabV3plus/dataset/list/hl/torso/'+dataset+'/'
    legpath = seg_root+'DeepLabV3plus/dataset/list/hl/leg/'+dataset+'/'

    mkdir(torsopath)
    mkdir(legpath)
    result=sorted(os.listdir(imgpath))
    record_torso=[]
    record_leg=[]
    for i in range(len(result)):
        # 获取摄像头号+视频号+ID号
        imgname=result[i].split('_')
        # 读取原始图片
        img=cv2.imread(imgpath+result[i])
        img=cv2.resize(img,(64,128))
        # 读取分割索引图
        indexMap=np.load(indexMapPath+result[i].split('.')[0]+'.npy')
        ix = np.where(indexMap == 22)
        ix2=np.where(indexMap == 14)
        # 上衣部分
        if (len(ix[0]) > 100):
            indexmap1 = np.where(indexMap == 22, 1, 0)
            result_img0 = indexmap1 * img[:, :, 0]
            result_img1 = indexmap1 * img[:, :, 1]
            result_img2 = indexmap1 * img[:, :, 2]
            result_img = np.stack((result_img0, result_img1, result_img2),axis=2)
            cv2.imwrite(torsopath+result[i], result_img)

        else:
            indexmap1 = np.where(indexMap == 22, 0, 0)
            result_img0 = indexmap1 * img[:, :, 0]
            result_img1 = indexmap1 * img[:, :, 1]
            result_img2 = indexmap1 * img[:, :, 2]
            result_img = np.stack((result_img0, result_img1, result_img2), axis=2)
            cv2.imwrite(torsopath + result[i], result_img)

            # 找出所有缺失上衣部件的图片
            # imgname_temp:摄像头号+视频号+ID号
            # if(imgname_temp not in record_torso ):
            #     record_torso.append(imgname_temp)
            # print('record_torso:',record_torso)
            # np.save('record_torso.npy',record_torso)

        # 裤子部分
        if (len(ix2[0]) > 100):
            indexmap2 = np.where(indexMap == 14, 1, 0)
            result_img0 = indexmap2 * img[:, :, 0]
            result_img1 = indexmap2 * img[:, :, 1]
            result_img2 = indexmap2 * img[:, :, 2]
            result_img = np.stack((result_img0, result_img1, result_img2), axis=2)
            cv2.imwrite(legpath+result[i], result_img)

        else:
            indexmap2 = np.where(indexMap == 14, 0, 0)
            result_img0 = indexmap2 * img[:, :, 0]
            result_img1 = indexmap2 * img[:, :, 1]
            result_img2 = indexmap2 * img[:, :, 2]
            result_img = np.stack((result_img0, result_img1, result_img2), axis=2)
            cv2.imwrite(legpath + result[i], result_img)

            # 找出所有缺失裤子部件的图片
            # imgname_temp:摄像头号+视频号+ID号
            # if (imgname_temp not in record_leg):
            #     record_leg.append(imgname_temp)
            # print('record_leg: ',record_leg)
            # np.save('record_torso.npy', record_leg)
            # 上衣特征提取

    return torsopath,legpath


def seg2Dist(indexMapPath,imgpath):
    torsopath, legpath = segResult_process(indexMapPath, imgpath)
    fea_torso = extFeaimg(torsopath)

    fea_leg = extFeaimg(legpath)
    g2g_torso_dist = calg2gdist(fea_torso)
    g2g_leg_dist = calg2gdist(fea_leg)
    np.save(reid_root+'static/data/data_result/'+dataset+'/' + 'g2g_torso_dist.npy', g2g_torso_dist)
    np.save(reid_root+'static/data/data_result/'+dataset+'/' + 'g2g_leg_dist.npy', g2g_leg_dist)
    print('Get g2g_torso_dist! ',g2g_torso_dist.shape)
    print('Get g2g_leg_dist! ',g2g_leg_dist.shape)
    return g2g_torso_dist,g2g_leg_dist



def other2png(imgpath):
    filename=os.listdir(imgpath)
    changepath=seg_root+'DeepLabV3plus/dataset/list/hl/test/'
    for i in range(len(filename)):
        img=cv2.imread(changepath+filename[i])
        cv2.imwrite(changepath+filename[i][0:-4]+'.png',img)
    return changepath


def dist(probe_path,gallery_path):
    probeFea = extFeaimg(probe_path)
    galFea=extFeaimg(gallery_path)
    print('probeFea: ',probeFea.shape)
    print('galFea: ', galFea.shape)
    p2g_dis=caldist(probeFea,galFea)
    print('Get p2g_dist! ', p2g_dis.shape)
    np.save(reid_root+'static/data/data_result/'+dataset+'/' + 'p2g_dist.npy', p2g_dis)
    return p2g_dis

def getRec(indexMapPath):
    nameMap = sorted(os.listdir(indexMapPath))
    # nameMap.sort(key=lambda x: int(x[0:3]))
    body_rect = []
    for i in range(len(nameMap)):

        indexMap = np.load(indexMapPath + nameMap[i])

        tor = np.argwhere(indexMap == 22)
        # if there is wrong segmentation for torso
        if(len(tor)==0):
            # 上半身位置信息框，这里是坐标所以要加1
            tor_y2 = 0
            tor_y1 = 0
            tor_x2 = 0
            tor_x1 = 0
            tor_w1 = 0
            tor_h1 = 0
        else:
            # 上半身位置信息框，这里是坐标所以要加1
            tor_y2 = np.max(tor[:, 0]) + 1
            tor_y1 = np.min(tor[:, 0]) + 1
            tor_x2 = np.max(tor[:, 1]) + 1
            tor_x1 = np.min(tor[:, 1]) + 1
            tor_w1 = tor_x2 - tor_x1
            tor_h1 = tor_y2 - tor_y1

        leg = np.argwhere(indexMap == 14)
        # if there is wrong segmentation for leg

        if(len(leg)==0):
            # xia半身位置信息框，这里是坐标所以要加1
            leg_y2 = 0
            leg_y1 = 0
            leg_x2 = 0
            leg_x1 = 0
            leg_w2 = 0
            leg_h2 = 0
        else:
            # xia半身位置信息框，这里是坐标所以要加1
            leg_y2 = np.max(leg[:, 0]) + 1
            leg_y1 = np.min(leg[:, 0]) + 1
            leg_x2 = np.max(leg[:, 1]) + 1
            leg_x1 = np.min(leg[:, 1]) + 1
            leg_w2 = leg_x2 - leg_x1
            leg_h2 = leg_y2 - leg_y1


        result = [[] for i in range(2)]
        # 上半身框信息
        result[0].append(tor_x1)
        result[0].append(tor_y1)
        result[0].append(tor_w1)
        # result[0].append(45)
        result[0].append(tor_h1)
        # 下半身框信息
        result[1].append(leg_x1)
        result[1].append(leg_y1)
        result[1].append(leg_w2)
        # result[1].append(45)
        result[1].append(leg_h2)
        body_rect.append(result)
    np.save(reid_root+'static/data/data_result/'+dataset+'/'+'body_div.npy', body_rect)
    return body_rect

def delOld():

    segResult_path=seg_root+'DeepLabV3plus/dataset/list/hl/D-segResult/'
    indexMap_path=seg_root+'DeepLabV3plus/dataset/list/hl/indexMap/'
    torso_path=seg_root+'DeepLabV3plus/dataset/list/hl/torso/'
    leg_path=seg_root+'DeepLabV3plus/dataset/list/hl/leg/'

    shutil.rmtree(segResult_path)
    shutil.rmtree(indexMap_path)
    shutil.rmtree(torso_path)
    shutil.rmtree(leg_path)

    mkdir(segResult_path)
    os.chmod(segResult_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

    mkdir(indexMap_path)
    os.chmod(indexMap_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

    mkdir(torso_path)
    os.chmod(torso_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

    mkdir(leg_path)
    os.chmod(leg_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

def getIdindex(probepath,gallerypath):
    # get probe-gallery idIndex
    probelist = sorted(os.listdir(probepath))
    gallerylist = sorted(os.listdir(gallerypath))
    probe_number = len(probelist)
    gallery_number = len(gallerylist)
    picIdindex = np.zeros((probe_number, 2))
    for i in range((probe_number)):
        for j in range(gallery_number):
            probeId = probelist[i].split('_')[0]
            galleryId = gallerylist[j].split('_')[0]
            if (probeId == galleryId):
                picIdindex[i][0] = i + 1
                picIdindex[i][1] = j + 1
                continue

    picIdindex = picIdindex.astype(np.int)
    print('Get picIdindex!!!')
    np.save(reid_root+'static/data/data_result/'+dataset+'/' + 'picIdindex.npy', picIdindex)
    return picIdindex

# 输入原始图片路径
# inputPath='./dataset/list/hl/test/'

probepath=reid_root+'static/Pic/probe/'
gallerypath=reid_root+'static/Pic/gallery/'

# 输入分割结果中索引图路径
indexMapPath= test_reid.segmentation(gallerypath)
# 产生相对应数据集的保存文件夹
data_result=reid_root+'static/data/data_result/' +dataset+'/'
mkdir(data_result)
# 根据分割结果获取位置框信息
getRec(indexMapPath)
# 获取不同部件的距离矩阵g2g_torso_dist,g2g_leg_dist
g2g_torso_dist,g2g_leg_dist=seg2Dist(indexMapPath,gallerypath)
# 获取p2g_dist
p2g_dist=dist(probepath,gallerypath)
# 获取probe-gallery的ID关联:Picidindex
Picidindex=getIdindex(probepath,gallerypath)


