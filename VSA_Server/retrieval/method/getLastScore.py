from method.parse_total import parse_totalv2,parse_total_viper
import numpy as np
from method.utils import *

# def getLastScore(total,featuresList,IniScore,tlist,llist,biaozhupath,probe_id,biaozhu):
#     bzinfo = parse_totalv2(total, probe_id, biaozhu)
#     alpha = 0.6
#     if (len(bzinfo) == 0):
#         LastScore = IniScore
#     # 一共有29个序列
#     else:
#         bzscore = np.zeros(29)
#         for i in range(len(bzinfo)):
#             for j in range(len(bzinfo[i])):
#                 imgname = bzinfo[i][0]
#                 Id = featuresList.index(imgname + '.npy')
#                 torso_value = float(bzinfo[i][1])
#                 leg_value = float(bzinfo[i][2])
#                 bzscore = bzscore + alpha*torso_value * np.array(tlist[Id]) + (1-alpha)*leg_value * np.array(llist[Id])
#         bzscore = np.array(min_Max(bzscore))
#         LastScore = IniScore + bzscore
#         LastScore = np.array(min_Max(LastScore))
#         np.save(biaozhupath + str(probe_id) + 'Score' + '.npy', LastScore)
#     return LastScore,bzinfo
def getLastScore(p2g_dist,total,featuresList,IniScore,tlist,llist,biaozhupath,probe_id,biaozhu,query_time,Dis2Score,video_tongji,username):
    bzinfo,oldlist = parse_totalv2(total, probe_id, biaozhu)

    if (query_time == 1):
        LastScore = IniScore
    if (query_time > 1):
        newbzinfo=effectiveBz_video(query_time, video_tongji, probe_id, bzinfo, username)

        LastScore = pcm14_core_video(p2g_dist,newbzinfo, IniScore, probe_id)
    np.save(biaozhupath + str(probe_id) + 'Score' + '.npy', LastScore)
    return LastScore, bzinfo

def getLastScore_viper(total,IniScore,tlist,llist,biaozhupath,probe_id,biaozhu,query_time,Dis2Score,viper_tongji,username):
    # parse total to bzInfo:[6,0.3,0.2],galleryid:6,torso:0.3,leg:0.2
    # 获取当前标注结果
    bzinfo,old_bzinfo = parse_total_viper(total, probe_id,biaozhu)

    # reid_score=parse_feedback(bzinfo, tlist, llist)
    # 若标注结果无效(反馈无变化，则重排结果不变)
    if (query_time == 1):
        LastScore = IniScore
    if (query_time > 1):
        newbzinfo, tongjiinfo = effectiveBz(query_time, viper_tongji, probe_id, bzinfo, username)
        # newbzinfo: [[20, -0.33, -0.53], [250, 0.0, 0.57], [223, -0.51, 0.7], [284, 0.0, 0.62]]
        # old_bzinfo: [['223', '-0.51', '0.7', '633_180', '5', 3, 'mmap'],['284', '0', '0.62', '819_135', '7', 3, 'mmap']]
        # delta_score=IniScore
        LastScore = pcm14_core(newbzinfo, IniScore, probe_id)


    # LastScore=np.array(min_Max(LastScore))
    np.save(biaozhupath + str(probe_id) + 'Score' + '.npy', LastScore)
    return LastScore,bzinfo,tongjiinfo

def getLastScore_new(total,IniScore,biaozhupath,probe_id,biaozhu,query_time,Dis2Score,viper_tongji,username):
    # 获取当前标注结果，将前端获取得来的total信息解析成标注信息
    bzinfo,old_bzinfo = parse_total_viper(total, probe_id,biaozhu)

    if (query_time == 1):
        LastScore = IniScore
    if (query_time > 1):
        # 每次判断有用的标注信息
        newbzinfo, tongjiinfo = effectiveBz(query_time, viper_tongji, probe_id, bzinfo, username)

        # 判断是否有标注信息，bzinfo代表当前标注信息
        if(len(bzinfo)==0):
            LastScore = IniScore
        else:
            LastScore = pcm14_core_new(newbzinfo, IniScore, probe_id)


    # LastScore=np.array(min_Max(LastScore))
    np.save(biaozhupath + str(probe_id) + 'Score' + '.npy', LastScore)
    return LastScore,bzinfo,tongjiinfo
