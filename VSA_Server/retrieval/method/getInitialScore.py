import os
import numpy as np
from method.utils import  *
def getInitialScore(query_time,biaozhupath,probe_id,Dis):
    # 这里的Dis是距离矩阵，要在这里转换为得分矩阵
    Dis2Score = np.exp(-Dis)
    if (query_time == 1):

        Inscore_path = biaozhupath + str(probe_id) + 'Score' + '.npy'

        # 检查初始得分是否存在
        isExists = os.path.exists(Inscore_path)
        if not isExists:
            IniScore = np.exp(-Dis)
            IniScore=np.array(min_Max(IniScore))
        else:
            IniScore = np.load(biaozhupath + str(probe_id) + 'Score' + '.npy')

        np.save(biaozhupath + str(probe_id) + 'Score' + '.npy', IniScore)


    if (query_time > 1):
        IniScore = np.load(biaozhupath + str(probe_id) + 'Score' + '.npy')
    IniScore=np.array(IniScore)
    return IniScore,Dis2Score