# def avgpos(posgroup):
#     for i in range(len(posgroup)):
#         for j in range(len(posgroup[i])):
#             if posgroup[i][j]:
#                 pos = posgroup[i][j]
#                 sumx, sumy = 0, 0
#                 for m in range(len(pos)):
#                     sumx += pos[m][0]
#                     sumy += pos[m][1]
#                 posgroup[i][j]=[sumx / len(pos), sumy / len(pos)]
#     return posgroup


def nearbypos(posgroup):
    camerapos = [[2, -21], [-2, 21], [-38, -29], [38, 29], [38, -29], [-2, -21], [-38, 29], [2, 21], [2, 5], [2, -5],
                 [0, -24], [0, 24]]
    for i in range(len(posgroup)):
        for j in range(len(posgroup[i])):
            dislist=[]
            if posgroup[i][j]:
                pos = posgroup[i][j]
                # print('nearby', pos)
                for m in range(len(pos)):
                    disx = pos[m][0] - camerapos[pos[m][2]][0]
                    disy = pos[m][1] - camerapos[pos[m][2]][1]
                    dis = (disx ** 2 + disy ** 2) ** (1 / 2)
                    dislist.append(dis)
                minn = dislist.index(min(dislist))
                posgroup[i][j]=[pos[minn][0], pos[minn][1] ,pos[minn][3]] 
    return posgroup

file = "./camera2/"
# f = ['-1.txt', '-2.txt', '-3.txt', '-4.txt']
No = [17, 18, 19, 20, 21, 23, 24, 25, 27, 28, 35, 36]

txtnum = 0
N=20
while txtnum<4:
    txtnum+=1
    suffix='-'+str(txtnum)+'.txt'
    posgroup = [None]*100
    for i in range(len(posgroup)):
        posgroup[i] = [None]*1201
    for i in range(12):
        # 遍历12个摄像头
        text = open(file + str(No[i]) + suffix)
        lines = text.readlines()
        for line in lines:
            tmp = line.split()
            # print(tmp)
            fid = int(tmp[0])
            x = (float(tmp[6]) / 717.5)
            y = (float(tmp[7]) / 717.5) * 1.18/1.1
            # print(fid,x,y)
            id = int(float(tmp[5]))%100
            t = tmp[8]
            # print(tmp[8])
            if posgroup[id][fid] ==None:
                posgroup[id][fid] = [[x, y, i, t]]

            else:
                posgroup[id][fid].append([x, y, i, t])
    # print('posgroup', posgroup)
    # posgroup = avgpos(posgroup)
    
    # for i in range(len(posgroup)):
    #     for j in range(len(posgroup[i])):
    #         if(posgroup[i][j]):
    #             print(posgroup[i][j])
    
    # 取邻近
    posgroup = nearbypos(posgroup)

    # for i in len(posgroup):
    #     if posgroup[i]

    for i in range(len(posgroup)):
        for j in range(1,len(posgroup[i]),N):
            if posgroup[i][j]:
                if 1200-j>=N:n=N
                else:n=1200-j
                for k in range(n,-1,-1):
                    if posgroup[i][j+k]:
                        break
                disx=posgroup[i][j+k][0]-posgroup[i][j][0]
                disy=posgroup[i][j+k][1]-posgroup[i][j][1]
                for m in range(1,k):
                    # print('posgroup[i][j+m]!',posgroup[i][j])
                    posgroup[i][j+m]=[posgroup[i][j][0]+disx/k*m,posgroup[i][j][1]+disy/k*m, posgroup[i][j][2]]
                    # print('posgroup[i][j+m]',posgroup[i][j+m])


    with open(file + 'position' + suffix, 'w') as w:
        for i in range(len(posgroup)):
            for j in range(len(posgroup[i])):
                if posgroup[i][j]:
                    pos = posgroup[i][j]
                    # print('pos',pos)
                    line = str(j)+' '+ str(pos[0])+' '+str(pos[1])+' '+str(i)+' '+str(pos[2])
                    print('line',line)
                    w.write(line + '\n')




