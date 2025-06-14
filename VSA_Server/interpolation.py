import os
import numpy as np

def interpolation(path):   #一帧是否会检测到两个人脸，坐标参数，cv画中文
    detect0_txt_path = os.path.join(path)
    if os.path.exists(detect0_txt_path):
        print('face detected!')
        fdetect = open(detect0_txt_path, encoding='utf-8')
        faces = fdetect.readlines()

        finish = {}
        this_frame_num = -1
        for line in faces:
            frame_num, identity, x1, y1, x2, y2 = line.split()
            x2 = int(x1) + int(x2)
            y2 = int(y1) + int(y2)
            if frame_num == this_frame_num:
                bbox_array = [int(x) for x in [x1, y1, x2, y2]]
                finish[int(frame_num)].append([identity,bbox_array])
            else:
                bbox_array = [int(x) for x in [x1, y1, x2, y2]]
                finish[int(frame_num)] = [[identity,bbox_array]]
                this_frame_num = frame_num

        for i in range(len(finish)):
            if i+1 < len(finish):
                key1 = int(list(finish.keys())[i])
                key2 = int(list(finish.keys())[i+1])
                for j in range(key1+1, key2):
                    finish[j] = []
                    for mess1 in finish[key1]:
                        for mess2 in finish[key2]:
                            if mess1[0] == mess2[0]:
                                finish[j].append([mess1[0], mess1[1]])
        for key in list(finish.keys()):
            if not finish.get(key):
                del finish[key]

        with open("interpolation.txt", 'w') as f:
            keys = list(int(key) for key in finish.keys())
            keys.sort()
            for key in keys:
                v = finish[key]
                # if v == []:
                #     print("null!!!")
                if v != []:
                    # print(k,v)
                    for value in v:
                        f.write(
                            str(key) + " " + str(value[1][0]) + " " + str(value[1][1]) + " " + str(value[1][2]) + " " + str(value[1][3]) + " " +
                            value[0] + '\n')
        # temp_num = int(faces[0].split()[0])
        # temp_id = faces[0].split()[1]
        # temp_x1, temp_y1, temp_hei, temp_hig =[int(x) for x in faces[0].split()[2:6]]
        # finish = {}
        # finish[str(temp_num)] = [temp_id, [temp_x1, temp_y1, temp_x1+temp_hei, temp_y1+temp_hig]]
        # for face in faces:
        #     frame_num = int(face.split()[0])
        #     name_id = face.split()[1]
        #     if (frame_num > temp_num) & (name_id == temp_id):
        #         for i in range(temp_num + 1, frame_num + 1):
        #             finish[str(i)] = [temp_id, [temp_x1, temp_y1, temp_x1+temp_hei, temp_y1+temp_hig]]
        #         temp_num = frame_num
        #         temp_id = name_id
        #     elif name_id != temp_id:
        #         temp_num = frame_num
        #         temp_id = name_id
        #         finish[str(temp_num)] = [temp_id,[temp_x1, temp_y1, temp_x1+temp_hei, temp_y1+temp_hig]]

        # with open("interpolation.txt",'w') as f:
        #     for k,v in finish.items():
        #         #print(k,v)
        #         f.write(str(k) +" "+ v[0] +'\n')

        return finish
    else: return 'None'

def interpolation2(path,n):
    n = n+1
    if os.path.exists(path):  ##
        mot_message0 = {}
        f = open(path)  ##
        lines = f.readlines()
        this_frame_num = 0
        for line in lines:
            frame_num, x1, y1, x2, y2, identity = line.split()
            if frame_num == this_frame_num:
                bbox_array = np.array([x1, y1, x2, y2])
                mot_message0[frame_num].append([bbox_array,identity])
            else:
                bbox_array = np.array([x1, y1, x2, y2])
                mot_message0[frame_num] = [[bbox_array,identity]]
                this_frame_num = frame_num

        for i in range(len(mot_message0)):
            if i+1 < len(mot_message0):
                key1 = int(list(mot_message0.keys())[i])
                key2 = int(list(mot_message0.keys())[i+1])
                if (key1+n) == key2:
                    for j in range(1,n):
                        mot_message0[str(key1+j)] = []
                        for mess1 in mot_message0[str(key1)]:
                            for mess2 in mot_message0[str(key2)]:
                                if mess1[1] == mess2[1]:
                                    box = ['0', '0', '0', '0']
                                    for k in range(4):
                                        box[k] = str(int(float(mess1[0][k])*j/n + float(mess2[0][k])*(n-j)/n))
                                    mot_message0[str(key1+j)].append([box, mess1[1]])

    with open("interpolation2.txt", 'w') as f:
        keys = list(int(key) for key in mot_message0.keys())
        keys.sort()
        for key in keys:
            key = str(key)
            v = mot_message0[key]
            # if v == []:
            #     print("null!!!")
            if v != []:
                # print(k,v)
                for value in v:
                    f.write(str(key) + " " + value[0][0] + " " + value[0][1] + " " + value[0][2] + " " + value[0][3] + " " + value[1] + '\n')

    return mot_message0

a=interpolation("./1624328707.txt")
b=interpolation2('./1623902039.txt', 1)

