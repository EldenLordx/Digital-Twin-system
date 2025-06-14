# -*- coding: utf-8 -*-
"""
@File    : remove_old_file.py
@Time    : 2021/2/26 11:31
@Author  : Dontla
@Email   : sxana@qq.com
@Software: PyCharm
"""
import os


def file_remove(filename):
    if not filename.split()[0].isdigit():
        print('this is not our file,it may be a temporary file')
        return
    if os.path.exists(filename):
        os.remove(filename)
        print('remove file: %s' % filename)
    else:
        print('no such file: %s' % filename)


def remove_old_file(FILE_DIR, max_file_saved=20):
    files = os.listdir(FILE_DIR) 
    # print('length of items:', len(ITEMS))
    file_num = len(files)
    if file_num > max_file_saved:
        files.sort()
    for i in range(file_num - max_file_saved): 
        file_location = os.path.join(FILE_DIR,files[i])
        file_remove(file_location)
            
            
            


if __name__ == '__main__':
    save_dir='/mnt/disk2/vsa/kdy/save_dir/'
    camera_num=25
    file_dir=os.path.join(save_dir,str(camera_num))
    remove_old_file(file_dir,10)
    
