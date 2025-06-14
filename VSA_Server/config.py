import os

fps = 20
batch_second = 30
batch_frames = fps*batch_second
max_video = 360

img_width = 960
img_height = 540

# root
# root_path = "/home/cliang/vsa_server/camera/"
root_path = '../hikcamera/'

# camera_local
# origin_path = "/mnt/disk2/vsa/VSA_Server/camera/origin/"  # 远程视频解码服务器写入
origin_path = os.path.join(root_path, 'camera/origin/')  # 远程视频解码服务器写入

# origin_path = "/home/mmap/cameraVideos/origin/"  # 远程视频解码服务器写入

# img path
# face_detect_img_path = "/home/yjy/PycharmProjects/VSA_Server/camera/imgs/face_detect/"
# person_detect_img_path = "/home/yjy/PycharmProjects/VSA_Server/camera/imgs/person_detect/"
face_detect_img_path = os.path.join(root_path, 'camera/imgs/face_detect')
person_detect_img_path = os.path.join(root_path, 'camera/imgs/person_detect')

# txt results path
# face_detect_txt_path = "/home/cliang/VSA_Server/camera/txt_results/face_detect/"
# face_mot_txt_path = "/home/cliang/VSA_Server/camera/txt_results/face_mot/"
# person_detect_txt_path = "/home/cliang/VSA_Server/camera/txt_results/person_detect/"
# person_mot_txt_path = "/home/cliang/VSA_Server/camera/txt_results/person_mot/"
# segmentation_txt_path = "/home/cliang/vsa_server/camera/txt_results/segmentation/"
face_detect_txt_path = os.path.join(root_path, 'camera/txt_results/face_detect')
face_mot_txt_path = os.path.join(root_path, 'camera/txt_results/face_mot')
person_detect_txt_path = os.path.join(root_path, 'camera/txt_results/person_detect')
person_mot_txt_path = os.path.join(root_path, 'camera/txt_results/person_mot')
segmentation_txt_path = os.path.join(root_path, 'camera/txt_results/segmentation')

# video results path
# face_detect_video_path = "/home/cliang/vsa_server/camera/video_results/face_detect/"
# face_mot_video_path = "/home/cliang/vsa_server/camera/video_results/face_mot/"
# person_detect_video_path = "/home/cliang/vsa_server/camera/video_results/person_detect/"
# person_mot_video_path = "/home/cliang/vsa_server/camera/video_results/person_mot/"
# segmentation_video_path = "/home/cliang/vsa_server/camera/video_results/segmentation/"
face_detect_video_path = os.path.join(root_path, 'camera/video_results/face_detect')
face_mot_video_path = os.path.join(root_path, 'camera/video_results/face_mot')
person_detect_video_path = os.path.join(root_path, 'camera/video_results/person_detect')
person_mot_video_path = os.path.join(root_path, 'camera/video_results/person_mot')
segmentation_video_path = os.path.join(root_path, 'camera/txt_results/segmentation')

person_detect_switch = True
face_mot_switch = False
person_mot_switch = True
segmentation_switch = False

person_detect_method = 3   # 0:YOLODetector  1:YOLOTinyDetector  2:YOLOBatchDetector   3:YOLOv5
seg_method = 0  #选择分割算法{0：BiseNet , 1：DeepLab-v3+}
seg_root = '/mnt/disk2/vsa/VSA_Server/segmentation/'

detect_skip = 1   #检测跳帧数(≥0的整数)     for more smooth
p = 1               #分割相对于检测的跳帧数(≥0的整数)
seg_skip = (p+1) * detect_skip+p   #分割跳帧数(≥0的整数)
mot_skip = detect_skip

# reid dataset name
dataset = 'cuhk02'

reid_root = '/mnt/disk2/vsa/VSA_Server/retrieval/'
