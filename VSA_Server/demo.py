import cv2
import os


videoPath = "/home/yjy/PycharmProjects/VSA_Server/camera/origin/27/1616729297.mp4"
vdo = cv2.VideoCapture()
vdo.open(videoPath)
input_arr = []
ret, frame = vdo.read()
img_width = 1280
img_height = 720


path = '/home/yjy/PycharmProjects/VSA_Server/camera/video_results/'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # avc1 or mp4v
if not os.path.exists(path):
    os.makedirs(path)
output_path = os.path.join(path, '1' + ".mp4")
output = cv2.VideoWriter(output_path, fourcc, 20.0, (img_width, img_height), True)

while(vdo.isOpened()):
    ret, frame = vdo.read()
    print(frame.shape)
    if ret==True:
        output.write(frame)
    else:
        break

# while ret:
#     # hbc0608  for batch loader
#     # frame = cv2.resize(frame, (416, 416))
#     cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     input_arr.append(frame)
#     ret, frame = vdo.read()
#     output.write(frame)

output.release()

