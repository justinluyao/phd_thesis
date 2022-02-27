#-------------------------------------#
#   调用摄像头或者视频进行检测
#   调用摄像头直接运行即可
#   调用视频可以将cv2.VideoCapture()指定路径
#   视频的保存并不难，可以百度一下看看
#-------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
import os
from frcnn import FRCNN

frcnn = FRCNN()
#-------------------------------------#
#   调用摄像头
# capture=cv2.VideoCapture("G:\my_data/DJI_0091_005.mp4")
#-------------------------------------#
capture=cv2.VideoCapture(0)
capture.set(3, 1280)
capture.set(4, 720)

#
fps = 0.0
data = []

test_set =[]
test_path = 'C:/Users/yl1220/OneDrive - University of Bristol/AAAI/test_images/'
for img in os.listdir(test_path):
    test_set.append(test_path+img)



# while(True):
for img_path in test_set:
    t1 = time.time()
    # 读取某一帧
    # ref,frame=capture.read()
    # if ref is not True:
    #     print(np.array(data).shape)
        # capture.release()
        # break

    frame = cv2.imread(img_path)


    # 格式转变，BGRtoRGB
    # frame = cv2.imread('C:/Users\yl1220\OneDrive - University of Bristol\ismar/test.jpg')
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    # frame = cv2.resize(frame,(960,540))
    # 转变成Image
    frame = Image.fromarray(np.uint8(frame))
    # 进行检测
    frame, re = frcnn.detect_image(frame)
    data.append(re)
    frame = np.array(frame)
    # RGBtoBGR满足opencv显示格式
    frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
    fps  = ( fps + (1./(time.time()-t1)) ) / 2
    print("fps= %.2f"%(fps))
    # frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("video",frame)

    c = cv2.waitKey(0) & 0xff
    if c==27:
        print(np.array(data).shape)

        capture.release()

        break