import numpy as np
import cv2
import threading
from copy import deepcopy
import sys
sys.path.append('I:/faster-rcnn/faster-rcnn-pytorch-master/')
#-------------------------------------#

from PIL import Image
from frcnn import FRCNN

#-------------------------------------#
#   调用摄像头
vpath = "G:\GTEA\Videos/S1_Hotdog_C1.mp4"
save_path =  "G:\GTEA\Videos/"

import threading
import time
import cv2
import numpy as np
import queue





import numpy as np
import cv2
import threading
from copy import deepcopy

# thread_lock = threading.Lock()
# thread_exit = False

class camThread(object):
    def __init__(self):
        super(camThread, self).__init__()
        self.image_queue = queue.Queue(10)
        self.cap = cv2.VideoCapture(0)
        self.video_thread = threading.Thread(target=self.get_image, args=(self.cap, self.image_queue))
        self.video_thread.start()

    def get_image(self,cap, image_queue):
        while 1:

            _, img = cap.read()

            img= cv2.imread('H:\object_detection_images/test\P09\P09_03/0000004741.jpg')
            image_queue.put(img, True)




class detectThread(object):
    def __init__(self, image_queue):
        super(detectThread, self).__init__()
        self.result_queue = queue.Queue(3)
        self.frcnn = FRCNN()
        self.detect_thread = threading.Thread(target=self.get_result, args=(self.frcnn, image_queue, self.result_queue))
        self.detect_thread.start()
        self.rst_img =[]

    def get_result(self,net, image_queue, result_queue):

        while 1:
            img=np.array((image_queue.get(True)))
            # print(np.array((image_queue.get(True))).shape,'asewqrdsf')


            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img,(960,540))
            # 转变成Image

            img = Image.fromarray(np.uint8(img))

            rst = net.detect_image(img)
            rst_frame,_=rst

            rst_frame = np.array(rst_frame)
            # RGBtoBGR满足opencv显示格式
            self.rst_frame = cv2.cvtColor(rst_frame,cv2.COLOR_RGB2BGR)
            # self.rst_frame = cv2.resize(self.rst_frame,(640,480))

            # result_queue.put(rst[0].copy(), True)
            # while 1:
            #     try:
            #         result_queue.put(rst[0].copy(), True)
            #         img = image_queue.get(False)
            #         rst[0] = img
            #     except:
            #         break
            while 1:
                try:

                    result_queue.put( self.rst_frame.copy(), False)
                    img = image_queue.get(False)
                    # result[0] = img
                except:
                    break
            #



        # return result



def main():

    cam_T = camThread()

    det_T = detectThread(cam_T.image_queue)
    # cam_T.get_frame()

    while 1:

        # thread_lock.acquire()

        # thread_lock.release()
        # thread_lock.acquire()
        # print(len(cam_T.image_queue.get(True)))
        # frame = det_T.get_results(cam_T.image_queue.get(True))
        # thread_lock.release()

        if det_T.result_queue.empty() is not True:
            cv2.imshow('Video', det_T.result_queue.get_nowait())
            cv2.waitKey(1000000)
        else:

            cv2.imshow('Video', cam_T.image_queue.get(True))
            cv2.waitKey(1)
        # if cv2.waitKey(50) & 0xFF == ord('q'):
        #     thread_exit = True

    # cam_T.video_thread.join()
    # det_T.detect_thread.join(yinweiwoyebudongd
if __name__ == "__main__":
    main()
