
import numpy as np
import cv2
import time
import math
import sys
import json
import os
import glob as gb
import numpy as np
from os import listdir
from os.path import isfile, join
import random

i_path="E:/train/collection01/"
path="E:/train/labels/collection01/"

# left_path="D:\ECCV\gtea\keypoint_labelling\data\demo\left/"
# right_path="D:\ECCV\gtea\keypoint_labelling\data\demo/right"
save_img_crops="E:/train/cropped/"
total_count=0
with open ("joints_label1.txt","w") as label:
    for tt in range(0,10):
        for sub_folder in os.listdir(path):
            for file in os.listdir(path+sub_folder):
                # for txt in os.listdir(path+sub_folder+"/"+file):
                    if file.endswith(".txt"):

                        img_path=i_path+'/'+file[0:-4]+".jpg"
                        print(img_path)
                        if os.path.exists(img_path):
                            # print("sssssssssssssssssssssss")


                            img=cv2.imread(img_path)
                            height,length,_=img.shape
                            joints=np.zeros([1,18])

                            with open(path+sub_folder+"/"+file) as txt_file:
                                for i,line in zip(range(0,18),txt_file):
                                    # print(line)
                                    line=float(line)
                                    if i<9 and line < length and line>0:
                                        joints[:, i] = line
                                    elif i>=9 and line < height and line>0:
                                        joints[:, i] = line
                                    else:
                                        joints[:, i]=-10000
                #
                #                 # joints=np.reshape(joints,[-1,2])
                #

                            Joints=np.zeros([9,2])
                            Joints[:,0]=joints[0,0:9]
                            Joints[:,1]=joints[0,9:18]
                            Joints=Joints[0:7,:]


                            count=0
                            center_x=0
                            center_y=0
                            for i in range(0,7):
                                if Joints[i,0]<0 or Joints[i,1]>1080 or Joints[i,0]>1920 or Joints[i,1]<0:
                                    #
                                    # Joints[i, 0]=-1000
                                    # Joints[i, 1]=-1000
                                    pass
                                else:
                                    count=count+1
                                    center_x=center_x+Joints[i,0]
                                    center_y=center_y+Joints[i,1]

                            u_x = np.max(Joints[:,0])
                            l_x = np.min(Joints[:,0])
                            u_y = np.max(Joints[:,1])
                            l_y = np.min(Joints[:,1])

                            mx=int((u_x - l_x)/2)
                            my=int((u_y - l_y)/2)


                            if count<7:
                                continue
                            else:
                                # print(np.max((np.max(Joints[:,0])-np.min(Joints[:,0]),abs(np.max(Joints[:,1])-np.min(Joints[:,1])))))
                                edge_m=250

                                edge_m=np.max([mx,my])
                                if count==0:
                                    edge_m=np.max((np.max(Joints[:,0])-np.min(Joints[:,0]),abs(np.max(Joints[:,1])-np.min(Joints[:,1]))))
                                # center_x=center_x/count
                                # center_y=center_y/count

                                center_x = (u_x+l_x)/2
                                center_y = (u_y+l_y)/2

                                rr=random.randint(50, 150)
                                rx=random.randint(-70, 50)
                                ry=random.randint(-50, 50)




                                # rr = 0
                                # rx = 0
                                # ry = 0

                                edge_x=edge_m+rr
                                edge_y1=edge_m+rr
                                edge_y2=edge_m+rr
                                center_x=center_x+rx
                                center_y=center_y+ry



                                a=int(center_x-edge_x)
                                b=int(center_y-edge_y1)
                                c=int(center_x+edge_x)
                                d=int(center_y+edge_y1)

                                if(a<0):
                                    center_x=edge_x+1
                                if(b<0):
                                    center_y=edge_y1+1
                                if(c>length):
                                    center_x=length-edge_x-1
                                if (d > height):
                                    center_y = height - edge_y1 - 1

                                a=int(center_x-edge_x)
                                b=int(center_y-edge_y1)
                                c=int(center_x+edge_x)
                                d=int(center_y+edge_y1)
                                # print(center_x,center_y,a,b,c,d)

                                crop_img = img.copy()[b:d,a:c]
                                l, h, _ = crop_img.shape
                                print(b,d,a,c)
                                # print(img.shape)

                                new_joints=Joints.copy()
                                for i in range(0, 7):

                                    new_joints[i,0]=(new_joints[i,0]-a)/l*128
                                    new_joints[i,1]=(new_joints[i,1]-b)/h*128

                                    if sub_folder=="right":
                                        new_joints[i, 0]=128-new_joints[i,0]

                                    # if new_joints[i, 0] < 0 or new_joints[i, 1] < 0 or new_joints[i, 0] > length or new_joints[i, 1] > height:
                                    #     new_joints[i, 0] = -1000
                                    #     new_joints[i, 1] = -1000

                                crop_img=cv2.resize(crop_img,(128,128))

                                pad = random.randint(0, 10)
                                if pad <=3:
                                    pad_num=random.randint(0, 35 )
                                    pad_up_down=random.randint(0, 4)
                                    img_mean=np.mean(crop_img)
                                    if pad_up_down==0:
                                        crop_img[(128-pad_num):,:,:]=int(img_mean)
                                    if pad_up_down == 1:
                                        crop_img[:(pad_num), :, :] = int(img_mean)
                                    # if pad_up_down == 2:
                                    #     crop_img[:, (128-pad_num):, :] = int(img_mean)
                                    if pad_up_down == 2:
                                        crop_img[:, :(pad_num), :] = int(img_mean)
                                    if pad_up_down == 3:
                                        crop_img[(128-pad_num):,:,:]=int(img_mean)
                                        crop_img[:, :(pad_num), :] = int(img_mean)
                                    if pad_up_down == 4:
                                        crop_img[:(pad_num), :, :] = int(img_mean)
                                        crop_img[:, :(pad_num), :] = int(img_mean)





                                total_count=total_count+1

                                if sub_folder == "right":
                                    crop_img=np.fliplr(crop_img)

                                label.write(save_img_crops+str(total_count)+".jpg")
                                for i in range(0, 7):
                                    label.write(" ")
                                    label.write(str(new_joints[i, 0]))
                                    label.write(" ")
                                    label.write(str(new_joints[i, 1]))
                                # if folder == "right":
                                #     label.write('R')
                                # if folder == "left":
                                #     label.write('L')
                                label.write("\n")

                                cv2.imwrite(save_img_crops+str(total_count)+".jpg",crop_img)
                                crop_img=crop_img.astype(np.uint8)
                                for i in range(0,7):
                                    # print(new_joints[i,:])
                                    crop_img = cv2.circle(crop_img, (int(new_joints[i, 0]),int(new_joints[i, 1])), 4, [100,100,100], 3)

                                # cv2.imshow("img_demo",img)
                                cv2.imshow("crop_img", crop_img)
                                cv2.waitKey(1)






