import os
import cv2
import numpy as np
import random
path='E:/UTG/v2/original videos/IMAGES/'


def crop_hand(img,joints):
    frame=img.copy()
    x_min=np.min(joints[0,:])
    y_min=np.min(joints[1,:])
    x_max=np.max(joints[0,:])
    y_max=np.max(joints[1,:])

    edge_random=random.randint(50,100)
    x_random=random.randint(-50,50)
    y_random=random.randint(-50,50)


    edge=np.max([int(y_max-y_min),int(x_max-x_min)])/2+edge_random
    center_x=(x_min+x_max)/2+x_random
    center_y=(y_max+y_min)/2+y_random
    img_width=frame.shape[1]
    img_height=frame.shape[0]

    if(center_x-edge<0):
        center_x=edge+1
    if(center_x+edge>img_width):
        center_x=(img_width-edge-1)
    if(center_y+edge<0):
        center_y=edge+1
    if (center_y + edge > img_height):
        center_y = (img_height - edge - 1)
    # cv2.circle(frame, (int(center_x), int(center_y)), 5, (255, 0, 0), 3)

    crop=frame[int(center_y-edge):int(center_y+edge),int(center_x-edge):int(center_x+edge)]

    new_joints=joints.copy()
    for i in range(0,9):
        new_joints[0,i]=int((joints[0,i]-(center_x-edge))/(edge*2)*128)
        new_joints[1,i]=int((joints[1,i]-(center_y-edge))/(2*edge)*128)

    return crop,new_joints


with open('joints_lable.txt','w') as f:
    img_num=100000
    for j in range(0,5):
        for file in os.listdir(path):
            print(file)
            if file.endswith('.jpg'):
                txt_file=file[0:-4]+'.txt'
                with open(path+'/'+txt_file,'r') as txt:
                    joints=[]
                    for line in txt:
                        joints.append(line[0:-1])
                    print(joints)
                    joints = np.array((list(map(float, joints))))
                    joints=np.reshape(joints,(2,-1))
                    useful=1
                    print((joints.shape))
                    img = cv2.imread(path + '/' + file)
                    h,l,c=(img.shape)
                    if joints.shape[1]>0:
                        for i in range(0,9):

                            if(float(joints[0,i])<0) or(float(joints[0,i]>l)) or (float(joints[1,i]<0))or(float(joints[1,i]>h)):
                                useful=0
                                break


                        # cv2.namedWindow('img',cv2.WINDOW_NORMAL)
                        # cv2.resizeWindow('img',1920,1080)

                        if useful == 1:
                            crop,new_joints=crop_hand(img,joints)
                            if(crop.shape[0]==0 or crop.shape[1]==0):
                                continue
                            print(crop.shape)
                            # cv2.imshow('crop', crop)
                            crop=cv2.resize(crop,(128,128))
                            cv2.imwrite('imgs_data/imgs/'+str(img_num)[1:]+'.jpg',crop)
                            f.write(str(img_num)[1:]+' ')
                            for i in range(0,9):
                                x=new_joints[0,i]
                                y=new_joints[1,i]
                                f.write(str(x)+' ')
                                f.write(str(y)+' ')
                            f.write('\n')

                            img_num=img_num+1

                            for i in range(0,9):

                                cv2.circle(crop, (int(new_joints[0,i]),int(new_joints[1,i])), 2,(255, 0, 0), 1)
                            cv2.imshow('crop',crop)

                            cv2.imshow('img',img)
                            cv2.waitKey(0)
                            # print(img.shape)
