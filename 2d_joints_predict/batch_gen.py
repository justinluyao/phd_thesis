#!/usr/bin/python2.7

import torch
import numpy as np
import random
import os
import cv2
from PIL import Image

from torchvision import transforms as transforms
# import PIL.Image as Image


class BatchGenerator(object):
    def __init__(self, img_path,label_path):
 
        self.img_path = img_path
        self.label_path = label_path
        self.index = 0



    def reset(self):
        self.index = 0
        random.shuffle(self.data_list)

    def has_next(self):
        if self.index < len(self.data_list):
            return True
        return False

    def read_data(self, path):#read list of labels
        self.data_list=self.get_list_of_data(path)
        # label_file_list = self.get_list_of_txt(path)
        # print(label_file_list)
        # self.list_of_examples = label_file_list
        random.shuffle(self.data_list)
        # print(self.data_list)

    def get_list_of_txt(self,path):
        label_file_list=[]
        for  files in os.listdir(path):
            if(files.endswith(".txt")):
                label_file_list.append(files)
        return label_file_list

    def get_list_of_data(self,path):
        data_list=[]
        with open(path,"r") as f:
            for line in f:
                data_list.append(line)

        return data_list


    def next_batch(self, batch_size):
        batch = self.data_list[self.index:self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_labels = []
        for vid in batch:
            hm, cv_crop=self.generate_hms_img(vid)
            # img = torch.from_numpy(cv_crop.transpose((2, 0, 1)))
            # hms = torch.from_numpy(hm.transpose((2, 0, 1)))

            batch_input .append(cv_crop)
            batch_target.append(hm)
            # batch_labels.append(label)
            # print((cv_crop.shape),"gg")


        # print(np.shape(batch_input))
        # batch_label_tensor = torch.zeros([len(batch_input)], dtype=torch.long)
        batch_input_tensor = torch.zeros([len(batch_input), np.shape(batch_input)[3],np.shape(batch_input)[1],np.shape(batch_input)[2]], dtype=torch.float)
        batch_target_tensor = torch.zeros([len(batch_target),np.shape(batch_target)[3],np.shape(batch_target)[1],np.shape(batch_target)[2]], dtype=torch.float)
        # print(batch_input_tensor)
        for i in range(len(batch_input)):
            # print(np.shape(batch_input[i].transpose((2, 0, 1))))

            bi=batch_input[i].copy()
            bt=batch_target[i].copy()


            batch_input_tensor[i] = torch.from_numpy(bi.transpose((2, 0, 1)))
            batch_target_tensor[i] = torch.from_numpy(bt.transpose((2, 0, 1)))
            # batch_label_tensor[i] = torch.tensor(batch_labels[i])

            # mask[i, :, :np.shape(batch_target[i])[0]] = torch.ones(self.num_classes, np.shape(batch_target[i])[0])
        # print(np.shape(batch_input_tensor))
        # print(np.shape(batch_target_tensor))

        return batch_input_tensor, batch_target_tensor

    def generate_hms_img(self,vid):
        # label_path=self.label_path + "/" +vid
        # print(vid)
        # print(vid[0:-1])
        # print(vid)
        vid=vid.split(" ")
        # print(vid)

        img_path=vid[0:1]
        # print(img_path)
        # print(len(vid))

        vid=vid[1:-1]
        cv_crop,vid=img_crop(vid,img_path)
        # print(vid)

        # img=cv2.imread(img_path)
        # vid=np.array(( list(map(float, vid))))
        # vid=np.reshape(vid,[21,-1])
        # print(vid)



        # for i in range(0, 21):
        #     cv2.circle(img, (int(vid[i, 0]), int(vid[i, 1])), 2, (255, 0, 0), 1)

        # cv2.imshow('img',img)
        # print(np.shape(vid))
        center=vid

        hm=create_label_hm(center)


        # hm = Image.fromarray(hm)

        # hm = hm.resize((32,32))

        # hm=cv2.resize(hm,(32,32))

        # print(hm.shape)

        # hm1=np.sum(hm[:,:,0:-1],-1)
        # hm1=cv2.resize(hm1,(32,32))
        # cv2.imshow('hm',hm1)
        # cv2.waitKey(0)

        # hm1 = np.expand_dims(hm1, axis=-1)
        #
        # hm=np.concatenate((hm1, hm[:,:,-2:-1]), axis=-1)

        # print(np.shape(hm))

        # demo_hm=[hm[:,:,0]]
        # demo_hm=np.sum(hm[:,:,0:-1],-1)
        # demo_hm = np.expand_dims(demo_hm, axis=-1)
        # demo_hm = np.repeat(demo_hm, 3, axis=-1)
        # demo_hm=cv2.resize(demo_hm,(640,480))
        # #
        # demo_hm=0.5*255*demo_hm+0.5*img
        # demo_hm=demo_hm.astype(np.uint8)
        # cv2.imshow("demo_hm",demo_hm)
        # cv2.waitKey(0)

        # print(np.shape(hm))
        # cv_crop=cv2.imread(img_path)
        # cv_crop = np.expand_dims(cv_crop, axis=-1)
        # cv_crop=Image.open(img_path[0])

        # cv_crop = Image.fromarray(cv_crop)


        # print(np.shape(cv_crop))
        # cv_crop=cv_crop/255-0.5
        # cv_crop = np.asarray(cv_crop)
        # print(np.shape(cv_crop))

        cv_crop = Image.fromarray(cv_crop.astype('uint8')).convert('RGB')

        # cv_crop = transforms.ColorJitter(brightness=0.2)(cv_crop)
        # cv_crop = transforms.ColorJitter(contrast=0.2)(cv_crop)
        # cv_crop = transforms.ColorJitter(saturation=0.2)(cv_crop)
        # cv_crop = transforms.ColorJitter(hue=0.2)(cv_crop)
        cv_crop=np.array(cv_crop)



        # cv_crop=cv2.cvtColor(cv_crop,cv2.COLOR_RGB2BGR)
        # cv2.imshow('cv_crop',cv_crop)
        # cv2.waitKey(0)

        # transform1 = transforms.Compose([transforms.ToTensor()])
        # cv_crop = transform1(cv_crop)
        # cv_crop = np.asarray(cv_crop)
        cv_crop=cv_crop/255-0.5


        return hm, cv_crop

        # hm_demo1 = np.sum(hm[:, :, 0:-1], -1)
        # hm_back_demo1 = hm[:, :, -1]
        #
        # hm_demo1 = cv2.resize(hm_demo1, (128, 128))
        # hm_back_demo1 = cv2.resize(hm_back_demo1, (128, 128))
        # hm_demo1 = hm_demo1 * 256
        # hm_demo1 = hm_demo1.astype(np.uint8)
        #
        # # cv2.imshow("hm_demo1",hm_demo1)
        # # cv2.imshow("hm_back_demo1",hm_back_demo1)
        # hm_demo1 = np.expand_dims(hm_demo1, axis=-1)
        # hm_demo1 = np.repeat(hm_demo1, 3, axis=-1)
        # cv_crop1=cv_crop1*0.5+hm_demo1*0.5
        # cv_crop1 = cv_crop1.astype(np.uint8)
        # cv2.imshow("cv_crop1", cv_crop1)
        # cv2.waitKey(0)

def img_crop(Joints,img_path):
    img_path=img_path[0]
    imp=img_path[0:24]
    imp1=img_path[24:]
    img_path=imp+'/'+imp1
    img=cv2.imread(img_path)
    height,length,c=img.shape
    Joints = list(map(float, Joints))
    Joints=np.reshape(Joints,(21,-1))
    # print(Joints)

    u_x = np.max(Joints[:, 0])
    l_x = np.min(Joints[:, 0])
    u_y = np.max(Joints[:, 1])
    l_y = np.min(Joints[:, 1])

    mx = int((u_x - l_x) / 2)
    my = int((u_y - l_y) / 2)


    # print(np.max((np.max(Joints[:,0])-np.min(Joints[:,0]),abs(np.max(Joints[:,1])-np.min(Joints[:,1])))))
    edge_m = 250

    edge_m = np.max([mx, my])

    # center_x=center_x/count
    # center_y=center_y/count
    center_x = (u_x + l_x) / 2
    center_y = (u_y + l_y) / 2

    rr = random.randint(30, 60)
    rx = random.randint(-20, 20)
    ry = random.randint(-20, 20)

    # rr = 0
    # rx = 0
    # ry = 0

    edge_x = edge_m + rr
    edge_y1 = edge_m + rr
    edge_y2 = edge_m + rr
    center_x = center_x + rx
    center_y = center_y + ry

    a = int(center_x - edge_x)
    b = int(center_y - edge_y1)
    c = int(center_x + edge_x)
    d = int(center_y + edge_y1)

    if (a < 0):
        center_x = edge_x + 1
    if (b < 0):
        center_y = edge_y1 + 1
    if (c > length):
        center_x = length - edge_x - 1
    if (d > height):
        center_y = height - edge_y1 - 1

    a = int(center_x - edge_x)
    b = int(center_y - edge_y1)
    c = int(center_x + edge_x)
    d = int(center_y + edge_y1)
    # print(center_x,center_y,a,b,c,d)

    crop_img = img.copy()[b:d, a:c]
    l, h, _ = crop_img.shape
    # print(b, d, a, c)
    # print(img.shape)

    new_joints = Joints.copy()
    for i in range(0, 21):
        new_joints[i, 0] = (new_joints[i, 0] - a) / l * 128
        new_joints[i, 1] = (new_joints[i, 1] - b) / h * 128

    crop_img = cv2.resize(crop_img, (128, 128))

    pad = random.randint(0, 10)
    if pad <= 3:
        pad_num = random.randint(0, 35)
        pad_up_down = random.randint(0, 1)
        img_mean = np.mean(crop_img)
        if pad_up_down == 0:
            crop_img[(128 - pad_num):, :, :] = int(img_mean)
        if pad_up_down == 1:
            crop_img[:(pad_num), :, :] = int(img_mean)
        pad_num1 = random.randint(0, 35)
        pad_left_right = random.randint(0, 1)

        if pad_left_right == 0:
            crop_img[:, :(pad_num1), :] = int(img_mean)
        if pad_left_right == 1:
            crop_img[:, (128-pad_num1):, :] = int(img_mean)
        # if pad_up_down == 3:
        #     crop_img[(128 - pad_num):, :, :] = int(img_mean)
        #     crop_img[:, :(pad_num), :] = int(img_mean)
        # if pad_up_down == 4:
        #     crop_img[:(pad_num), :, :] = int(img_mean)
        #     crop_img[:, :(pad_num), :] = int(img_mean)


    crop_img = crop_img.astype(np.uint8)
    return crop_img,new_joints
    # for i in range(0, 21):
        # print(new_joints[i,:])
        # crop_img = cv2.circle(crop_img, (int(new_joints[i, 0]), int(new_joints[i, 1])), 4, [100, 100, 100], 3)

    # cv2.imshow("img_demo",img)
    # cv2.imshow("crop_img", crop_img)
    # cv2.waitKey(0)




def create_label_hm(joints):
    # new_j = _relative_joints(joints)
    hm = _generate_hm(32,32, joints)
    hm[:, :, -1] = np.ones((32,32))-np.sum(hm[:,:,0:-1],-1)

    #hm=cv2.resize(hm,(32,24))
    return hm

def _relative_joints(joints):

    """ Convert Absolute joint coordinates to crop box relative joint coordinates
    (Used to compute Heat Maps)
    Args:
        box			: Bounding Box
        padding	: Padding Added to the original Image
        to_size	: Heat Map wanted Size
    """
    # new_j = np.copy(joints)
    # if(offset1>0):
    #     new_j[:,0] = new_j[:,0]*  64/ (offset2)
    #     new_j[:,1] = new_j[:,1]*  64/ (offset1)
    #
    # else:
    #     new_j = new_j * 0

    return joints.astype(np.int8)

def _generate_hm( height, width,joints):

    # print(joints.shape[0],"joints")
    num_joints = joints.shape[0]
    # print(num_joints)
    hm = np.zeros((int(height), int(width), num_joints+1), dtype = np.float32)
    for i in range(num_joints):
        if not(np.array_equal(joints[i], [0,0])):
            s = 3
            hm[:,:,i] = _makeGaussian(height, width, sigma= s, center= (joints[i,0]/4, joints[i,1]/4))
            # hm[:, :, i] = _makeMVGaussian(height, width,joints[i,2],joints[i,3], center=(joints[i, 0], joints[i, 1]))
        else:
            hm[:,:,i] = np.zeros((height,width))

    return hm

def _makeGaussian(height, width, sigma = 3, center=None):
		""" Make a square gaussian kernel.
		size is the length of a side of the square
		sigma is full-width-half-maximum, which
		can be thought of as an effective radius.
		"""
		x = np.arange(0, width, 1, float)
		y = np.arange(0, height, 1, float)[:, np.newaxis]
		if center is None:
			x0 =  width // 2
			y0 = height // 2
		else:
			x0 = center[0]
			y0 = center[1]
		return np.exp(-6*np.log(2) * ((x-x0)**2 + (y-y0)**2) / sigma**2)

def _makeMVGaussian(height, width,tall,wide, center=None):
    X = np.arange(0, width, 1, float)
    Y = np.arange(0, height, 1, float)[:, np.newaxis]
    s2x = 3*wide#wide
    s2y = 3*tall#tall
    sx = np.sqrt(s2x)#variance
    sy = np.sqrt(s2y)
    Cov = 0
    r = Cov / (sx * sy)
    a = 1 / (2 * np.pi * sx * sy * np.sqrt(1 - r ** 2))
    b1 = -1 / (2 * (1 - r ** 2))
    b2 = ((X - center[0])/ sx)**2
    b3 = ((Y - center[1]) / sy)** 2
    b4 = 2 * r* (X - center[0])* (Y - center[1])/ (sx * sy)

    mm=np.max(a * np.exp(b1 * (b2 + b3 - b4)))

    # print(np.max(a * np.exp(b1 * (b2 + b3 - b4))/mm))

    return a * np.exp(b1 * (b2 + b3 - b4))/mm

