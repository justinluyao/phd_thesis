
import sys
import torch
import numpy as np
import cv2

save_dir = "E:/3d_hand/2d_joints_predict/models"
save_dird3 = "E:/3d_hand/3d_joints_predict/models"

# sys.path.insert(0, 'G:\projects\joint_label/2d_joints_detector')

# model_dir = "G:\projects\joint_label/all_in_one_detector/models"
# sys.path.insert(0, 'G:\projects\joint_label/all_in_one_detector')

import model_j as model
sys.path.insert(0, 'E:/3d_hand/3d_Joints_predict/')
import model_bbx as model_b


class joints_detector():
    def __init__(self):

        self.detect_model = model.Hand_Detect_layer()
        self.detect_model.eval()
        self.detect_model.to('cuda')
        checkpoint = torch.load(save_dir + "/epoch-" + str(40002) + ".pt")
        self.detect_model.load_state_dict(checkpoint['model_state_dict'])

        self.bbx_model = model_b.MLP()
        self.bbx_model.eval()
        self.bbx_model.to('cuda')

        self.bbx_model.load_state_dict(torch.load(save_dird3 + "/epoch-" + str(138502) + ".model"))
        # self.detect_model.load_state_dict(checkpoint['model_state_dict'])


        # joint_coord_set = np.zeros((7, 2))


    def predict_hm(self,crop):
        crop=cv2.resize(crop,(128,128))
        # cv2.imshow('crop', crop)

        # crop=cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)
        crop = crop / 255 - 0.5
        input_x = torch.tensor(crop, dtype=torch.float)
        input_x = input_x.permute(2, 0, 1)
        input_x = input_x.unsqueeze_(0)
        # print(np.shape(input_x))
        input_x = input_x.to('cuda')
        hm1, hm2, hm3 = self.detect_model(input_x)
        predictions = hm3[0]
        # print(predictions.shape)

        hm_demo1 = predictions.permute(1, 2, 0)
        hm_demo1 = hm_demo1.cpu().detach().numpy()

        return hm_demo1

    def predict_joints(self,hm_demo1, edge,x,y):

        # hm_demo1=hm_demo1.astype(np.uint8)
        pos_re = []
        joint_coord_set = np.zeros([21, 2])
        for k in range(0, 21):

            array = (hm_demo1[:, :, k])

            # array = np.reshape(array, (5, 5))
            # array = np.multiply(array, 100)
            array = array * 255
            maxa = np.max(array)
            # print(maxa)

            coorda = np.where(array == maxa)
            coorda = np.squeeze(coorda)
            # print('numpy自带求最大值坐标： ', coorda,coorda.shape)

            fmwidth = 32
            soft_argmax_x = np.zeros((fmwidth, fmwidth))
            soft_argmax_y = np.zeros((fmwidth, fmwidth))
            for i in range(1, fmwidth + 1, 1):
                soft_argmax_x[i - 1, :] = i / fmwidth
            for j in range(1, fmwidth + 1, 1):
                soft_argmax_y[:, j - 1] = j / fmwidth
            array_softmax = np.exp(array - maxa) / np.sum(np.exp(array - maxa))
            xcoord = np.sum(np.multiply(array_softmax, soft_argmax_x))
            ycoord = np.sum(np.multiply(array_softmax, soft_argmax_y))
            # print('softargmax求出的最大值坐标：', round(xcoord * 32) - 1, round(ycoord * 32) - 1)
            pos_re.append([4 * (round(xcoord * 32) - 1), 4 * (round(ycoord * 32) - 1)])

            joint_coord = [xcoord, ycoord]
            joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
            # if flag == 0:
            #     self.kalman_filter_array[k].correct(joint_coord)
            #     kalman_pred = self.kalman_filter_array[k].predict()
            # if flag == 1:
            #     self.kalman_filter_array1[k].correct(joint_coord)
            #     kalman_pred = self.kalman_filter_array1[k].predict()
            # joint_coord_set[k, :] = np.array([(kalman_pred[0] * 128), (kalman_pred[1] * 128)]).reshape((2))
            joint_coord_set[k, :] = np.array([(joint_coord[0] * 128), (joint_coord[1] * 128)]).reshape((2))


        # print(pos)
        pos = joint_coord_set

        # cv2.line(new_crop, (int(pos[0][1]), int(pos[0][0])), (int(pos[1][1]), int(pos[1][0])), (0, 255, 0), 3)
        # cv2.line(new_crop, (int(pos[1][1]), int(pos[1][0])), (int(pos[2][1]), int(pos[2][0])), (255, 0, 0), 3)
        # cv2.line(new_crop, (int(pos[3][1]), int(pos[3][0])), (int(pos[4][1]), int(pos[4][0])), (0, 0, 255), 3)
        # cv2.line(new_crop, (int(pos[4][1]), int(pos[4][0])), (int(pos[5][1]), int(pos[5][0])), (0, 255, 0), 3)
        # cv2.line(new_crop, (int(pos[6][1]), int(pos[6][0])), (int(pos[3][1]), int(pos[3][0])), (100, 100, 0), 3)
        # cv2.line(new_crop, (int(pos[0][1]), int(pos[0][0])), (int(pos[3][1]), int(pos[3][0])), (0, 100, 100), 3)
        # cv2.line(new_crop, (int(pos[0][1]), int(pos[0][0])), (int(pos[6][1]), int(pos[6][0])), (100, 100, 100), 3)

        joint_coord_set = joint_coord_set / 128 * edge




        joint_coord_set[:, 0] = joint_coord_set[:, 0] + y
        joint_coord_set[:, 1] = joint_coord_set[:, 1] + x

        # print(x,y)
        # print(joint_coord_set,'000000000000000000')




        return joint_coord_set

    def predict_3d(self,d2_joints):
        center=np.sum(d2_joints,axis=0)/21
        d2_joints=(d2_joints-center)/100
        temp=d2_joints[:,0].copy()
        d2_joints[:,0]=d2_joints[:,1]
        d2_joints[:,1]=temp
        # print(d2_joints.shape)
        d2_joints=np.reshape(d2_joints,(-1,42))
        d2tensor=torch.zeros([1, 42], dtype=torch.float)
        print(d2tensor.shape,d2_joints.shape)
        d2tensor[0] = torch.from_numpy(d2_joints)
        print(d2tensor.shape)
        d2tensor = d2tensor.to('cuda')
        d3_joints=self.bbx_model(d2tensor)
        d3_joints = d3_joints.cpu().detach().numpy()

        d3_joints=d3_joints*0.093
        return d3_joints
