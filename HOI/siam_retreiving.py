#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import cv2
import torchvision.models as models



class MultiStageModel(nn.Module):
    def __init__(self, num_stages):
        super(MultiStageModel, self).__init__()
        # self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes)
        self.feature_extract1 = Feature_extraction(3,1)
        self.feature_extract2 = Feature_extraction(3,1)
        self.feature_extract3 = Feature_extraction(3,1)
        self.feature_extract4 = Feature_extraction(1,0)

        self.normal_cnn_layer = Normal_CNN_Layer(3,66,66)
        self.conv_1x1_in = nn.Conv2d(44, 22, 1)

        self.stages = nn.ModuleList([copy.deepcopy(SingleStageModel()) for s in range(num_stages-1)])
        self.conv_out = nn.Conv2d(44, 22, kernel_size=1, padding=0)
        self.num_stages=num_stages

    def forward(self, x):
        # print(mask)
        # print(np.shape(x))

        out = self.feature_extract1(x)
        # out2 = self.feature_extract2(x)
        # out3 = self.feature_extract3(x)
        # out4 = self.feature_extract4(x)


        # concat_layer=torch.cat((out1,out2,out3,out4),1)
        # concat_layer=concat_layer.permute(0, 2, 3, 1)
        # print(np.shape(x),"  out")
        # concat_layer=torch.reshape(concat_layer,[-1,32*32,66])
        # concat_layer = self.conv_1x1_in(concat_layer)


        out1=self.normal_cnn_layer(out)
        concat_layer1=torch.cat((out,out1),1)
        concat_layer1 = self.conv_1x1_in(concat_layer1)
        out2=self.normal_cnn_layer(concat_layer1)
        concat_layer2=torch.cat((out2,out),1)
        concat_layer2 = self.conv_1x1_in(concat_layer2)

        out3=self.normal_cnn_layer(concat_layer2)
        # out=self.conv_out(out)
        # out = torch.nn.functional.adaptive_avg_pool2d(out, (2,21))


        print(np.shape(out),"out")


        # print(np.shape(out),"test")


        # print(np.shape(out))
        # print(np.shape(outputs))

        # for i,s in zip(range(0,self.num_stages), self.stages):
        #     # out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
        #
        #     # print(np.shape(torch.cat((out,features),1)))
        #     # print(np.shape(out), "out")
        #     # print(np.shape(torch.cat((out,features),1)), "features")
        #
        #     # x = self.conv_1_1(torch.cat((out,features),1))
        #     # print(np.shape(x), "x")
        #
        #     out=s(out)
        #
        #     print(np.shape(out))




            # print(np.shape(outputs))
            # print(np.shape(out.unsqueeze(0)))
            # print(np.shape(out),"safsdf")

            # outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)


        return out1,out2,out3

class Feature_extraction(nn.Module):
    def __init__(self,k,p):
        super(Feature_extraction, self).__init__()
        self.conv_1 = nn.Conv2d(3, 64, kernel_size=5, padding=2)
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=k, padding=p)
        self.conv_3 =nn.Conv2d(64, 128, kernel_size=k, padding=p)
        self.conv_4 = nn.Conv2d(128, 128, kernel_size=k, padding=p)
        self.conv_5 = nn.Conv2d(128, 256, kernel_size=k, padding=p)
        self.conv_6 = nn.Conv2d(256, 256, kernel_size=k, padding=p)
        self.conv_7 = nn.Conv2d(256,128, 1)
        self.conv_8 = nn.Conv2d(128,22, 1)

        # self.conv_256_3 = nn.Conv2d(256, 256, kernel_size=7, padding=3)

        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        # self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        # self.conv_out_128 = nn.Conv2d(256,128, 1)
        # self.conv_out_22 = nn.Conv2d(128,22, 1)

    def forward(self, x):
        # print(np.shape(x),"  input")
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        pool1= self.pool(out2)
        out3 = self.conv_3(pool1)
        out4 = self.conv_4(out3)
        pool2= self.pool(out4)
        out5 = self.conv_5(pool2)
        out6 = self.conv_6(out5)
        out7 = self.conv_7(out6)
        out = self.conv_8(out7)


        return out

class SingleStageModel(nn.Module):
    def __init__(self):
        super(SingleStageModel, self).__init__()
        self.conv_1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv_5 = nn.Conv2d(256,128, 1)
        self.conv_out = nn.Conv2d(128,22, 1)


        # self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, 66-i, 66-2*(1+i*2))) for i in range(4)])
        # self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out1 = self.conv_1(x)
        out2 = self.conv_2(out1)
        out3 = self.conv_3(out2)
        out4 = self.conv_4(out3)
        out5 = self.conv_5(out4)
        out = self.conv_out(out5)
        return out

class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(256, 256, 3, padding=0, dilation=1)
        self.conv_dilated1 = nn.Conv1d(256, 256, 3, padding=1, dilation=3)
        self.conv_dilated2= nn.Conv1d(256, 256, 3, padding=1, dilation=5)
        self.conv_dilated3 = nn.Conv1d(256, 256, 3, padding=1, dilation=7)
        self.conv_dilated4 = nn.Conv1d(256, 256, 3, padding=0, dilation=9)
        # self.conv_dilated5 = nn.Conv1d(256, 256, 3, padding=0, dilation=3)


        self.conv_1x1 = nn.Conv1d(256, 256, 1)

        self.dropout = nn.Dropout()

    def forward(self, x):
        # print(np.shape(x),"  input")

        # print(np.shape(x),"  out")
        out = F.relu(self.conv_dilated(x))
        # print(np.shape(out),"  out")
        out = F.relu(self.conv_dilated1(out))
        # print(np.shape(out),"  out")
        out = F.relu(self.conv_dilated2(out))
        # print(np.shape(out),"  out")
        out = F.relu(self.conv_dilated3(out))
        # print(np.shape(out),"  out")
        out = F.relu(self.conv_dilated4(out))
        # print(np.shape(out),"  out")
        # out = self.dropout(out)

        return out

class Normal_CNN_Layer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(Normal_CNN_Layer, self).__init__()
        self.conv_dilated = nn.Conv2d(22, 128, 3, padding=1, dilation=1)
        self.conv_dilated1 = nn.Conv2d(128, 128, 3, padding=1, dilation=1)
        self.conv_dilated2= nn.Conv2d(128, 256, 3, padding=1, dilation=1)
        self.conv_dilated3 = nn.Conv2d(256, 128, 3, padding=1, dilation=1)
        self.conv_dilated4 = nn.Conv2d(128, 22, 3, padding=1, dilation=1)
        # self.conv_dilated5 = nn.Conv1d(256, 256, 3, padding=0, dilation=3)


        self.conv_1x1 = nn.Conv2d(256, 256, 1)

        self.dropout = nn.Dropout()

    def forward(self, x):
        # print(np.shape(x),"  input")

        # print(np.shape(x),"  out")
        out = F.relu(self.conv_dilated(x))
        # print(np.shape(out),"  out")
        out = F.relu(self.conv_dilated1(out))
        # print(np.shape(out),"  out")
        out = F.relu(self.conv_dilated2(out))
        # print(np.shape(out),"  out")
        out = F.relu(self.conv_dilated3(out))
        # print(np.shape(out),"  out")
        out = (self.conv_dilated4(out))

        out = (out)
        # print(np.shape(out),"  out")
        # out = self.dropout(out)

        return out

class Hand_Detect_layer(nn.Module):
    def __init__(self):
        super(Hand_Detect_layer, self).__init__()

        self.conv_1 = nn.Conv2d(3, 64, 5, stride=1,padding=2)
        self.conv_2 = nn.Conv2d(64, 64, 3,stride=1, padding=1)
        self.conv_3= nn.Conv2d(64, 128, 3,stride=1, padding=1)
        self.conv_4 = nn.Conv2d(128, 128, 3,stride=1, padding=1)
        self.conv_5 = nn.Conv2d(128, 256, 3, stride=1,padding=1)
        self.conv_6 = nn.Conv2d(256, 256, 3, stride=1,padding=1)

        # self.conv_dilated5 = nn.Conv1d(256, 256, 3, padding=0, dilation=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stg1_conv0=nn.Conv2d(256, 256, 3,stride=1, padding=1)
        self.stg1_conv1=nn.Conv2d(256, 128, 3,stride=1, padding=1)
        self.stg1_conv2=nn.Conv2d(128, 128, 3,stride=1, padding=1)
        self.stg1_conv3=nn.Conv2d(128, 64, 3,stride=1, padding=1)
        self.stg1_conv4=nn.Conv2d(64, 64, 3,stride=1, padding=1)
        self.stg1_conv5=nn.Conv2d(64, 10, 3,stride=1, padding=1)

        self.stg2_conv0=nn.Conv2d(266, 256, 3,stride=1, padding=1)
        self.stg2_conv1=nn.Conv2d(256, 128, 3,stride=1, padding=1)
        self.stg2_conv2=nn.Conv2d(128, 128, 3,stride=1, padding=1)
        self.stg2_conv3=nn.Conv2d(128, 64, 3,stride=1, padding=1)
        self.stg2_conv4=nn.Conv2d(64, 64, 3,stride=1, padding=1)
        self.stg2_conv5=nn.Conv2d(64, 10, 3,stride=1, padding=1)

        self.stg3_conv0=nn.Conv2d(266, 256, 3,stride=1, padding=1)
        self.stg3_conv1=nn.Conv2d(256, 128, 3,stride=1, padding=1)
        self.stg3_conv2=nn.Conv2d(128, 128, 3,stride=1, padding=1)
        self.stg3_conv3=nn.Conv2d(128, 64, 3,stride=1, padding=1)
        self.stg3_conv4=nn.Conv2d(64, 64, 3,stride=1, padding=1)
        self.stg3_conv5=nn.Conv2d(64, 10, 3,stride=1, padding=1)

        for p in self.parameters():
            p.requires_grad = False
        self.vgg_model = models.vgg16(pretrained=True).features[0:15]
        self.vgg_model = self.vgg_model.train()  # 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
        self.vgg_model.cuda()

        self.sobj_conv0= nn.Conv2d(256, 256, 3,stride=1, padding=1)
        self.sobj_conv1 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.sobj_conv2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.sobj_conv3 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.obj_hm=nn.Conv2d(128, 2, 3,stride=1, padding=1)

        self.sobj_conv4 = nn.Conv2d(258, 256, 3,stride=1, padding=1)
        self.sobj_conv5 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.sobj_conv6 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.sobj_conv7 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.obj_hm1=nn.Conv2d(128, 2, 3,stride=1, padding=1)

        self.sobj_conv8 = nn.Conv2d(258, 256, 3, stride=1, padding=1)
        self.sobj_conv9 = nn.Conv2d(256, 256, 3, stride=1, padding=1)
        self.sobj_conv10 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.sobj_conv11 = nn.Conv2d(128, 128, 3, stride=1, padding=1)
        self.obj_hm2 = nn.Conv2d(128, 2, 3, stride=1, padding=1)

        self.dropout = nn.Dropout()

    def forward(self, x):
        # print(np.shape(x),"  input")

        test = self.vgg_model(x)

        # print(np.shape(x),"  out")
        # out = F.relu(self.conv_1(x))
        # out = F.relu(self.conv_2(out))
        # pool=self.pool(out)
        # out = F.relu(self.conv_3(pool))
        # out = F.relu(self.conv_4(out))
        # pool1=self.pool(out)
        # out = (self.conv_5(pool1))
        # out = (self.conv_6(out))

        # stg1_out = self.stg1_conv0(test)
        # stg1_out = F.relu(stg1_out)
        #
        # stg1_out=self.stg1_conv1(stg1_out)
        # stg1_out = F.relu(stg1_out)
        #
        # stg1_out=self.stg1_conv2(stg1_out)
        # stg1_out = F.relu(stg1_out)
        #
        # stg1_out=self.stg1_conv3(stg1_out)
        # stg1_out = F.relu(stg1_out)
        #
        # stg1_out=self.stg1_conv4(stg1_out)
        # stg1_out = F.relu(stg1_out)
        #
        # stg1_out=self.stg1_conv5(stg1_out)
        # stg1_out = F.relu(stg1_out)
        #
        #
        #
        # con1_out=torch.cat((test,stg1_out),1)
        #
        # stg2_out = self.stg2_conv0(con1_out)
        # stg2_out = F.relu(stg2_out)
        #
        # stg2_out=self.stg2_conv1(stg2_out)
        # stg2_out = F.relu(stg2_out)
        #
        # stg2_out=self.stg2_conv2(stg2_out)
        # stg2_out = F.relu(stg2_out)
        #
        # stg2_out=self.stg2_conv3(stg2_out)
        # stg2_out = F.relu(stg2_out)
        #
        # stg2_out=self.stg2_conv4(stg2_out)
        # stg2_out = F.relu(stg2_out)
        #
        # stg2_out=self.stg2_conv5(stg2_out)
        # stg2_out = F.relu(stg2_out)
        #
        #
        #
        # con2_out=torch.cat((test,stg2_out),1)
        #
        # stg3_out = self.stg3_conv0(con2_out)
        # stg3_out = F.relu(stg3_out)
        #
        # stg3_out=self.stg3_conv1(stg3_out)
        # stg3_out = F.relu(stg3_out)
        #
        # stg3_out=self.stg3_conv2(stg3_out)
        # stg3_out = F.relu(stg3_out)
        #
        # stg3_out=self.stg3_conv3(stg3_out)
        # stg3_out = F.relu(stg3_out)
        #
        # stg3_out=self.stg3_conv4(stg3_out)
        # stg3_out = F.relu(stg3_out)
        #
        # stg3_out=self.stg3_conv5(stg3_out)
        # stg3_out = F.relu(stg3_out)
        #


        # obj_out=torch.cat((test,stg3_out),1)

        obj_out=self.sobj_conv0(test)
        obj_out = F.relu(obj_out)

        obj_out=self.sobj_conv1(obj_out)
        obj_out = F.relu(obj_out)

        obj_out=self.sobj_conv2(obj_out)
        obj_out = F.relu(obj_out)

        obj_out=self.sobj_conv3(obj_out)
        obj_out = F.relu(obj_out)

        obj_out0 = self.obj_hm(obj_out)
        obj_out0 = F.relu(obj_out0)



        obj_out = torch.cat((test, obj_out0), 1)

        obj_out = self.sobj_conv4(obj_out)
        obj_out = F.relu(obj_out)

        obj_out = self.sobj_conv5(obj_out)
        obj_out = F.relu(obj_out)

        obj_out = self.sobj_conv6(obj_out)
        obj_out = F.relu(obj_out)

        obj_out = self.sobj_conv7(obj_out)
        obj_out = F.relu(obj_out)

        obj_out1 = self.obj_hm1(obj_out)
        obj_out1 = F.relu(obj_out1)



        obj_out = torch.cat((test, obj_out1), 1)

        obj_out = self.sobj_conv8(obj_out)
        obj_out = F.relu(obj_out)

        obj_out = self.sobj_conv9(obj_out)
        obj_out = F.relu(obj_out)

        obj_out = self.sobj_conv10(obj_out)
        obj_out = F.relu(obj_out)

        obj_out = self.sobj_conv11(obj_out)
        obj_out = F.relu(obj_out)

        obj_out2 = self.obj_hm2(obj_out)
        obj_out2 = F.relu(obj_out2)

        return obj_out0, obj_out1, obj_out2, obj_out0, obj_out1, obj_out2

class SiameseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        p = 1
        self.cnn1 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=5, padding=2),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=p),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=p),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=p),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=p),

        )

        # self.fc1 = out = F.adaptive_avg_pool2d(x, (1, 1))

    # def forward_once(self, x):
    #     output = self.cnn1(x)
    #     output = output.view(output.size()[0], -1)
    #     output = self.fc1(output)
    #     return output

    def forward(self, input1):
        output = self.cnn1(input1)
        
        # print(output.shape)

        output = F.adaptive_avg_pool2d(output,[1,1])
        output = F.sigmoid(output)

        # output2 = self.forward_once(input2)
        # print(output.shape)
        return output

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim=True)
        print(euclidean_distance,'eu')
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
class Trainer:
    def __init__(self):
        self.model = SiameseNetwork()
        self.mse = nn.MSELoss(reduction='mean')

    def adjust_learning_rate(self,optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        self.criterion = ContrastiveLoss(margin=1.0)
        self.BCE = nn.BCELoss()# 定义损失函数

        # self.model=torch.load(save_dir + "/epoch-" + str(28002) + ".pth")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(save_dir + "/epoch-" + str(502) + ".pth", map_location='cuda')
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        self.lr=learning_rate
        # optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=learning_rate, betas=(0.9, 0.999),
                               eps=1e-08, weight_decay=1e-5)

        self.step = 0
        for epoch in range(num_epochs):
            if self.step % 1000==1:
                self.lr = self.lr*0.99
                self.adjust_learning_rate(optimizer, self.lr)
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                self.step=self.step+1
                batch_org_img_tensor, batch_label_img_tensor, labels_tensor = batch_gen.next_batch(batch_size)
                batch_org_img_tensor, batch_label_img_tensor, labels_tensor = batch_org_img_tensor.to(device), batch_label_img_tensor.to(device),labels_tensor.to(device)

                optimizer.zero_grad()
                output = self.model(batch_org_img_tensor)


                # loss_contrastive = self.criterion(output1, output2, labels_tensor)
                # print(output.view(-1,1),labels_tensor.view(-1,1))

                loss = self.BCE(output.view(-1,1),labels_tensor.view(-1,1))

                original_rgb = ((batch_org_img_tensor[0].permute(1, 2, 0).cpu().detach().numpy() + 0.5) * 250).astype(np.uint8)
                label_img = ((batch_label_img_tensor[0].permute(1, 2, 0).cpu().detach().numpy() + 0.5) * 250).astype(np.uint8)
                cv2.imshow('original_rgb', original_rgb[:,:,0:3])
                cv2.imshow('label_img', label_img)
                print('label',labels_tensor[0], 'pred', output[0])
                # print(loss)

                print("epoch ",epoch, " step ",self.step)
                cv2.waitKey(1)



                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()



                # print(optimizer.step())
                if self.step%500==1:
                    # torch.save(self.model, save_dir + "/epoch-" + str(self.step + 1) + ".pth")
                    torch.save(self.model.state_dict(),  save_dir + "/epoch-" + str(self.step + 1) + ".pth")

                # _, predicted = torch.max(predictions[-1].data, 1)
                # correct += ((predicted == batch_target).float()*mask[:, 0, :].squeeze(1)).sum().item()
                # total += torch.sum(mask[:, 0, :]).item()
            batch_gen.reset()
            # print("[epoch %d]: epoch loss = %f,   acc = %f" % (
            #     epoch + 1, epoch_loss / len(batch_gen.data_list),
            #     float(correct) / total))

    def predict(self, model_dir, results_dir, features_path, vid_list_file, epoch, actions_dict, device, sample_rate):
        self.model.eval()
        with torch.no_grad():
            self.model.to(device)
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(6002) + ".model"))
            file_ptr = open(vid_list_file, 'r')
            list_of_vids = file_ptr.read().split('\n')[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                print (vid)
                features = np.load(features_path + vid.split('.')[0] + '.npy')
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                predictions = self.model(input_x, torch.ones(input_x.size(), device=device))
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    recognition = np.concatenate((recognition, [actions_dict.keys()[actions_dict.values().index(predicted[i].item())]]*sample_rate))
                f_name = vid.split('/')[-1].split('.')[0]
                f_ptr = open(results_dir + "/" + f_name, "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(' '.join(recognition))
                f_ptr.close()

    def visualize(self,batch_input,batch_bbx_target,batch_hm_target,hm,obj):

        original_rgb = ((batch_input[0].permute(1, 2, 0).cpu().detach().numpy()+0.5)*255).astype(np.uint8)
        original_hand_label = (batch_bbx_target[0].permute(1, 2, 0).cpu().detach().numpy()* 255)[:,:,0]
        original_obj_label = (batch_bbx_target[0].permute(1, 2, 0).cpu().detach().numpy()* 255)[:,:,1]
        joint_label = batch_hm_target[0].permute(1, 2, 0).cpu().detach().numpy()

        original_hand_label = cv2.resize(original_hand_label,(128,128))
        original_obj_label = cv2.resize(original_obj_label,(128,128))
        joint_label = np.sum(joint_label[:, :, 0:-1], -1)*200
        joint_label = cv2.resize(joint_label, (128, 128))



        joints_pred = hm[0].permute(1, 2, 0).cpu().detach().numpy()
        hand_seg_pred = (obj[0].permute(1, 2, 0).cpu().detach().numpy()* 255)[:,:,0]
        obj_seg_pred = (obj[0].permute(1, 2, 0).cpu().detach().numpy()* 255)[:,:,1]

        hand_seg_pred = cv2.resize(hand_seg_pred,(128,128))
        obj_seg_pred = cv2.resize(obj_seg_pred,(128,128))
        joints_pred = np.sum(joints_pred[:, :, 0:-1], -1)*200
        joints_pred = cv2.resize(joints_pred, (128, 128))


        data_visualize = np.ones([300,600,3])

        data_visualize[0:128,0:128,:] = original_rgb
        data_visualize[0:128,148:276,0] = original_hand_label
        data_visualize[0:128,296:424,0] = original_obj_label
        data_visualize[0:128,444:572,0] = joint_label
        data_visualize[0:128,148:276,1] = original_hand_label
        data_visualize[0:128,296:424,1] = original_obj_label
        data_visualize[0:128,444:572,1] = joint_label
        data_visualize[0:128,148:276,2] = original_hand_label
        data_visualize[0:128,296:424,2] = original_obj_label
        data_visualize[0:128,444:572,2] = joint_label

        data_visualize[148:276,148:276,0] = hand_seg_pred
        data_visualize[148:276,296:424,0] = obj_seg_pred
        data_visualize[148:276,444:572,0] = joints_pred
        data_visualize[148:276, 148:276, 1] = hand_seg_pred
        data_visualize[148:276, 296:424, 1] = obj_seg_pred
        data_visualize[148:276, 444:572, 1] = joints_pred
        data_visualize[148:276, 148:276, 2] = hand_seg_pred
        data_visualize[148:276, 296:424, 2] = obj_seg_pred
        data_visualize[148:276, 444:572, 2] = joints_pred

        data_visualize = data_visualize.astype(np.uint8)

        cv2.imshow("data_visualize", data_visualize)

        cv2.waitKey(1)

