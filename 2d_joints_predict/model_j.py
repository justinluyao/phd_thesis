#!/usr/bin/python2.7

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
import cv2



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

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 5, stride=1, padding=2),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(True)

        )


        # self.conv_dilated5 = nn.Conv1d(256, 256, 3, padding=0, dilation=3)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

        self.stg0_conv0=nn.Conv2d(256, 256, 3,stride=1, padding=1)
        self.stg0_conv1=nn.Conv2d(256, 128, 3,stride=1, padding=1)
        self.stg0_conv2=nn.Conv2d(128, 128, 3,stride=1, padding=1)
        self.stg0_conv3=nn.Conv2d(128, 64, 3,stride=1, padding=1)
        self.stg0_conv4=nn.Conv2d(64, 64, 3,stride=1, padding=1)
        self.stg0_conv5=nn.Conv2d(64, 22, 3,stride=1, padding=1)

        self.stg1_conv0=nn.Conv2d(278, 256, 3,stride=1, padding=1)
        self.stg1_conv1=nn.Conv2d(256, 128, 3,stride=1, padding=1)
        self.stg1_conv2=nn.Conv2d(128, 128, 3,stride=1, padding=1)
        self.stg1_conv3=nn.Conv2d(128, 64, 3,stride=1, padding=1)
        self.stg1_conv4=nn.Conv2d(64, 64, 3,stride=1, padding=1)
        self.stg1_conv5=nn.Conv2d(64, 22, 3,stride=1, padding=1)

        self.stg2_conv0=nn.Conv2d(278, 256, 3,stride=1, padding=1)
        self.stg2_conv1=nn.Conv2d(256, 128, 3,stride=1, padding=1)
        self.stg2_conv2=nn.Conv2d(128, 128, 3,stride=1, padding=1)
        self.stg2_conv3=nn.Conv2d(128, 64, 3,stride=1, padding=1)
        self.stg2_conv4=nn.Conv2d(64, 64, 3,stride=1, padding=1)
        self.stg2_conv5=nn.Conv2d(64, 22, 3,stride=1, padding=1)

        self.stg3_conv0=nn.Conv2d(278, 256, 3,stride=1, padding=1)
        self.stg3_conv1=nn.Conv2d(256, 128, 3,stride=1, padding=1)
        self.stg3_conv2=nn.Conv2d(128, 128, 3,stride=1, padding=1)
        self.stg3_conv3=nn.Conv2d(128, 64, 3,stride=1, padding=1)
        self.stg3_conv4=nn.Conv2d(64, 64, 3,stride=1, padding=1)
        self.stg3_conv5=nn.Conv2d(64, 22, 3,stride=1, padding=1)


        self.dropout = nn.Dropout()

    def forward(self, x):
        # print(np.shape(x),"  input")

        # print(np.shape(x),"  out")
        # out = F.relu(self.conv_1(x))
        # out = F.relu(self.conv_2(out))
        # pool=self.pool(out)
        # out = F.relu(self.conv_3(pool))
        # out = F.relu(self.conv_4(out))
        # pool1=self.pool(out)
        # out = (self.conv_5(pool1))
        # out = (self.conv_6(out))
        out=self.features(x)
        stg0_out = self.stg0_conv0(out)
        stg0_out = F.relu(stg0_out)

        stg0_out=self.stg0_conv1(stg0_out)
        stg0_out = F.relu(stg0_out)

        stg0_out=self.stg0_conv2(stg0_out)
        stg0_out = F.relu(stg0_out)

        stg0_out=self.stg0_conv3(stg0_out)
        stg0_out = F.relu(stg0_out)

        stg0_out=self.stg1_conv4(stg0_out)
        stg0_out = F.relu(stg0_out)

        stg0_out=self.stg0_conv5(stg0_out)
        stg0_out = F.relu(stg0_out)

        con0_out=torch.cat((out,stg0_out),1)

        stg1_out = self.stg1_conv0(con0_out)
        stg1_out = F.relu(stg1_out)

        stg1_out=self.stg1_conv1(stg1_out)
        stg1_out = F.relu(stg1_out)

        stg1_out=self.stg1_conv2(stg1_out)
        stg1_out = F.relu(stg1_out)

        stg1_out=self.stg1_conv3(stg1_out)
        stg1_out = F.relu(stg1_out)

        stg1_out=self.stg1_conv4(stg1_out)
        stg1_out = F.relu(stg1_out)

        stg1_out=self.stg1_conv5(stg1_out)
        stg1_out = F.relu(stg1_out)



        con1_out=torch.cat((out,stg1_out),1)

        stg2_out = self.stg2_conv0(con1_out)
        stg2_out = F.relu(stg2_out)

        stg2_out=self.stg2_conv1(stg2_out)
        stg2_out = F.relu(stg2_out)

        stg2_out=self.stg2_conv2(stg2_out)
        stg2_out = F.relu(stg2_out)

        stg2_out=self.stg2_conv3(stg2_out)
        stg2_out = F.relu(stg2_out)

        stg2_out=self.stg2_conv4(stg2_out)
        stg2_out = F.relu(stg2_out)

        stg2_out=self.stg2_conv5(stg2_out)
        stg2_out = F.relu(stg2_out)




        con2_out=torch.cat((out,stg2_out),1)

        stg3_out = self.stg3_conv0(con2_out)
        stg3_out = F.relu(stg3_out)

        stg3_out=self.stg3_conv1(stg3_out)
        stg3_out = F.relu(stg3_out)

        stg3_out=self.stg3_conv2(stg3_out)
        stg3_out = F.relu(stg3_out)

        stg3_out=self.stg3_conv3(stg3_out)
        stg3_out = F.relu(stg3_out)

        stg3_out=self.stg3_conv4(stg3_out)
        stg3_out = F.relu(stg3_out)

        stg3_out=self.stg3_conv5(stg3_out)
        stg3_out = F.relu(stg3_out)


        # out = self.dropout(out)

        return stg1_out,stg2_out,stg3_out




class Trainer:
    def __init__(self):
        self.model = Hand_Detect_layer()
        self.mse = nn.MSELoss(reduction='sum')

    def adjust_learning_rate(self,optimizer, lr):
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr



    def train(self, save_dir, batch_gen, num_epochs, batch_size, learning_rate, device):
        # self.model.train()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.model.to(device)

        checkpoint = torch.load(save_dir + "/epoch-" + str(40002) + ".pt")
        self.model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        #epoch = checkpoint['epoch']
        # loss = checkpoint['loss']



        self.lr=learning_rate

        # optimizer = optimizer.load_state_dict(self.optimizer)
        # optimizer = self.optimizer
        self.step = 0
        for epoch in range(num_epochs):
            if self.step % 1000==1:
                self.lr = self.lr*0.99
                self.adjust_learning_rate(optimizer, lr)
            epoch_loss = 0
            correct = 0
            total = 0
            while batch_gen.has_next():
                self.step=self.step+1
                batch_input, batch_target = batch_gen.next_batch(batch_size)
                # print(np.shape(batch_joints))


                cv_crop1=batch_input[0]
                predictions=batch_target[0]
                cv_crop1=cv_crop1.permute(1, 2, 0)
                cv_crop1=cv_crop1.cpu().detach().numpy()
                cv_crop1=(cv_crop1+0.5)*255
                hm1=predictions.permute(1, 2, 0)
                hm1=hm1.cpu().detach().numpy()
                #
                hm_demo1=np.sum(hm1[:,:,0:-1],-1)
                hm_demo1=cv2.resize(hm_demo1,(128,128))
                hm_demo1=hm_demo1*255
                hm_demo1=hm_demo1.astype(np.uint8)
                hm_demo1 = np.expand_dims(hm_demo1, axis=-1)
                hm_demo1=np.repeat(hm_demo1, 3,axis=-1)
                cv_crop1=0.5*cv_crop1+0.5*hm_demo1
                cv_crop1=cv_crop1.astype(np.uint8)
                # cv2.imshow("cv_crop1",cv_crop1)
                # cv2.imshow("hm_demo1",hm_demo1)
                # cv2.waitKey(1)



                # print(np.shape(batch_target))

                batch_input, batch_target = batch_input.to(device), batch_target.to(device)
                optimizer.zero_grad()
                hm1,hm2,hm3 = self.model(batch_input)
                # print(np.shape(hm1))
                # print(np.shape(batch_target))

                loss1=self.mse(hm1, batch_target)/batch_size
                loss2=self.mse(hm2, batch_target)/batch_size
                loss3=self.mse(hm3, batch_target)/batch_size
                # loss4=self.mse(joints, batch_joints)

                loss = loss1+loss2+loss3
                print(loss1)
                print(loss2)
                print(loss3)
                print("epoch ",epoch, " step ",self.step)


                #
                # print(loss,' loss')
                # print(loss2,' loss2')
                # print(loss3,' loss3')
                # print(loss4,' loss4')

                # cv2.imshow("hm_demo1",hm_demo1)

                # loss = 0
                # for p in predictions:#for each stage
                    # print(np.shape(p))
                    # print(np.shape(p.transpose(2, 1).contiguous().view(-1, self.num_classes)))

                    # print(np.shape(p.transpose(2, 1).contiguous().view(-1, self.num_classes)))
                    # print(np.shape(batch_target))
                    # loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), batch_target.view(-1))
                    # loss += torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)*mask[:, :, 1:])
                    # print((loss),"loss")





                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                cv_crop1=batch_input[0]
                predictions=hm3[0]
                cv_crop1=cv_crop1.permute(1, 2, 0)
                cv_crop1=cv_crop1.cpu().detach().numpy()
                cv_crop1=(cv_crop1+0.5)*255
                hm2=predictions.permute(1, 2, 0)
                hm2=hm2.cpu().detach().numpy()

                hm_demo2=np.sum(hm2[:,:,0:-1],-1)
                hm_demo2=cv2.resize(hm_demo2,(128,128))
                hm_demo2=hm_demo2*255
                hm_demo2=hm_demo2.astype(np.uint8)
                hm_demo2 = np.expand_dims(hm_demo2, axis=-1)
                hm_demo2=np.repeat(hm_demo2, 3,axis=-1)
                cv_crop1=0.5*cv_crop1+0.5*hm_demo1




                cv_crop1=cv_crop1.astype(np.uint8)
                cv2.imshow("cv_crop1",cv_crop1)
                cv2.imshow("hm_demo1",hm_demo2)
                cv2.waitKey(1)
                # print(optimizer.step())
                if self.step%500==1:
                    torch.save(self.model, save_dir + "/epoch-" + str(self.step + 1) + ".model")
                    torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(self.step + 1) + ".opt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                    }, save_dir + "/epoch-" + str(self.step + 1) + ".pt")

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
            self.model.load_state_dict(torch.load(model_dir + "/epoch-" + str(2002) + ".model"))
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
