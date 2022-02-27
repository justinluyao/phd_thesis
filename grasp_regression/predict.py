import cv2
import sys
import numpy as np
import model_j as model
import torch
import os
from utils import DecodeBox, loc2bbox, nms
model_dir = "models"
results_dir = "results"
def visualize(frame,hm,obj):


    joints_pred = hm[0].permute(1, 2, 0).cpu().detach().numpy()
    hand_seg_pred = (obj[0].permute(1, 2, 0).cpu().detach().numpy()* 200)[:,:,0]
    obj_seg_pred = (obj[0].permute(1, 2, 0).cpu().detach().numpy()* 200)[:,:,1]

    hand_seg_pred = cv2.resize(hand_seg_pred,(128,128))
    obj_seg_pred = cv2.resize(obj_seg_pred,(128,128))
    joints_pred = np.sum(joints_pred[:, :, 0:-1], -1)*200
    joints_pred = cv2.resize(joints_pred, (128, 128))


    data_visualize = np.ones([200,600,3])
    data_visualize[0:128,0:128,:] = frame
    data_visualize[0:128,148:276,0] = hand_seg_pred
    data_visualize[0:128,296:424,0] = obj_seg_pred
    data_visualize[0:128,444:572,0] = joints_pred
    data_visualize[0:128, 148:276, 1] = hand_seg_pred
    data_visualize[0:128, 296:424, 1] = obj_seg_pred
    data_visualize[0:128, 444:572, 1] = joints_pred
    data_visualize[0:128, 148:276, 2] = hand_seg_pred
    data_visualize[0:128, 296:424, 2] = obj_seg_pred
    data_visualize[0:128, 444:572, 2] = joints_pred

    data_visualize = data_visualize.astype(np.uint8)

    cv2.imshow("data_visualize", data_visualize)

    cv2.waitKey(1)
device="cuda"


def generate_anchor_base( base_size=16, ratios=[0.5, 1, 2],
                     anchor_scales=[0.5, 1, 2]):
    offsets = []
    offsets.append([0.5, 0.5, 0.5, 0.5])
    offsets.append([0.5, 1.5, 0.5, 0])
    offsets.append([0.5, 0, 0.5, 1.5])

    offsets.append([1.5, 1.5, 0.5, 0.5])
    offsets.append([1.5, 0.5, 0.5, 0.5])
    offsets.append([1.5, -1.5, 0.5, 0.5])

    if base_size < 8:
        base_size = 8
    if base_size > 16:
        base_size = 16

    anchor_base = np.zeros((len(ratios) * len(anchor_scales) * len(offsets), 4),
                           dtype=np.float32)
    index = 0
    for i in range(len(ratios)):
        for j in range(len(anchor_scales)):
            for k in offsets:
                h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
                w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

                anchor_base[index, 0] = - h * k[0]*0.5
                anchor_base[index, 1] = - w * k[1]*0.5
                anchor_base[index, 2] = h * k[2]*0.5
                anchor_base[index, 3] = w * k[3]*0.5
                index = index + 1

    return anchor_base


with torch.no_grad():
    detect_model = model.Hand_Detect_layer()
    detect_model.eval()
    detect_model.to(device)
    model_dict = detect_model.state_dict()
    pretrained_dict = torch.load(model_dir + "/epoch-" + str(70502) + ".pth")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    detect_model.load_state_dict(model_dict)

    rpn = model.Resnet50RoIHead()
    rpn.eval()
    rpn.to(device)
    model_dict = rpn.state_dict()
    pretrained_dict = torch.load(model_dir + "/class_epoch-" + str(3002) + ".pth")
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    rpn.load_state_dict(model_dict)


    video = cv2.VideoCapture(1)

    if not video.isOpened():
        print ("Could not open video")
        sys.exit()
    frame_count=0
    save_stack = []
    while video.isOpened():
        frame_count=frame_count+1
        # save_frame=np.zeros([32,32,2])
        save_coordinate=np.zeros([1,4])
        ok, frame = video.read()
        if ok:
            #
            # frame = np.fliplr(frame)
            # Frame = np.expand_dims(Frame, axis=-1)
            ori_frame = cv2.imread('G:\my_data\epic_images\pos\DJI_0023_1 001.jpg')
            ori_frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
            ori_frame = cv2.resize(ori_frame,(1920,1080))
            box_data = [1100, 10, 1800, 710]
            frame = ori_frame[box_data[1]:box_data[3],box_data[0]:box_data[2],:]
            Frame = cv2.resize(frame, (128, 128))

            img=Frame/255-0.5
            input_x = torch.tensor(img, dtype=torch.float)
            input_x=input_x.permute(2,0,1)
            input_x=input_x.unsqueeze_(0)
            # print(np.shape(input_x))
            input_x = input_x.to(device)
            hm1, hm2, hm3,obj0,obj1,obj2,obj3 = detect_model(input_x)

            obj_seg_pred = (obj3[0].permute(1, 2, 0).cpu().detach().numpy() * 250)[:, :, 0]
            # obj_seg_pred = obj_seg_pred.astype(np.uint8)
            # cv2.imshow('hm', obj_seg_pred)
            # cv2.waitKey(0)



            obj_seg_pred[obj_seg_pred < 80] = 0
            obj_seg_pred = obj_seg_pred / np.max(obj_seg_pred) * 255

            obj_seg_pred[obj_seg_pred < 100] = 0
            obj_seg_pred[obj_seg_pred >= 100] = 255

            min_x = 32
            min_y = 32
            max_x = 0
            max_y = 0
            for m in range(0, 32):
                for n in range(0, 32):
                    value = obj_seg_pred[n, m]
                    if value == 255:
                        if min_x > m:
                            min_x = m
                        if min_y > n:
                            min_y = n
                        if max_x < m:
                            max_x = m
                        if max_y < n:
                            max_y = n

            print(min_x, min_y, max_x, max_y)
            cv2.rectangle(obj_seg_pred, (min_x, min_y), (max_x, max_y), (255, 0, 255))
            obj_seg_pred = cv2.resize(obj_seg_pred, (128, 128))
            obj_seg_pred = obj_seg_pred.astype(np.uint8)
            cv2.imshow('obj_mask',obj_seg_pred)
            cv2.waitKey(0)





            # cv2.rectangle(image_data,(int(box_data[0]),int(box_data[1])), (int(box_data[2]),int(box_data[3])),(255,255,255))
            # org_image = org_image.astype(np.uint8)


            anchors = generate_anchor_base(base_size=((max_x-min_x)+(max_y-min_y))/3)



            roi = []
            # cv2.rectangle(new_img, (int(box_data[0]), int(box_data[1])), (int(box_data[2]), int(box_data[3])), (255, 0, 0))
            # print(shape)
            # cv2.waitKey(0)

            for anchor in anchors:
                x1 = int((box_data[0] + (anchor[0] + min_x ) / 32 * 700 ))
                x2 = int((box_data[1] + (anchor[1] + min_y ) / 32 * 700 ))
                x3 = int((box_data[0] + (anchor[2] + max_x ) / 32 * 700 ))
                x4 = int((box_data[1] + (anchor[3] + max_y ) / 32 * 700 ))
                # cv2.rectangle(ori_frame, (x1,x2), (x3,x4), (255, 0, 0))

                roi.append(np.array([x1, x2, x3, x4]))


            ori_frame = ori_frame.astype(np.uint8)
            cv2.imshow('ori_frame', ori_frame)

            roi = np.array(roi)
            # print(roi)

            min_x = np.min(roi[:, 0])
            min_y = np.min(roi[:, 1])
            max_x = np.max(roi[:, 2])
            max_y = np.max(roi[:, 3])

            iw, ih = 1920, 1080

            if min_x < 0:
                min_x = 0
            if min_y < 0:
                min_y = 0
            if max_x > iw:
                max_x = iw
            if max_y > ih:
                max_y = ih

            print(min_y,max_y,min_x,max_x)

            image_data = ori_frame.copy()[min_y:max_y, min_x:max_x, :]
            w, h, _ = image_data.shape
            scale = 0
            print(image_data.shape)
            cv2.waitKey(0)


            if h == 0 or w == 0:
                continue
            if w > h:
                n_h = w / h * 256
                image_data = cv2.resize(image_data, (256, int(n_h)))
                scale = 256 / h
            else:
                n_w = h / w * 256
                image_data = cv2.resize(image_data, (int(n_w), 256))
                scale = 256 / w

            roi[:, 0] = (roi[:, 0] - min_x) * scale
            roi[:, 1] = (roi[:, 1] - min_y) * scale
            roi[:, 2] = (roi[:, 2] - min_x) * scale
            roi[:, 3] = (roi[:, 3] - min_y) * scale
            label = np.array([0, 0]).reshape(1, 2)
            box_data = np.array(box_data).reshape(1, 4)

            box_data[:, 0] = (box_data[:, 0] - min_x) * scale
            box_data[:, 1] = (box_data[:, 1] - min_y) * scale
            box_data[:, 2] = (box_data[:, 2] - min_x) * scale
            box_data[:, 3] = (box_data[:, 3] - min_y) * scale

            img = image_data / 255 - 0.5
            input_x = torch.tensor(img, dtype=torch.float)
            input_x = input_x.permute(2, 0, 1)
            input_x = input_x.unsqueeze_(0)
            input_x = input_x.to('cuda')

            sample_roi_index = torch.zeros(len(roi))
            roi_cls_loc, roi_score = rpn(input_x, roi, sample_roi_index)

            mean = torch.Tensor([0, 0, 0, 0]).repeat(1)[None]
            std = torch.Tensor([0.1, 0.1, 0.2, 0.2]).repeat(1)[None]

            mean = mean.cuda()
            std = std.cuda()
            decodebox = DecodeBox(std, mean, 1)


            # print(roi)

            # print(roi_cls_loc.shape, 'roi_cls_loc', roi.shape, 'roi', roi_score.shape)

            outputs = decodebox.forward(roi_cls_loc, roi_score, roi, height=256,width=256, nms_iou=0.2, score_thresh=0.5)


            # outputs = decodebox1.forward(roi_cls_locs, importance, rois, height = height, width = width, nms_iou = self.iou, scor_thresh = self.confidence)

        #     if len(outputs) == 0:p
        #         return old_image
            print(outputs)
            bbox = outputs[:, :4]

            for i in range(0,len(bbox)):
                cv2.rectangle(image_data, (bbox[i,0],bbox[i,1]), (bbox[i,2],bbox[i,3]), (255, 0, 0))
            cv2.imshow('image_data', image_data)
            cv2.waitKey(0)

            # print(bbox)


        #     conf = outputs[:, 4]
        #     label = outputs[:, 5]
        #
            # bbox[:, 0::2] = (bbox[:, 0::2]) / width * old_width
            # bbox[:, 1::2] = (bbox[:, 1::2]) / height * old_height
            # bbox = np.array(bbox, np.int32)
        #
        # image = old_image
        # thickness = (np.shape(old_image)[0] + np.shape(old_image)[1]) // old_width * 2
        # font = ImageFont.truetype(font='model_data/simhei.ttf',
        #                           size=np.floor(1e-2 * np.shape(image)[1] + 0.5).astype('int32'))
        #
        # for i, c in enumerate(label):
        #     predicted_class = self.class_names[int(c)]
        #     score = conf[i]
        #     # print(score)
        #
        #     left, top, right, bottom = bbox[i]
        #     top = top - 5
        #     left = left - 5
        #     bottom = bottom + 5
        #     right = right + 5
        #
        #     top = max(0, np.floor(top + 0.5).astype('int32'))
        #     left = max(0, np.floor(left + 0.5).astype('int32'))
        #     bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
        #     right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))
        #
        #     # 画框框
        #
        #     label = '{} {:.2f} '.format(predicted_class, score)
        #     # label1 = '{} {:.2f} {:.2f} {:.2f}'.format(predicted_class, score[0,1],score[1,1],score[2,1])
        #
        #     draw = ImageDraw.Draw(image)
        #     label_size = draw.textsize(label, font)
        #     label = label.encode('utf-8')
        #     # label1 = label1.encode('utf-8')
        #
        #     if top - label_size[1] >= 0:
        #         text_origin = np.array([left, top - label_size[1]])
        #         text_origin1 = np.array([left, top - label_size[1] + 10])
        #     else:
        #         text_origin = np.array([left, top + 1])
        #         text_origin1 = np.array([left, top + 10])
        #
        #     for i in range(thickness):
        #         draw.rectangle(
        #             [left + i, top + i, right - i, bottom - i],
        #             outline=self.colors[int(c)])
        #     draw.rectangle(
        #         [tuple(text_origin), tuple(text_origin + label_size + 10)],
        #         fill=self.colors[int(c)])
        #     draw.text(text_origin1, str(label, 'UTF-8'), fill=(0, 0, 0), font=font)
        #     # draw.text(text_origin, str(label1,'UTF-8'), fill=(0, 0, 0), font=font)
        #
        #     del draw
        #
        #























            # box_data = box_data[0]
            # cut = cuts[0]
            # shape = shapes[0]
            # new_img = new_imgs[0]
















            # visualize(Frame,hm3,obj2)



        else:
            break

