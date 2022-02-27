#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
import sys
sys.path.append('I:/faster-rcnn/faster-rcnn-pytorch-master/')
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
from frcnn import FRCNN

frcnn = FRCNN()
#-------------------------------------#
#   调用摄像头
vpath = "C:/Users/yl1220/OneDrive - University of Bristol/BMVC/test4.mp4"
save_path = "C:/Users/yl1220/OneDrive - University of Bristol/BMVC/"
capture=cv2.VideoCapture(vpath)
capture.set(3, 1280)
capture.set(4, 720)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('G:\my_data/tea_making\detected/output2.avi',fourcc, 30.0, (1280,720))
#-------------------------------------#
# capture=cv2.VideoCapture(3)
def takeFirst(elem):
    return elem[0]

import numpy as np
import siam_retreiving as siam_retreiving_model
import torch
import cv2
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy import editor
from datetime import timedelta
import os
import os.path
from os import path

hand_side=0
#0left 1right

class Manager(object):
    def __init__(self):
        self.frame_count = 0
        self.path_data = vpath+'.npy'
        self.path = vpath
        self.capture = cv2.VideoCapture(self.path)
        np_load_old = np.load
        np.load = lambda *a, **k: np_load_old(*a, allow_pickle=True, **k)
        self.data = np.load(self.path_data)
        clip = editor.VideoFileClip(self.path)
        rate = clip.fps

        self.final_left_starts=[]
        self.final_right_starts=[]
        self.final_left_ends=[]
        self.final_right_ends=[]

        self.tresh_sum = int(7)
        self.tresh_start_cut = int(4)
        self.tresh_cut = int(4)
        self.tresh_snippt = int(15)
        self.num_similarity_images = int(rate/2)
        self.score =[]

        self.left_score = []
        self.right_score = []

        self.count_l = 0
        self.count_r = 0

    def draw_boxes(self,frame,data,frame_count):
        frame_data = data[frame_count]



        label, bbx = frame_data



        if len(label)==0:
            return frame,0,0,0,[],[],[]

        active_obj_mask = label == 0
        active_right_mask = label == 1
        active_left_mask = label == 2


        bbx_obj = bbx[(active_obj_mask), :]
        bbx_right = bbx[(active_right_mask), :]
        bbx_left = bbx[(active_left_mask), :]

        flag_obj = 0
        flag_left = 0
        flag_right = 0


        if len(bbx_obj) == 0:
            pass
        else:
            flag_obj = 1
            for i in range(bbx_obj.shape[0]):
                cv2.rectangle(frame, (bbx_obj[i, 0], bbx_obj[i, 1]), (bbx_obj[i, 2], bbx_obj[i, 3]), (255, 0, 0), 3)

        if len(bbx_right) == 0:
            pass
        else:
            flag_right = 1
            cv2.rectangle(frame, (bbx_right[0, 0], bbx_right[0, 1]), (bbx_right[0, 2], bbx_right[0, 3]), (0, 255, 0),3)

        if len(bbx_left) == 0:
            pass
        else:
            flag_left = 1
            cv2.rectangle(frame, (bbx_left[0, 0], bbx_left[0, 1]), (bbx_left[0, 2], bbx_left[0, 3]), (255, 0, 255),3)


        return frame, flag_obj, flag_right, flag_left, bbx_obj, bbx_right, bbx_left
    def draw_relation(self,crop_frame,frame, bbx_obj, bbx_right, bbx_left):

        left_relation = 0
        right_relation = 0
        all_relation = 0
        left_obj_crop = []
        right_obj_crop =[]
        all_obj_crop = []

        left_energy = 0
        right_energy = 0

        right_active_mask = (manager.bbox_i(bbx_right, bbx_obj)>1)
        if len(right_active_mask)>0:
            right_active = bbx_obj[right_active_mask[0]]
            if self.count_r > 6:
                cv2.circle(frame,
                           (int((bbx_right[0, 0] + bbx_right[0, 2]) / 2), int((bbx_right[0, 1] + bbx_right[0, 3]) / 2)),50, (255, 255, 255), -1)
            if len(right_active)>0:
                right_relation = 1




                cv2.line(frame, (int((bbx_right[0, 0] + bbx_right[0, 2]) / 2), int((bbx_right[0, 1] + bbx_right[0, 3]) / 2)),
                         (int((right_active[0, 0] + right_active[0, 2]) / 2),
                          int((right_active[0, 1] + right_active[0, 3]) / 2)), (255, 255, 255), 3)
                right_obj_crop = crop_frame[int(right_active[0, 1]):int(right_active[0, 3]),int(right_active[0, 0]):int(right_active[0, 2])]
                a = np.array([(bbx_right[0, 0] + bbx_right[0, 2]) / 2,(bbx_right[0, 1] + bbx_right[0, 3]) / 2])
                b = np.array([(bbx_right[0, 0] + bbx_right[0, 2]) / 2,540])

                right_energy = np.linalg.norm([a-b])





        left_active_mask = (manager.bbox_i(bbx_left, bbx_obj)>1)
        if len(left_active_mask)>0:
            if self.count_l > 6:
                cv2.circle(frame,
                           (int((bbx_left[0, 0] + bbx_left[0, 2]) / 2), int((bbx_left[0, 1] + bbx_left[0, 3]) / 2)), 50,(255, 255, 255), -1)
            left_active = bbx_obj[left_active_mask[0]]
            if len(left_active)>0:
                left_relation = 1


                cv2.line(frame, (int((bbx_left[0, 0] + bbx_left[0, 2]) / 2), int((bbx_left[0, 1] + bbx_left[0, 3]) / 2)),
                         (int((left_active[0, 0] + left_active[0, 2]) / 2),
                          int((left_active[0, 1] + left_active[0, 3]) / 2)), (255, 255, 255), 3)
                left_obj_crop = crop_frame[int(left_active[0, 1]):int(left_active[0, 3]),int(left_active[0, 0]):int(left_active[0, 2])]
                c = np.array([(bbx_left[0, 0] + bbx_left[0, 2]) / 2,(bbx_left[0, 1] + bbx_left[0, 3]) / 2])
                d = np.array([(bbx_left[0, 0] + bbx_left[0, 2]) / 2,540])
                left_energy = np.linalg.norm([c-d])


        return left_relation, right_relation, all_relation, left_obj_crop, right_obj_crop, all_obj_crop, left_energy, right_energy
    def bbox_iou(self,bbox_a, bbox_b):
        if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
            print(bbox_a, bbox_b)
            raise IndexError
        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
        area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
        return area_i / (area_a[:, None] + area_b - area_i)

    def bbox_i(self,bbox_a, bbox_b):
        if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
            print(bbox_a, bbox_b)
            raise IndexError
        tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
        br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
        area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
        return area_i
    def tracker_initialization(self):
        ref, frame = manager.capture.read()
        self.tracker = cv2.TrackerCSRT_create()
        self.bbox = cv2.selectROI(frame, False)
        ok = self.tracker.init(frame, self.bbox)
    def tracker_update(self,frame):
        ok, bbox = manager.tracker.update(frame)
        if ok:
            # Tracking success
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
        else:
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    def get_status(self):
        self.status = []
        with open('data.txt', 'r') as lines:
            for line in lines:
                line = line.split(' ')
                line = list(map(int, line))
                self.status.append(line)
    def segment_filtering(self, hand_num, summation, trigger_thresh, untrigger_thresh):
        status = np.array(self.status)
        energy = 0
        if hand_num == 0:
            status = np.array(self.status)[:, 0]
            energy = np.array(self.status)[:, 2]
        if hand_num == 1:
            status = np.array(self.status)[:, 1]
            energy = np.array(self.status)[:, 3]

        trigger = 0

        self.starts = []
        self.ends = []

        self.score = energy

        for i in range(summation, len(status)):
            value = sum(status[i-summation:i])
            # if trigger ==1:
                # score =score + energy[i]
            print(value,i)

            if value >= trigger_thresh and trigger == 0:
                trigger = 1
                # status[i - summation:i] = 1
                self.starts.append(i)

            if (value < untrigger_thresh and trigger == 1):
                trigger = 0
                # status[i-summation:i]=0
                self.ends.append(i)
                # self.score.append(score)
                score =0
            if ( i==len(status) -1) and trigger==1:
                self.ends.append(i)
                break
                # self.score.append(score)
                # score=0



        print(len(self.starts),self.starts)
        print(len(self.ends),self.ends)

    def get_video_clip(self, start, end,i,hand_side,name):
        if hand_side==0:
            folder = 'data/left/'
        elif hand_side==1:
            folder='data/right/'
        else:
            folder = 'data/fusion/'



        clip = editor.VideoFileClip(self.path)
        rate = clip.fps
        # n_frames = clip.reader.nframes

        start = (start / rate)
        end = ((end) / rate)

        print(start,end)

        if path.isdir(folder+name):
            pass
        else:
            os.makedirs(folder+name)



        ffmpeg_extract_subclip(self.path, start, end, targetname=folder+name+'/clip_'+str(i)+'.mp4')
        # out = folder+name+"/clip_" + str(i) + ".mp4"
        # str = "ffmpeg -i " + self.path + " -ss  " + start + " -to " + end + " -c copy " + out



        # count = 0
        # self.capture.set(cv2.CAP_PROP_POS_FRAMES, start - 1)
        # while count <(end-start):
        #     res, frame = self.capture.read()
        #     frame = cv2.resize(frame, (960, 540))
        #     frame, f1, f2, f3, bbx_obj, bbx_right, bbx_left = manager.draw_boxes(frame, manager.data, start+count)
        #     cv2.imshow('clip_frame',frame)
        #     cv2.waitKey(0)
        #
        #     count = count+1
    def clip_saver(self, hand_side):

        if hand_side==0:
            Path = 'data/left/'

        else:
            Path ='data/right/'

        clips =[]
        for file in os.listdir(Path+'clips/'):
            clips.append(file)




        for relation in range(len(clips)-1):
            count1 = 0
            count2 = 0
            if path.isdir(Path+'relation_comparision/relation_'+str(relation)+'/1/'):
                pass
            else:os.makedirs(Path+'relation_comparision/relation_'+str(relation)+'/1/')
            if path.isdir(Path+'relation_comparision/relation_'+str(relation)+'/2'):
                pass
            else:os.makedirs(Path+'relation_comparision/relation_'+str(relation)+'/2/')


            capture1 = cv2.VideoCapture(Path+'clips/' + clips[relation])


            while True:
                res, frame = capture1.read()

                if count1<manager.ends[relation] - manager.starts[relation] - self.num_similarity_images:
                    count1=count1+1
                    continue

                if res:
                    # frame = cv2.resize(frame, (960, 540))
                    crop_frame = frame.copy()

                    frame, f1, f2, f3, bbx_obj, bbx_right, bbx_left = manager.draw_boxes(frame, manager.data, self.starts[relation]+count1)
                    if f1 != 0 or f2 != 0 or f3 != 0:
                        results = manager.draw_relation(crop_frame,frame, bbx_obj, bbx_right, bbx_left)
                        crop = results[3+hand_side]

                        if len(crop):
                            cv2.imwrite(Path+'relation_comparision/relation_'+str(relation)+'/1/' + 'frame_' + str(count1) + '.jpg', crop)
                    count1 = count1 + 1
                else:
                    break



            capture2 = cv2.VideoCapture(Path+'clips/' + clips[relation+1])

            while True:
                res, frame = capture2.read()

                if count2 > self.num_similarity_images:
                    break

                if res:
                    # frame = cv2.resize(frame, (960, 540))
                    crop_frame = frame.copy()

                    frame, f1, f2, f3, bbx_obj, bbx_right, bbx_left = manager.draw_boxes(frame, manager.data, self.starts[relation+1] + count2)
                    if f1 != 0 or f2 != 0 or f3 != 0:
                        results = manager.draw_relation(crop_frame, frame, bbx_obj, bbx_right, bbx_left)
                        crop = results[3 + hand_side]

                    if len(crop):
                        cv2.imwrite(Path+'relation_comparision/relation_'+str(relation)+'/2/' + 'frame_' + str(count2) + '.jpg', crop)
                    count2 = count2 + 1

                else:
                    break




                # print(file,i)

            # clip1_start = self.starts[clip1]
            # clip1_end = self.ends[clip1]
            #
            # clip2_start = self.starts[clip2]
            # clip2_ends = self.ends[clip2]
            #
            # capture1 = cv2.VideoCapture(self.path)
            # capture1.set(cv2.CAP_PROP_POS_FRAMES, clip1_start - 1)
            #
            # capture2 = cv2.VideoCapture(self.path)
            # capture2.set(cv2.CAP_PROP_POS_FRAMES, clip2_start - 1)
    def clip_similarity_check(self,siam_retreiver,hand_side):
        new_starts =[]
        new_ends = []
        jump=0
        if hand_side==0:
            path = 'data/left/'
        else:
            path = 'data/right/'

        link =[]

        for num,relation in enumerate(os.listdir(path+'relation_comparision')):
            if jump==1:
                jump=0
                continue
            clip1=[]
            clip2=[]
            scores = []
            for clip in os.listdir(path+'relation_comparision/'+relation+'/1/'):
                clip1.append(path+'relation_comparision/'+relation+'/1/'+clip)
            for clip in os.listdir(path+'relation_comparision/'+relation+'/2/'):
                clip2.append(path+'relation_comparision/'+relation+'/2/'+clip)

            sum = 0
            count = 1
            avg_score=0
            for i in range(len(clip1)):
                img1 = cv2.imread(clip1[i])
                for j in range(len(clip2)):
                    img2 = cv2.imread(clip2[j])

                    out = siam_retreiver.similarity_detect(img1,img2)

                    sum =sum+out

                    count=count+1

                    avg_score = sum/count


            print(relation,sum/count,self.ends[num]-self.starts[num],self.ends[num+1]-self.starts[num+1])
            # print()

            if avg_score>0.5 and ((self.starts[num+1+jump]-self.ends[num+jump])<150):

                new_starts.append(self.starts[num+jump])
                new_ends.append(self.ends[num+1+jump])
                jump=1



            else:
                new_starts.append(self.starts[num])
                new_ends.append(self.ends[num])

            # if num>0 and new_ends[-1] == new_ends[-2]:
            #     new_starts.pop(-1)
            #     new_ends.pop(-1)

        print(self.starts)
        print(self.ends)
        print(new_starts)
        print(new_ends)
        linked_starts = new_starts.copy()
        linked_ends = new_ends.copy()

        for i in range(len(new_starts)-1):
            link_error = new_starts[i+1]-new_ends[i]
            if link_error<self.tresh_snippt/2 and ((self.ends[i]-self.starts[i])<self.tresh_snippt or(self.ends[i+1]-self.starts[i+1])<self.tresh_snippt):
                linked_starts[i+1]=0
                linked_ends[i]=0

        final_starts =[]
        final_end=[]

        for i in range(len(linked_starts)):
            if linked_starts[i] != 0:
                final_starts.append(linked_starts[i])

            if linked_ends[i] != 0:
                final_end.append(linked_ends[i])
        print(final_starts)
        print(final_end)

        new_final_starts =[]
        new_final_ends =[]

        for i in range(len(final_starts)):
            if (final_end[i]-final_starts[i])>self.tresh_snippt:
                new_final_starts.append(final_starts[i])
                new_final_ends.append(final_end[i])

        if hand_side==0:
            seg = 'left.txt'
        else:seg = 'right.txt'
        print(new_final_starts)
        print(new_final_ends)

        with open('data/seg_'+seg,'w')as seg_writer:

            for i in range(len(new_final_starts)):

                manager.get_video_clip(new_final_starts[i],new_final_ends[i],i, hand_side,'new_clips')

                # manager.get_video_clip(new_final_starts[i],new_final_ends[i],i,'new_clips')
                count =1
                total_energy = 0
                for iter in range (new_final_starts[i], new_final_ends[i]):
                    energy = self.score[iter]
                    total_energy = total_energy+energy
                    if energy!=0:
                        count = count+1
                total_energy = total_energy/count



                seg_writer.write(str(new_final_starts[i])+' '+str(new_final_ends[i])+' '+str(int(total_energy)))
                seg_writer.write('\n')

                print('segment', i,new_final_starts[i],new_final_ends[i])

        # print(new_starts)
        # print(new_ends)
        # print(linked_starts)
        # print(linked_ends)

    def fusion_cut(self):
        fusion_starts =[]
        fusion_ends = []
        new_fusion_starts =[]
        new_fusion_pairs =[]
        fusion_pairs=[]

        with open('data/seg_left.txt') as seg_left:
            for line in seg_left:
                line=line[0:-1].split(' ')
                self.final_left_starts.append(int(line[0]))
                self.final_left_ends.append(int(line[1]))
                self.left_score.append(int(line[2]))

        with open('data/seg_right.txt') as seg_right:
            for line in seg_right:
                line=line[0:-1].split(' ')
                self.final_right_starts.append(int(line[0]))
                self.final_right_ends.append(int(line[1]))
                self.right_score.append(int(line[2]))

        print(self.final_left_starts)
        print(self.final_left_ends)
        print(self.final_right_starts)
        print(self.final_right_ends)



        fusion_starts = self.final_left_starts + self.final_right_starts
        fusion_ends = self.final_left_ends + self.final_right_ends

        for i in range(len(fusion_starts)):
            fusion_pairs.append((fusion_starts[i],fusion_ends[i]))

        #
        # fusion_starts = list(set(fusion_starts))
        # fusion_ends = list(set(fusion_ends))
        fusion_pairs.sort(key=takeFirst)
        print(fusion_pairs)

        skip=0
        for i in range(len(fusion_pairs)-1):
            if skip ==1:
                skip=0
                continue

            start = np.minimum(fusion_pairs[i][0], fusion_pairs[i + 1][0])
            end = np.maximum(fusion_pairs[i][1], fusion_pairs[i + 1][1])
            intersect = ((fusion_pairs[i][1] - fusion_pairs[i][0]) + (fusion_pairs[i+1][1] - fusion_pairs[i+1][0])) - (end - start)
            if intersect>0:
                # print((fusion_pairs[i][1] - fusion_pairs[i][0]) + (fusion_pairs[i+1][1] - fusion_pairs[i+1][0]),(end - start))
                iou = intersect/(end-start)
                left_overlap = intersect/(fusion_pairs[i][1]-fusion_pairs[i][0])
                right_overlap = intersect/(fusion_pairs[i+1][1]-fusion_pairs[i+1][0])
                print(iou, left_overlap,right_overlap)


                if left_overlap>0.7 and left_overlap>right_overlap:
                    fusion_pairs[i] = (fusion_pairs[i + 1][0], fusion_pairs[i + 1][1])
                    fusion_pairs[i+1] = (fusion_pairs[i + 1][0], fusion_pairs[i + 1][1])

                elif right_overlap>0.7 and right_overlap >= left_overlap:
                    fusion_pairs[i] = (fusion_pairs[i ][0], fusion_pairs[i ][1])
                    fusion_pairs[i+1] = (fusion_pairs[i ][0], fusion_pairs[i ][1])
        for i in range(len(fusion_pairs) - 1):

                # new_fusion_pairs.append((start,end))
                    # skip=1
            #     else:
            #         new_fusion_pairs.append((fusion_pairs[i][0], fusion_pairs[i][1]))
            #         if i==len(fusion_pairs)-2:
            #             new_fusion_pairs.append((fusion_pairs[i+1][0], fusion_pairs[i+1][1]))
            # else:
            #     new_fusion_pairs.append((fusion_pairs[i][0], fusion_pairs[i][1]))
            #     if i == len(fusion_pairs) - 2:
            #         new_fusion_pairs.append((fusion_pairs[i + 1][0], fusion_pairs[i + 1][1]))

            if fusion_pairs[i]!=fusion_pairs[i+1]:
                new_fusion_pairs.append(fusion_pairs[i])
                if i == len(fusion_pairs)-2:
                    new_fusion_pairs.append(fusion_pairs[i+1])

        print(fusion_pairs)
        print(new_fusion_pairs)


        for i in range(len(new_fusion_pairs)):
            print('segment', i)
            self.get_video_clip(new_fusion_pairs[i][0],new_fusion_pairs[i][1],i, 2,'clips')

class Hand_Status(object):
    def __init__(self):
        self.last_hand_crop = []
        self.current_hand_crop = []
        self.current_obj_crop = []
        self.current_hand_status = 0


    def status_update(self,l_s, r_l, a_l, lo_crop, ro_crop, ao_crop):


        if len(self.current_hand_crop):
            self.last_hand_crop = self.current_hand_crop

        if len(lo_crop):
            self.current_hand_crop = lo_crop

class Siam_Retreiver(object):
    def __init__(self):
        self.device = "cuda"
        with torch.no_grad():
            self.detect_model = siam_retreiving_model.SiameseNetwork()
            self.detect_model.eval()
            self.detect_model.to(self.device)
            model_dict = self.detect_model.state_dict()
            pretrained_dict = torch.load('models/' + "siam_model.pth")
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            self.detect_model.load_state_dict(model_dict)

    def similarity_detect(self, img1, img2):
        if len(img1) and len(img2):

            input_pad1 = cv2.resize(img1,(100,100)).astype(np.uint8)
            input_pad2= cv2.resize(img2,(100,100)).astype(np.uint8)

            # cv2.imshow('input_pad1', input_pad1)
            # cv2.imshow('input_pad2', input_pad2)
            # cv2.waitKey(0)
            input_pad1 = cv2.cvtColor(input_pad1, cv2.COLOR_BGR2RGB)
            input_pad2 = cv2.cvtColor(input_pad2, cv2.COLOR_BGR2RGB)


            input = np.concatenate([input_pad1, input_pad2], -1)
            input = input / 255 - 0.5
            input = torch.tensor(input, dtype=torch.float)
            input = input.permute(2, 0, 1)
            input = input.unsqueeze_(0)

            output = self.detect_model(input.cuda())
            output = output.detach().cpu().numpy()[0][0][0][0]

        else:
            print('no_data')

        return output



#
fps = 0.0
data = []
while(True):
    t1 = time.time()
    # 读取某一帧
    ref,frame=capture.read()
    if ref is not True:
        print(np.array(data).shape)
        np.save(vpath,np.array(data))
        capture.release()
        # out.release()
        break

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

    c = cv2.waitKey(1) & 0xff
    if c==27:
        print(np.array(data).shape)

        np.save(save_path, np.array(data))
        capture.release()
        # out.release()

        break

#
#

# segment filtering##


manager = Manager()
# manager.tracker_initialization()
left_hand = Hand_Status()
siam_similarity = Siam_Retreiver()

#
#
with open('data.txt','w') as data_writer:
    while(True):
        # 读取某一帧
        ref,frame=manager.capture.read()

        if ref is not True:
            manager.capture.release()
            break
        # frame = cv2.resize(frame, (1280, 720))

        crop_frame = frame.copy()

        frame, f1, f2, f3, bbx_obj, bbx_right, bbx_left = manager.draw_boxes(frame,manager.data,manager.frame_count)
        l_s, r_s, a_s, le, re = 0, 0, 0, 0, 0
        lo_crop, ro_crop, ao_crop =[],[],[]
        if f1 != 0 or f2 != 0 or f3 != 0:
            l_s, r_s, a_s, lo_crop, ro_crop, ao_crop, le, re = manager.draw_relation(crop_frame,frame, bbx_obj, bbx_right, bbx_left)

            if l_s==1:
                # cv2.putText(frame, 'Left_Interaction_score : 1', (10, 40), 5, 2, (255, 0, 255), 2)
                manager.count_l =manager.count_l+1
            else:
                manager.count_l =manager.count_l-1

            # if l_s==0:
            #     cv2.putText(frame, 'Left_Interaction_score : 0', (10, 40), 5, 2, (255, 0, 255), 2)

            if r_s==1:
                # cv2.putText(frame, 'Right_Interaction_score : 1', (10, 80), 5, 2, (0, 255, 0), 2)
                manager.count_r =manager.count_r+1
            else:
                manager.count_r =manager.count_r-1


            # if r_s==0:
            #     cv2.putText(frame, 'Right_Interaction_score : 0', (10, 80), 5, 2, (0, 255, 0), 2)

        else:
            manager.count_l =manager.count_l-1
            manager.count_r =manager.count_r-1


        # cv2.putText(frame, 'Left Scores : '+ str(manager.count_l), (10, 40), 5, 2, (255, 0, 255), 2)
        # cv2.putText(frame, 'Right Scores : '+ str(manager.count_r), (10, 80), 5, 2, (0, 255, 0), 2)

        if manager.count_l>=9:
            manager.count_l=9
        if manager.count_l<=0:
            manager.count_l=0
        if manager.count_r >= 9:
            manager.count_r = 9
        if manager.count_r <= 0:
            manager.count_r = 0


        left_hand.status_update(l_s, r_s, a_s, lo_crop, ro_crop, ao_crop)

        out.write(frame.astype(np.uint8))

        # siam_similarity.similarity_detect(left_hand.current_hand_crop,left_hand.last_hand_crop)

        data_writer.write(str(l_s))
        data_writer.write(' ')
        data_writer.write(str(r_s))
        data_writer.write(' ')
        data_writer.write(str(int(le)))
        data_writer.write(' ')
        data_writer.write(str(int(re)))
        data_writer.write(' ')
        data_writer.write(str(a_s))
        data_writer.write('\n')

        cv2.imshow('frame',frame)
        cv2.waitKey(1)
        print(manager.frame_count)
        manager.frame_count = manager.frame_count+1
        c = cv2.waitKey(1) & 0xff

        if c == 27:
            manager.capture.release()
            out.release()

for i in range(1,2):
    manager.get_status()
    hand_side=i
    manager.segment_filtering(hand_side,manager.tresh_sum,manager.tresh_start_cut,manager.tresh_cut)
    ##Video segmentation##
    for i in range(len(manager.starts)):
        print('segment', i)
        manager.get_video_clip(manager.starts[i],manager.ends[i],i, hand_side,'clips')
    ##crops_saver##
    manager.clip_saver(hand_side)
    ##similarity_check##
    manager.clip_similarity_check(siam_similarity,hand_side)


manager.fusion_cut()






