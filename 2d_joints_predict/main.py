
#                        .::::.
#                      .::::::::.
#                     :::::::::::
#                  ..:::::::::::'
#               '::::::::::::'
#                 .::::::::::
#            '::::::::::::::..
#                 ..::::::::::::.
#               ``::::::::::::::::
#                ::::``:::::::::'        .:::.
#               ::::'   ':::::'       .::::::::.
#             .::::'      ::::     .:::::::'::::.
#            .:::'       :::::  .:::::::::' ':::::.
#           .::'        :::::.:::::::::'      ':::::.
#          .::'         ::::::::::::::'         ``::::.
#      ...:::           ::::::::::::'              ``::.
#     ```` ':.          ':::::::::'                  ::::..
#                        '.:::::'                    ':'````..

import torch
from model_j import Trainer
from batch_gen import BatchGenerator
import os
import argparse
import random


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device="cuda"
seed = 1538574472
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

# parser = argparse.ArgumentParser()
# parser.add_argument('--action', default='train')
# parser.add_argument('--dataset', default="gtea")
# parser.add_argument('--split', default='1')
#
# args = parser.parse_args()

# num_stages = 4
# num_layers = 3
# num_f_maps = 64
# features_dim = 2048
bz = 32
lr = 0.0005
num_epochs = 100
#
# # use the full temporal resolution @ 15fps
# sample_rate = 1
# # sample input features @ 15fps instead of 30 fps
# # for 50salads, and up-sample the output to 30 fps
# if args.dataset == "50salads":
#     sample_rate = 2

# vid_list_file = "data/"+args.dataset+"/splits/train.split"+args.split+".bundle"
# vid_list_file_tst = "data/"+args.dataset+"/splits/test.split"+args.split+".bundle"
# features_path = "data/"+args.dataset+"/features/"
# gt_path = "data/"+args.dataset+"/groundTruth/"


training_data_path="E:/train/cropped_data"
training_label_path="E:/3d_hand/ho3d-master/ho3d-master/label_2d.txt"

# mapping_file = "data/"+args.dataset+"/mapping.txt"

model_dir = "models"
results_dir = "results"
 
# if not os.path.exists(model_dir):
#     os.makedirs(model_dir)
# if not os.path.exists(results_dir):
#     os.makedirs(results_dir)
#
# file_ptr = open(mapping_file, 'r')
# actions = file_ptr.read().split('\n')[:-1]
#
# file_ptr.close()
# actions_dict = dict()
# for a in actions:
#     actions_dict[a.split()[1]] = int(a.split()[0])
#
# num_classes = len(actions_dict)
# print(num_classes)

trainer = Trainer()
# if args.action == "train":
batch_gen = BatchGenerator(training_data_path,training_label_path)
batch_gen.read_data(training_label_path)
trainer.train(model_dir, batch_gen, num_epochs=num_epochs, batch_size=bz, learning_rate=lr, device=device)

# if args.action == "predict":
#     trainer.predict(model_dir, results_dir, features_path, vid_list_file_tst, num_epochs, actions_dict, device, sample_rate)
