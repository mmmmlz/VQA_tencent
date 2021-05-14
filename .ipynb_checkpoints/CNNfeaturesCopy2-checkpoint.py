"""Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50"""
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
# 
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=KoNViD-1k --frame_batch_size=64
# CUDA_VISIBLE_DEVICES=1 python CNNfeatures.py --database=CVD2014 --frame_batch_size=32
# CUDA_VISIBLE_DEVICES=0 python CNNfeatures.py --database=LIVE-Qualcomm --frame_batch_size=8

import torch
from torchvision import transforms, models
import torch.nn as nn
from torch.utils.data import Dataset
import skvideo.io
from PIL import Image
import os
import h5py
import numpy as np
import random
from argparse import ArgumentParser
import sys
import pandas as pd
class VideoDataset(Dataset):
    """Read data from the original dataset for feature extraction"""
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
       # self.score = video_names
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_name = self.video_names[idx].split("_")[0]
       # print("ttttttt",video_name,self.video_names[idx])
        
        video_data = skvideo.io.vread(os.path.join(self.videos_dir, video_name+'.mp4'))
        video_score = self.video_names[idx].split("_")[1]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = video_data.shape[0]
        if video_length >38000:
            return
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        transformed_video = torch.zeros([video_length, video_channel,  video_height, video_width])
        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score,
                  "name":video_name+'.mp4',
                 }

        return sample


class ResNet50(torch.nn.Module):
    """Modified ResNet50 for feature extraction"""
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    """2D global standard variation pooling"""
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data,extractor, frame_batch_size=64, device='cuda'):
    """feature extraction"""
   # extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    output2 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
	    while frame_end < video_length:
	        batch = video_data[frame_start:frame_end].to(device)
	        features_mean, features_std = extractor(batch)
	        output1 = torch.cat((output1, features_mean), 0)
	        output2 = torch.cat((output2, features_std), 0)
	        frame_end += frame_batch_size
	        frame_start += frame_batch_size
        
	    last_batch = video_data[frame_start:video_length].to(device)
	    features_mean, features_std = extractor(last_batch)
	    output1 = torch.cat((output1, features_mean), 0)
	    output2 = torch.cat((output2, features_std), 0)
	    output = torch.cat((output1, output2), 1).squeeze()

    return output


if __name__ == "__main__":
    parser = ArgumentParser(description='"Extracting Content-Aware Perceptual Features using Pre-Trained ResNet-50')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--database', default='MY-dataset', type=str,
                        help='database name (default: KoNViD-1k)')
    parser.add_argument('--frame_batch_size', type=int, default=64,
                        help='frame batch size for feature extraction (default: 64)')

    parser.add_argument('--disable_gpu', action='store_true',
                        help='flag whether to disable GPU')
    args = parser.parse_args()

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True

#     if args.database == 'KoNViD-1k':
#         videos_dir = '/home/ldq/Downloads/KoNViD-1k/'  # videos dir
#         features_dir = 'CNN_features_KoNViD-1k/'  # features dir
#         datainfo = 'data/KoNViD-1kinfo.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
#     if args.database == 'CVD2014':
#         videos_dir = '/media/ldq/Research/Data/CVD2014/'
#         features_dir = 'CNN_features_CVD2014/'
#         datainfo = 'data/CVD2014info.mat'
#     if args.database == 'LIVE-Qualcomm':
#         videos_dir = '/media/ldq/Others/Data/12.LIVE-Qualcomm Mobile In-Capture Video Quality Database/'
#         features_dir = 'CNN_features_LIVE-Qualcomm/'
#         datainfo = 'data/LIVE-Qualcomminfo.mat'


    videos_dir = "/cfs/cfs-3cab91f9f/liuzhang/video_data/video_clarity_vid2"
    
    features_dir = "/cfs/cfs-3cab91f9f/liuzhang/video_data/CNN_features_mydata2/"
    
    datainfo = "./data/Result1.csv"
    
    
        
        
    video_list = os.listdir(videos_dir)    
        
        
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)

    device = torch.device("cuda" if not args.disable_gpu and torch.cuda.is_available() else "cpu")
    
    Info = pd.read_csv(datainfo)
    video_format = "RGB"
    
    video_names1 = list(Info["vid"])
    scores = list(Info["score"])
    extractor = ResNet50().to(device)
    video_name_score = []
    if len(video_names1)!= len(scores):
        print("wrong_data")
        sys.exit()
        
    for i in range(len(video_names1)):
        name = video_names1[i] +"_"+str(scores[i])
        video_name_score.append(name)
    print(len(video_name_score))
        
    width = height=0
    video_names =[]
    
    for name in video_name_score:
        name1 = name.split("_")[0]
        if name1+".mp4" in video_list:
            video_names.append(name)
    print(len(video_names))
    print(video_names[-1])
    
   # sys.exit()
 
#     Info = h5py.File(datainfo, 'r')
#     video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
#     scores = Info['scores'][0, :]
#     video_format = Info['video_format'][()].tobytes()[::2].decode()
#     width = int(Info['width'][0])
#     height = int(Info['height'][0])
    dataset = VideoDataset(videos_dir, video_names[::-1], [], video_format, width, height)
    max_len = 0
    print(len(dataset))
    #sys.exit()
    for i in range(len(dataset)):
        current_data = dataset[i]
        if not current_data:
            continue
        current_video = current_data['video']
        current_score = current_data['score']
        current_name = current_data["name"]
        
        print('Video {}: length {} name {} socre {}'.format(i, current_video.shape[0],current_name,current_score))
        max_len=max(max_len,current_video.shape[0])
        features = get_features(current_video,extractor, args.frame_batch_size, device)
        np.save(features_dir + current_name + '_resnet-50_res5c'+"--"+str(current_score), features.to('cpu').numpy())
        #np.save(features_dir + current_name + '_score', current_score)
        
