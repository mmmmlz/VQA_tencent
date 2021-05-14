"""Test Demo for Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2018/3/27
#
import torch
from torchvision import transforms

import skvideo
#skvideo.setFFmpegPath(r'D:\apps\ffmpeg-N-102166-g1ab74bc193-win64-gpl\bin')
import os

import skvideo.io
from PIL import Image
import numpy as np
from VSFA import VSFA
from CNNfeatures import get_features
from argparse import ArgumentParser
import time

with open("video_info.txt","r") as f:
    all_data = f.readlines()
video_dir = "/cfs/cfs-3cab91f9f/liuzhang/video_data/video_clarity_vid"
model_path ="./models/VSFA.pt"numpy
print(skvideo.getFFmpegPath())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#start = time.time()
video_list = os.listdir(video_dir)
model = VSFA()
model.load_state_dict(torch.load(model_path))  #
model.to(device)
model.eval()
A = {}
C = {}

for data in all_data:
    name = data.split(" ")[0][:-3]+".mp4"
    label = data.split(" ")[0][-2:-1]
    if name in video_list:
        video_path = os.path.join(video_dir,name)


        video_data = skvideo.io.vread(video_path)
    
    
        print(video_data.shape)
    
        video_length = video_data.shape[0]
        video_channel = video_data.shape[3]
        video_height = video_data.shape[1]
        video_width = video_data.shape[2]
        
        transformed_video = torch.zeros([video_length, video_channel, video_height, video_width])
        transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

        for frame_idx in range(video_length):
            frame = video_data[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        print('Video length: {}'.format(transformed_video.shape[0]))

        # feature extraction
        features = get_features(transformed_video, frame_batch_size=32, device=device)
        features = torch.unsqueeze(features, 0)  # batch size 1

        # quality prediction using VSFA

        with torch.no_grad():
            input_length = features.shape[1] * torch.ones(1, 1)
            outputs = model(features, input_length)
            y_pred = outputs[0][0].to('cpu').numpy()
            print("Predicted quality: {}".format(y_pred))
            print("labeled qulity:{}".format(label))
            if label =="A":
                A[name] = y_pred
            elif label =="C":
                C[name] = y_pred
                
    #    end = time.time()

   #     print('Time: {} s'.format(end-start))
        
with open("resultA.txt","a") as f:
    total = 0
    for name,socre in A.items():
        f.write(name)
        f.write("    ")
        f.write(str(socre))
        f.write("\n")
        total += float(socre)
    f.write(str(total/len(A)))
with open("resultC.txt","a") as f:
    total = 0
    for name,socre in C.items():
        f.write(name)
        f.write("    ")
        f.write(str(socre))
        f.write("\n")
        total += float(socre)
    f.write(str(total/len(C)))
