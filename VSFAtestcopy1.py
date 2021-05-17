"""Quality Assessment of In-the-Wild Videos, ACM MM 2019"""
#
# Author: Dingquan Li
# Email: dingquanli AT pku DOT edu DOT cn
# Date: 2019/11/8
#
# tensorboard --logdir=logs --port=6006
# CUDA_VISIBLE_DEVICES=1 python VSFA.py --database=KoNViD-1k --exp_id=0

from argparse import ArgumentParser
import os
import h5py
import torch
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import random
from scipy import stats
#from tensorboardX import SummaryWriter
import datetime
import pandas as pd
import sys
from tqdm import tqdm

class VQADataset(Dataset):
    def __init__(self, features_dir='CNN_features_KoNViD-1k/', index=None, max_len=8000, feat_dim=4096, scale=1):
        super(VQADataset, self).__init__()
        self.folders = index
        self.features_dir = features_dir
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.scale = scale
#         self.features = np.zeros((len(index), max_len, feat_dim))
#         self.length = np.zeros((len(index), 1))
#         self.mos = np.zeros((len(index), 1))
#         for i in range(len(index)):
#             features = np.load(features_dir + index[i] + '_resnet-50_res5c.npy')
#             if features.shape[0] > max_len:
#                 features = features[0:max_len,:]
#             self.length[i] = features.shape[0]
            
            
            
#             self.features[i, :features.shape[0], :] = features
#             self.mos[i] = np.load(features_dir + index[i] + '_score.npy')  #
            
        
#         self.scale = scale  #
#         self.label = self.mos / self.scale  # label normalization
      #  print(self.features.shape,self.length.shape,self.label.shape)
    def __len__(self):
        return len(self.folders)

    
    def get_img(self,path):
        
        data = np.zeros((max_len, self.feat_dim))
        features = np.load(features_dir + path)
        if features.shape[0] > max_len:
            features = features[0:max_len,:]
        length = features.shape[0]
        label = int(path.split("--")[1][0])
        data[:length,:] = features
        name = path.split("_")[0]
        return data,length,label,name
        
        
    def __getitem__(self, idx):
        
        img_data,length,label,name= self.get_img(self.folders[idx])
        
        
        
        
        sample = img_data,length,label,name
        return sample


class ANN(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, n_ANNlayers=1, dropout_p=0.5):
        super(ANN, self).__init__()
        self.n_ANNlayers = n_ANNlayers
        self.fc0 = nn.Linear(input_size, reduced_size)  #
        self.dropout = nn.Dropout(p=dropout_p)  #
        self.fc = nn.Linear(reduced_size, reduced_size)  #

    def forward(self, input):
        input = self.fc0(input)  # linear
        for i in range(self.n_ANNlayers-1):  # nonlinear
            input = self.fc(self.dropout(F.relu(input)))
        return input


def TP(q, tau=12, beta=0.5):
    """subjectively-inspired temporal pooling"""
    q = torch.unsqueeze(torch.t(q), 0)
    qm = -float('inf')*torch.ones((1, 1, tau-1)).to(q.device)
    qp = 10000.0 * torch.ones((1, 1, tau - 1)).to(q.device)  #
    l = -F.max_pool1d(torch.cat((qm, -q), 2), tau, stride=1)
    m = F.avg_pool1d(torch.cat((q * torch.exp(-q), qp * torch.exp(-qp)), 2), tau, stride=1)
    n = F.avg_pool1d(torch.cat((torch.exp(-q), torch.exp(-qp)), 2), tau, stride=1)
    m = m / n
    return beta * m + (1 - beta) * l


class VSFA(nn.Module):
    def __init__(self, input_size=4096, reduced_size=128, hidden_size=32):

        super(VSFA, self).__init__()
        self.hidden_size = hidden_size
        self.ann = ANN(input_size, reduced_size, 1)
        self.rnn = nn.GRU(reduced_size, hidden_size, batch_first=True)
        self.q = nn.Linear(hidden_size, 1)

    def forward(self, input, input_length):
        #print(input.shape,input_length.shape)
        input = self.ann(input)  # dimension reduction
       # print(input.shape,input_length.shape)
        self.rnn.flatten_parameters()
        outputs, _ = self.rnn(input, self._get_initial_state(input.size(0), input.device))
       # print(outputs.shape)
        q = self.q(outputs)  # frame quality
      #  print(q.shape)
        score = torch.zeros_like(input_length, device=q.device)  #
        for i in range(input_length.shape[0]):  #
            qi = q[i, :np.int(input_length[i].cpu().numpy())]
            #print(qi.shape)
            qi = TP(qi)
            score[i] = torch.mean(qi)  # video overall quality
        return score

    def _get_initial_state(self, batch_size, device):
        h0 = torch.zeros(1, batch_size, self.hidden_size, device=device)
        return h0


if __name__ == "__main__":
    parser = ArgumentParser(description='"VSFA: Quality Assessment of In-the-Wild Videos')
    parser.add_argument("--seed", type=int, default=19920517)
    parser.add_argument('--lr', type=float, default=0.00001,
                        help='learning rate (default: 0.00001)')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='number of epochs to train (default: 2000)')

    parser.add_argument('--database', default='my_dataset', type=str,
                        help='database name (default: CVD2014)')
    parser.add_argument('--model', default='VSFA', type=str,
                        help='model name (default: VSFA)')



    args = parser.parse_args()

    args.decay_interval = int(args.epochs/10)
    args.decay_ratio = 0.8

    torch.manual_seed(args.seed)  #
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    torch.utils.backcompat.broadcast_warning.enabled = True


    
    num_for_val = 10
    os.environ['CUDA_VISIBLE_DEVICE']='0,1,2,3,4'
   # videos_dir = "/cfs/cfs-3cab91f9f/liuzhang/video_data/video_clarity_vid"
    features_dir = "/cfs/cfs-3cab91f9f/liuzhang/video_data/CNN_features_mydata3/" 
  #  datainfo = "./data/Result1.csv"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    # 训练数据集合
    videos_pic1 = []
    result = os.listdir(features_dir)        
    
    
    total_videos = len(result)
    print(total_videos)
    
    
    width = height=0
    max_len = 8000

      
    print("训练数据目录：",features_dir)
    
    #result =result[0:-100]
    test_num = len(result)
    print("test: {},".format(test_num))
    test_dataset = VQADataset(features_dir, result)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset)
    
    model = VSFA().to(device)  #


        
    pre_model = "./models/model_351.pth"
    criterion = nn.L1Loss()  # L1 loss
   # optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
   # scheduler = lr_scheduler.StepLR(optimizer, step_size=args.decay_interval, gamma=args.decay_ratio)
   # best_val_criterion = -1  # SROCC min
            
    # Test
    if 1:
        check_point = torch.load(pre_model)
        new_state_dict={}
        for k, v in check_point["model"].items():
            name =k[7:] # remove `module.`，表面从第7个key值字符取到最后一个字符，正好去掉了module.
            new_state_dict[name] = v #新字典的key值对应的value为一一对应的值。
        
        
        model.load_state_dict(new_state_dict)
        print("load model {} success".format(pre_model))
        
        model.eval()
        with torch.no_grad():
            badcase = {}
          
          
            y_pred = np.zeros(len(result))
            y_test = np.zeros(len(result))
            L = 0
            for i, (features, length, label,name) in enumerate(tqdm(test_loader)):
                y_test[i] = 1 * label.item() 
                features = features.to(device).float()
                label = label.to(device).float()
                outputs = model(features, length.float())
                if outputs >1.5:
                    y_pred[i] =2
                elif outputs <0.5:
                    y_pred[i] = 0
                else:
                    y_pred[i] =1
                if y_pred[i] != label:
                    badcase[name[0]] = [float(outputs),int(label)]
                #y_pred[i] = 1 * outputs.item()
                loss = criterion(outputs, label)
                L = L + loss.item()
                
                
          
          
        test_loss = L / (i + 1)
            
        RMSE = np.sqrt(((y_pred-y_test) ** 2).mean())
        
        re = y_pred == y_test
        print(sum(re)/test_num)
        #print(badcase)
        print("wrong_num:",len(badcase))
          
          
#         print("Test results: test loss={:.4f}, SROCC={:.4f}, KROCC={:.4f}, PLCC={:.4f}, RMSE={:.4f}"
#               .format(test_loss, SROCC, KROCC, PLCC, RMSE))
        
