import torch
from torchvision import transforms
import numpy as np
import os
from PIL import Image
import pandas as pd

img_all = "/mnt/localfolder/liuzhang/video_data/video_clarity_vid2/pic"
img_all2 = "/mnt/localfolder/liuzhang/video_data/video_clarity_vid/pic"

imgdirs = os.listdir(img_all)
imgdirs2 = os.listdir(img_all2)

with open("video_2A1C_2C1A_all.txt","r") as f:
    all_data = f.readlines()
    
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path ="./model/model.pt"
model = torch.jit.load(model_path).to(device)



trans = transforms.Compose(
[
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def make_data(all_data,imgdirs,imgdirs2):
    test_data =[]
    
    for data in all_data:
        name = data.split("\t")[0]
        label = data.split("\t")[1][0:-1]     
        if name in imgdirs:
            p = os.path.join(img_all,name)
            test_data.append([p,label])
        elif name in imgdirs2:
            p = os.path.join(img_all2,name)
            test_data.append([p,label])
    return test_data

test_data = make_data(all_data,imgdirs,imgdirs2)

print(len(test_data))
print(test_data[0])







result = {"2A1C":[],"2C1A":[]}

video_score = {}
#print(len(imgdirs))
for p,label in test_data:
 

  #  print(name,label)
 #   break
  #  img_path = os.path.join(img_all,name)
#    print(img_path)
    imgs = os.listdir(p)
    #print(imgs)
    scores = []
    name = p.split("/")[-1]
    for img in imgs:
        path = os.path.join(p,img)

        image = Image.open(path).convert('RGB')

        image = trans(image).to(device)

        image.unsqueeze_(0)
        #print(image.shape)
        preict = model(image)
        preict = torch.nn.functional.softmax(preict, dim=1)
        score = preict.argmax()
        scores.append(score)
    final_score = sum(scores)/len(scores)
    print([final_score,label])
    result[label].append(float(final_score))
    video_score[name] =[float(final_score),label]
    #break
df3 = pd.DataFrame.from_dict(video_score, orient='index',columns=['score',"label"])
df3 = df3.reset_index().rename(columns = {'index':'vid'})    
df3.to_csv('predic_2A1C_2C1A_score_pic12.csv',index=0)    
for label,score in result.items():
    print(label,sum(score)/len(score))
