import pandas as pd
import xml.dom.minidom
import os

def doit(x):
    if x=="echinus":
        return 1
    if x=="starfish":
        return 2
    if x=="scallop":
        return 3
    return 4


train = pd.DataFrame()
filename = []
minx = []
maxx = []
miny = []
maxy = []
defect = []

root_dir = "/media/hszc/data/datafountain/underwater/train/train/box/"
for i in os.listdir(root_dir):
    img_file = root_dir + i
    dom = xml.dom.minidom.parse(img_file)
    root = dom.documentElement
    for j in root.getElementsByTagName('xmin'):
        minx.append(j.firstChild.data)
        filename.append(i.split('.')[0]+".jpg")
    for j in root.getElementsByTagName('xmax'):
        maxx.append(j.firstChild.data)
    for j in root.getElementsByTagName('ymin'):
        miny.append(j.firstChild.data)
    for j in root.getElementsByTagName('ymax'):
        maxy.append(j.firstChild.data)
    for j in root.getElementsByTagName('name'):
        defect.append(j.firstChild.data)

train['filename'] = filename
train['minx'] = minx
train['maxx'] = maxx
train['miny'] = miny
train['maxy'] = maxy
train['label'] = defect
train = train[train['label']!="waterweeds"]
train['label'] = train['label'].apply(lambda x:doit(x))

train['flg'] = 1
for i in range(train.shape[0]):
    if float(train['minx'].iloc[i]) - float(train['maxx'].iloc[i])==0:
        train['flg'].iloc[i] = 0
    if float(train['miny'].iloc[i]) - float(train['maxy'].iloc[i])==0:
        train['flg'].iloc[i] = 0
train=train[train['flg']!=0]
train.to_csv("train_bbox.csv",index=False)
print(train['label'].value_counts())