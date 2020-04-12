import pandas as pd
import cv2
from PIL import Image,ImageDraw
train = pd.read_csv("train_bbox.csv")
print(train.head(10))
res = pd.DataFrame()
ID = []
bbox = []

for i in train['filename'].unique():
    dealres = train[train['filename'] == i]

    list = ""
    for j in range(dealres.shape[0]):
        list = list + str(dealres['minx'].iloc[j]) + " " + str(dealres['miny'].iloc[j]) + " "+ str(dealres['maxx'].iloc[j]) + " "+ str(dealres['maxy'].iloc[j]) \
               + " " + str(dealres['label'].iloc[j])+ ";"

    ID.append("/media/hszc/data/datafountain/underwater/train/train/image/"+str(i))
    bbox.append(list[:-1])

res['ID'] = ID
res['bbox'] = bbox
print(len(res['ID'].unique()))
res.to_csv("train.csv",index=False)