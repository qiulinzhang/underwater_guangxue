import pandas as pd
import os
import cv2
img_dir = "/media/hszc/data/datafountain/underwater/test-B-image/"

shape = pd.DataFrame()
image_id = []
w=[]
h=[]
for i in os.listdir(img_dir):
    img = cv2.imread(img_dir+i)
    image_id.append(i.replace(".jpg",'.xml'))
    w.append(img.shape[0])
    h.append(img.shape[1])

shape['image_id'] = image_id
shape['w'] = w
shape['h'] = h

res = pd.read_csv("testB_10.csv")
print(res.shape)
res = pd.merge(res,shape,on='image_id',how='left')
res['xmax'] = res[['xmax','h']].apply(lambda x:min(x.xmax,x.h),axis=1)
res['ymax'] = res[['ymax','w']].apply(lambda x:min(x.ymax,x.w),axis=1)
res[['name','image_id','confidence','xmin','ymin','xmax','ymax']].to_csv("transB_10.csv",index=False)



