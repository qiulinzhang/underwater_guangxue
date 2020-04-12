import pandas as pd
import os
import tqdm
import networkx as nx
import mxnet as mx
import torch
import numpy as np
import json
from mmdet.ops.nms.nms_wrapper import nms

def soft_bbox_vote(det, vote_thresh, score_thresh=0.01):
    if det.shape[0] <= 1:
        return det
    order = det[:, 4].ravel().argsort()[::-1]
    det = det[order, :]
    dets = []
    while det.shape[0] > 0:
        # IOU
        area = (det[:, 2] - det[:, 0] + 1) * (det[:, 3] - det[:, 1] + 1)
        xx1 = np.maximum(det[0, 0], det[:, 0])
        yy1 = np.maximum(det[0, 1], det[:, 1])
        xx2 = np.minimum(det[0, 2], det[:, 2])
        yy2 = np.minimum(det[0, 3], det[:, 3])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        o = inter / (area[0] + area[:] - inter)

        # get needed merge det and delete these det
        merge_index = np.where(o >= vote_thresh)[0]
        det_accu = det[merge_index, :]
        det_accu_iou = o[merge_index]
        det = np.delete(det, merge_index, 0)

        if merge_index.shape[0] <= 1:
            try:
                dets = np.row_stack((dets, det_accu))
            except:
                dets = det_accu
            continue
        else:
            soft_det_accu = det_accu.copy()
            soft_det_accu[:, 4] = soft_det_accu[:, 4] * (1 - det_accu_iou)
            soft_index = np.where(soft_det_accu[:, 4] >= score_thresh)[0]
            soft_det_accu = soft_det_accu[soft_index, :]

            det_accu[:, 0:4] = det_accu[:, 0:4] * np.tile(
                det_accu[:, -1:], (1, 4))
            max_score = np.mean(det_accu[:, 4])
            det_accu_sum = np.zeros((1, 5))
            det_accu_sum[:, 0:4] = np.sum(
                det_accu[:, 0:4], axis=0) / np.sum(det_accu[:, -1:])
            det_accu_sum[:, 4] = max_score

            if soft_det_accu.shape[0] > 0:
                det_accu_sum = np.row_stack((det_accu_sum, soft_det_accu))

            try:
                dets = np.row_stack((dets, det_accu_sum))
            except:
                dets = det_accu_sum

    order = dets[:, 4].ravel().argsort()[::-1]
    dets = dets[order, :]
    return dets



root = "submit/"

fileter_thresh =0.01

res1 = pd.read_csv("submit/atss50_1.csv")
# res1=res1[res1["confidence"]>=0.001]
res1['area'] = res1[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res1.shape)
# res1 = res1[res1['area']<=20000*20000]
res1 = res1[res1['area']>=48*48]
res1 = res1[res1['confidence']>fileter_thresh]
print(res1.shape)


res2 = pd.read_csv("submit/atss50_2.csv")
# res2=res2[res2["confidence"]>=0.001]
res2['area'] = res2[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res2.shape)
# res2 = res2[res2['area']<=20000*20000]
# res2 = res2[res2['area']>=48*48]
res2 = res2[res2['confidence']>fileter_thresh]
print(res2.shape)

res3 = pd.read_csv("submit/atss50_3.csv")
# res2=res2[res2["confidence"]>=0.001]
res3['area'] = res3[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res3.shape)
res3 = res3[res3['area']<=500*500]
# res3 = res3[res3['area']>=48*48]
res3 = res3[res3['confidence']>fileter_thresh]
print(res3.shape)

res4 = pd.read_csv("submit/atss101_1.csv")
# res2=res2[res2["confidence"]>=0.001]
res4['area'] = res4[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res4.shape)
# res4 = res4[res4['area']<=20000*20000]
res4 = res4[res4['area']>=48*48]
res4 = res4[res4['confidence']>fileter_thresh]
print(res4.shape)

res5 = pd.read_csv("submit/atss101_2.csv")
# res2=res2[res2["confidence"]>=0.001]
res5['area'] = res5[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res5.shape)
# res5 = res5[res5['area']<=20000*20000]
# res5 = res5[res5['area']>=48*48]
res5 = res5[res5['confidence']>fileter_thresh]
print(res5.shape)

res6 = pd.read_csv("submit/atss101_3.csv")
# res2=res2[res2["confidence"]>=0.001]
res6['area'] = res6[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res6.shape)
res6 = res6[res6['area']<=500*500]
# res6 = res6[res6['area']>=48*48]
res6 = res6[res6['confidence']>fileter_thresh]
print(res6.shape)

res7 = pd.read_csv("submit/atss50_4.csv")
# res2=res2[res2["confidence"]>=0.001]
res7['area'] = res7[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res7.shape)
res7 = res7[res7['area']<=400*400]
# res7 = res7[res7['area']<=48*48]
res7 = res7[res7['confidence']>fileter_thresh]
print(res7.shape)

res8 = pd.read_csv("submit/atss101_4.csv")
# res2=res2[res2["confidence"]>=0.001]
res8['area'] = res8[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res8.shape)
res8 = res8[res8['area']<=400*400]
# res8 = res8[res8['area']>=48*48]
res8 = res8[res8['confidence']>fileter_thresh]
print(res8.shape)

# res9 = pd.read_csv("submit/atss50_5.csv")
# # res2=res2[res2["confidence"]>=0.001]
# res9['area'] = res9[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
# print(res9.shape)
# res9 = res9[res9['area']<=270*270]
# # res9 = res9[res9['area']<=48*48]
# res9 = res9[res9['confidence']>fileter_thresh]
# print(res9.shape)
#
# res10 = pd.read_csv("submit/atss101_5.csv")
# # res2=res2[res2["confidence"]>=0.001]
# res10['area'] = res10[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
# print(res10.shape)
# res10 = res10[res10['area']<=270*270]
# # res10 = res10[res10['area']>=48*48]
# res10 = res10[res10['confidence']>fileter_thresh]
# print(res10.shape)

res9 = pd.read_csv("submit/atss101_6.csv")
# res2=res2[res2["confidence"]>=0.001]
res9['area'] = res9[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res9.shape)
res9 = res9[res9['area']>=56*56]
# res9 = res9[res9['area']<=48*48]
res9 = res9[res9['confidence']>fileter_thresh]
print(res9.shape)

res10 = pd.read_csv("submit/atss101_7.csv")
# res2=res2[res2["confidence"]>=0.001]
res10['area'] = res10[['xmin','xmax','ymin','ymax']].apply(lambda x:(x.xmax-x.xmin+1)*(x.ymax-x.ymin+1),axis=1)
print(res10.shape)
res10 = res10[res10['area']>=48*48]
# res10 = res10[res10['area']>=48*48]
res10 = res10[res10['confidence']>fileter_thresh]
print(res10.shape)


deal_nms = pd.concat([res1,res2,res3,res4,res5,res6,res7,res8,res9,res10])

print(deal_nms.shape)



final = pd.DataFrame()

name = []
image_id =[]
con = []
xmin = []
xmax = []
ymin = []
ymax = []


for filename in tqdm.tqdm(deal_nms['image_id'].unique()):
    for defect_label in ['echinus','starfish','scallop','holothurian']:
        base_dets = deal_nms[deal_nms['image_id']==filename]
        base_dets = base_dets[base_dets['name'] == defect_label]

        dets = np.array(base_dets[['xmin','ymin','xmax','ymax','confidence']])
        # scores = torch.FloatTensor(np.array(base_dets[['confidence']])).to(0)
        iou_thr = 0.62
        keep_boxes = soft_bbox_vote(dets,iou_thr,fileter_thresh)
        for bbox in zip(keep_boxes):
            # print(bbox)
            x1, y1, x2, y2,score = bbox[0][:5]
            # score = press_score.cpu().numpy()
            # x1, y1, x2, y2 = round(float(x1), 2), round(float(y1), 2), round(float(x2), 2), round(float(y2), 2)  # save 0.00
            xmin.append(max(1,1+round(x1)))
            xmax.append(max(1,1+round(x2)))
            ymin.append(round(y1))
            ymax.append(round(y2))
            con.append(score)
            name.append(defect_label)
            image_id.append(filename)

final['xmin'] = xmin
final['xmax'] = xmax
final['ymin'] = ymin
final['ymax'] = ymax
final['name'] = name
final['image_id'] = image_id
final['confidence'] = con
final[['name','image_id','confidence','xmin','ymin','xmax','ymax']].to_csv("submit/testB_10.csv",index=False)




