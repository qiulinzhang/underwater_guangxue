import  pickle
# 重点是rb和r的区别，rb是打开二进制文件，r是打开文本文件
f=open('/home/detao/Videos/mmdetection-master/tools/atss50_1.pkl','rb')
data = pickle.load(f)
for i in data:
    print(i)