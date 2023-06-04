import os
import numpy as np

#计算准确率
#output：推理结果  label：标签对应的数字的张量
def get_acc(output, label):
    total = output.shape[0]
    pred_label = output.argmax(dim=1)
    num_correct = (pred_label == label).float().sum().item()
    return num_correct / total

#从数据目录建立对应的字典，key是文件夹名，value是序号
def getKey(root_dir):
    key=os.listdir(root_dir)
    label_dir={}
    for val in range(len(key)):
        label_dir[key[val]]=val
        #print(key[val])
        #print(val)
    np.save('label_dir.npy', label_dir) 

#getKey("enrollment_data")
