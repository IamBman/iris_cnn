import os
import numpy as np
from PIL import Image

#计算准确率
#output：推理结果  label：标签对应的数字的张量
def get_acc(output, label):
    total = output.shape[0]
    pred_label = output.argmax(dim=1)
    num_correct = (pred_label == label).float().sum().item()
    return num_correct / total

#从数据目录建立对应的字典，key是文件夹名，value是序号
def getKey(root_dir):
    label_dir=os.listdir(root_dir)
    np.save('label_dir.npy', label_dir) 

#不使用torch进行图像预处理的方法
def my_transform(img_path):
    # 重设大小为 224x224，且改为黑白单通道
    resized_image = Image.open(img_path).resize((32, 32)).convert('L')
    # 添加 batch 维度，期望 4 维输入：NCHW，取值为0到1。
    img_data = np.asarray(resized_image).astype("float32")
    img_data = img_data / 255
    img_data = np.expand_dims(img_data, axis=0)
    img_data = np.expand_dims(img_data, axis=0)
    return img_data