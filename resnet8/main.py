import sys
import os
from resnet import ResNet18
import numpy as np
from torchsummary import summary
import torch

sys.path.append("..") 
from my_train import train
from my_test import test
from my_onnx import Convert_ONNX
from my_bench import benchmark


def main():
    train_dir="../enrollment_data"
    test_dir="../test_data"
    label_dir = np.load('../label_dir.npy',allow_pickle=True)
    
    model = ResNet18()
    train(root_dir=train_dir,label_dir=label_dir,model=model,model_name="resnet18")

    acc_list = test(root_dir=test_dir,label_dir=label_dir,model=model,model_name="resnet18")
    ranks = np.argsort(acc_list)[::-1]

    path = "./ResNet18_%d.pth"%ranks[0]
    #print(path)
    model.load_state_dict(torch.load(path))
    #因为量化不支持GPU所以模型加载完成后同一导出为cpu并测试
    model = model.to('cpu')
    Convert_ONNX(model=model,size=(1,1,32,32),model_name="resnet.onnx")
    summary(model,input_size=(1,32,32),device='cpu')
    benchmark(model=model,size=(1,1,32,32))
    for i in ranks[1::]:
        os.remove("./ResNet18_%d.pth"%i)




if __name__ == "__main__":
    main()
