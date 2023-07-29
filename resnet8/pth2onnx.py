from resnet import ResNet18
from torchsummary import summary
import torch

import sys
sys.path.append("..")
from my_onnx import Convert_ONNX
from my_bench import benchmark



#Function to Convert to ONNX 

model=ResNet18()
model = model.to('cuda')
model.load_state_dict(torch.load("./ResNet18_15.pth"))
#因为量化不支持GPU所以模型加载完成后同一导出为cpu并测试
model = model.to('cpu')
Convert_ONNX(model=model,size=(1,1,32,32),model_name="resnet.onnx")
summary(model,input_size=(1,32,32),device='cpu')
benchmark(model=model,size=(1,1,32,32))
