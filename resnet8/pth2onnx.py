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
Convert_ONNX(model=model,size=(1,1,32,32),model_name="resnet.onnx")
summary(model,input_size=(1,32,32))
benchmark(model=model,size=(1,1,32,32))
