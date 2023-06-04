from resnet import ResNet18
from torchsummary import summary
import torch

import sys
sys.path.append("..")
from my_onnx import Convert_ONNX



#Function to Convert to ONNX 

model=ResNet18()
model = model.to('cuda')
model.load_state_dict(torch.load("./ResNet18_19.pth"))
Convert_ONNX(model=model,size=(64,1,32,32),model_name="resnet.onnx")
summary(model,input_size=(1,32,32))
