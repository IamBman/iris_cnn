import torch

import sys
sys.path.append("..")
from my_bench import benchmark

model = torch.jit.load("resnet_quant.pt")
benchmark(model=model,size=(1,1,32,32))