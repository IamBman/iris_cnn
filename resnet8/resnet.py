import torch
import torch.nn as nn
from torch.nn import functional as F

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class ResBlock(nn.Module):
    def __init__(self,ch_in,ch_out,stride=2):
        super(ResBlock,self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ch_out),
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or ch_in != ch_out:
            self.shortcut = LambdaLayer(lambda x:F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_out//4, ch_out//4), "constant", 0))



    def forward(self,x):
        out = self.left(x)
        #将2个卷积层的输出跟处理过的x相加，实现ResNet的基本结构
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet18(nn.Module):
    def __init__(self,num_classes = 497):
        super(ResNet18,self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = ResBlock(64, 128, stride=2)
        self.layer2 = ResBlock(128, 256, stride=2)
        self.layer3 = ResBlock( 256, 512, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def forward(self,x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        out = F.adaptive_avg_pool2d(out,[1,1])
        out = torch.flatten(out,1)
        out = self.fc(out)
        return out

#def main():
  #blk = ResBlock(3,64,stride=2)
  #tmp = torch.randn(2,3,32,32)
  #out = blk(tmp)
  #print(out.shape)

#  x = torch.randn(1,1,128,128)
#  model = ResNet18()
#  out = model(x)
#  print(model)
#if __name__ == '__main__':
#   main()
