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
        self.conv1 = nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=stride,padding=1) # ! (h-3+2)/2 + 1 = h/2 图像尺寸减半
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1) # ! h-3+2*1+1=h 图像尺寸没变化
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=False)

        self.extra = nn.Sequential()
        if stride != 1 or ch_in != ch_out:
#            self.extra = LambdaLayer(lambda x:F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, ch_out//4, ch_out//4), "constant", 0))
            self.extra = nn.Sequential(nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=stride), # ! 这句话是针对原图像尺寸写的，要进行element wise add
                                      )


    def forward(self,x):
        out = x
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        # short cut
        # ! element wise add [b,ch_in,h,w] [b,ch_out,h,w] 必须当ch_in = ch_out时才能进行相加
        out =self.relu(self.extra(out)+x) # todo self.extra强制把输出通道变成一致
        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3), # ! 
            nn.BatchNorm2d(64),
        )
        # 4个ResBlock
        #  [b,64,h,w] --> [b,128,h,w]
        self.block1x = ResBlock(64,128)
        self.block1 = ResBlock(64,64,stride=1)
        self.block1_ = ResBlock(64,64,stride=1)
        #  [b,128,h,w] --> [b,256,h,w]
        self.block2x = ResBlock(128,256)
        self.block2 = ResBlock(128,128,stride=1)
        #  [b,256,h,w] --> [b,512,h,w]
        self.block3x = ResBlock(256,512)
        self.block3 = ResBlock(256,256,stride=1)
        #  [b,512,h,w] --> [b,512,h,w]
        self.block4 = ResBlock(512,512,stride=1)

        self.outlayer = nn.Linear(512,497)

    def forward(self,x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        # [b,64,h,w] --> [b,1024,h,w]
        x = self.block1(x)
        x = self.block1_(x)
        x = self.block1x(x)
        x = self.block2(x)
        x = self.block2x(x)
        x = self.block3(x)
        x = self.block3x(x)
        x = self.block4(x)
        #print("after conv:",x.shape)
        #[b,512,h,w] --> [b,512,1,1]
        
        x = F.adaptive_avg_pool2d(x,[1,1])
        x = torch.flatten(x,1)
        #print("after conv:",x.shape)
        #flatten
        #x = x.view(x.shape[0],-1)
        x = self.outlayer(x)
        return x

# def main():
#     blk = ResBlock(3,64,stride=2)
#     tmp = torch.randn(2,3,32,32)
#     out = blk(tmp)
#     print(out.shape)

#     x = torch.randn(2,3,32,32)
#     model = ResNet18()
#     out = model(x)
#     print(out.shape)
# if __name__ == '__main__':
#     main()
