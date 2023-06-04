import torch
import os
from torch.utils.data import DataLoader
from resnet import ResNet18
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..") 
from data import myData
from tool import get_acc


def main():
    root_dir="../enrollment_data"
    label_dir = np.load('../label_dir.npy',allow_pickle=True).item()
    
    my_dataset=[]
    for (key,val) in label_dir.items():
        #print(key)
        #print(val)
        my_dataset+=myData(root_dir , key,val)

    train_data=my_dataset
    #img,label=train_data[450]
    #print('train:', len(train_data),"label:",label)


#    val_data=transforms.Compose([
#        transforms.Resize([224,224]),
#        transforms.ToTensor()
#    ])

    data_loader_train = DataLoader(train_data,batch_size=64, shuffle=True)

    # x,label = iter(cifar10_train).next()
    # print('x:',x.shape,'label:',label.shape)

    device = torch.device('cuda')
    model = ResNet18()
    #model.load_state_dict(torch.load("./ResNet18_9.pth"))
    model.to(device)
    print(model)
    criteon = nn.CrossEntropyLoss().to(device) #包含了softmax操作
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    for epoch in range(20):
        train_loss = 0
        train_acc = 0

        model.train()
        for i_batch,(x,label) in enumerate(data_loader_train):
            #[b,3,32,32]
            #[b]
            x= x.to(device)
            label = label.to(device)
            #print(label)
            # y_:[b,10]
            # label:[b]
            y_ = model(x)
            loss = criteon(y_,label)

            #反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += get_acc(y_,label)

        model.eval()
        print("epoch:%d,train_loss:%f,train_acc:%f"%(epoch, train_loss / len(data_loader_train),
            train_acc / len(data_loader_train)))  
        torch.save(model.state_dict(),"./ResNet18_%d.pth"%(epoch))
        #torch.save(model,'./ResNet18_%d.pt'%(epoch))




if __name__ == "__main__":
    main()
