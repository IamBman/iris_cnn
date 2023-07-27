import torch
import os
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from data import myData
from tool import get_acc
import shutil

#root_dir是训练集
def train(model,root_dir,label_dir,model_name):
    
    my_dataset=[]

    for val in range(len(label_dir)):
        my_dataset+=myData(root_dir , label_dir[val] , val)

    train_data=my_dataset

    data_loader_train = DataLoader(train_data,batch_size=64, shuffle=True)

    device = torch.device('cuda')
    model.to(device)
    #print(model)
    criteon = nn.CrossEntropyLoss().to(device) #包含了softmax操作
    optimizer = torch.optim.Adam(model.parameters(),lr=1e-3)
    
    list = os.listdir(".")
    for name in list :
        if name == "logs" :
            shutil.rmtree("./logs") 

    writer = SummaryWriter("./logs")
    fake_img = torch.zeros((1, 1, 32, 32), device=device)
    writer.add_graph(model, fake_img)

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
        torch.save(model.state_dict(),"./%s_%d.pth"%(model_name,epoch))
        writer.add_scalar('training loss',train_loss / len(data_loader_train), epoch )
        writer.add_scalar('train_acc',train_acc / len(data_loader_train), epoch )
        writer.close()
        





