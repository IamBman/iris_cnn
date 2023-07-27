import torch
from data import myData
from torch.utils.data import DataLoader
from resnet import ResNet18
import numpy as np
import sys
sys.path.append("..") 
from tool import get_acc


def main():
    root_dir="../test_data"
    label_dir = np.load('../label_dir.npy',allow_pickle=True)
    my_dataset=[]
    for val in range(len(label_dir)):
        my_dataset+=myData(root_dir , label_dir[val] , val)
 

    test_data=my_dataset
    #print('train:', len(test_data),"label:",label)

    data_loader_test = DataLoader(test_data,batch_size=64, shuffle=True)


    device = torch.device('cuda')

    for i in range(20):
        path="./ResNet18_"+str(i)+".pth"
        model = ResNet18()
        model.load_state_dict(torch.load(path))
        model.eval()
        model.to(device)
        test_acc = 0

        for i_batch,(x,label) in enumerate(data_loader_test):
            x= x.to(device)
            label = label.to(device)
            #print("label",label)
            y_ = model(x)
            #print("pre",y_.argmax(dim=1))

            test_acc += get_acc(y_,label)

        print("test_acc:%f,path:"%( test_acc / len(data_loader_test)),path) 

if __name__ == "__main__":
    main()
