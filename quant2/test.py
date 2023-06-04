import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from shufflenet import ShuffleNet_v2
import torch.nn as nn
import numpy as np
import sys
sys.path.append("..") 
from data import myData


def get_acc(output, label):
    total = output.shape[0]
    pred_label = output.argmax(dim=1)
    num_correct = (pred_label == label).float().sum().item()
    return num_correct / total

def main():
    root_dir="../test_data"
    label_dir = np.load('../label_dir.npy',allow_pickle=True).item()
    my_dataset=[]
    for (key,val) in label_dir.items():
        my_dataset+=myData(root_dir , key,val)
 

    test_data=my_dataset
    #print('train:', len(test_data),"label:",label)

    data_loader_test = DataLoader(test_data,batch_size=64, shuffle=True)


    device = torch.device('cuda')

    for i in range(20):
        path="./ShuffleNet_v2_"+str(i)+".pth"
        #path="./ShuffleNet_v2.pth"
        #print(path)
        model = ShuffleNet_v2()
        model.load_state_dict(torch.load(path))
        model.eval()
        model.to(device)
        #print(model)
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
