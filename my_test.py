import torch
from torch.utils.data import DataLoader
from data import myData
from tool import get_acc

def test(model,root_dir,label_dir,model_name):
    acc_list = []
    my_dataset=[]

    for val in range(len(label_dir)):
        my_dataset+=myData(root_dir , label_dir[val] , val)

    test_data=my_dataset

    data_loader_test = DataLoader(test_data,batch_size=64, shuffle=True)

    device = torch.device('cuda')

    for i in range(20):
        path="%s_%d.pth"%(model_name,i)
        #print(path)
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
        acc_list.append(test_acc/len(data_loader_test))

    return acc_list

