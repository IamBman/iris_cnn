from shufflenet import ShuffleNet_v2
import torch.onnx 
from torch.ao.quantization.quantize import prepare, convert
from torch.ao.quantization.qconfig import get_default_qconfig
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary
import sys
sys.path.append("..") 
from data import myData
from tool import get_acc

weights = './ShuffleNet_V2_19.pth'

def main():
    device = torch.device('cpu')
    root_dir="../enrollment_data"
    label_dir = np.load('../label_dir.npy',allow_pickle=True).item()
    
    my_dataset=[]
    for (key,val) in label_dir.items():
        my_dataset+=myData(root_dir , key,val)


    train_data=my_dataset
    data_loader_train = DataLoader(train_data,batch_size=64, shuffle=True)

    model=ShuffleNet_v2()
    model.eval()
    model.load_state_dict(torch.load(weights))
    #print(model)

    fuse_list=[["conv1.0","conv1.1","conv1.2"],
          ["layer1.dwconv_l1.seq.0","layer1.dwconv_l1.seq.1"],
          ["layer1.conv_l2.seq.0","layer1.conv_l2.seq.1","layer1.conv_l2.seq.2"],
          ["layer1.conv_r1.seq.0","layer1.conv_r1.seq.1","layer1.conv_r1.seq.2"],
          ["layer1.dwconv_r2.seq.0","layer1.dwconv_r2.seq.1"],
          ["layer1.conv_r3.seq.0","layer1.conv_r3.seq.1","layer1.conv_r3.seq.2"],
          ["layer2.dwconv_l1.seq.0","layer2.dwconv_l1.seq.1"],
          ["layer2.conv_l2.seq.0","layer2.conv_l2.seq.1","layer2.conv_l2.seq.2"],
          ["layer2.conv_r1.seq.0","layer2.conv_r1.seq.1","layer2.conv_r1.seq.2"],
          ["layer2.dwconv_r2.seq.0","layer2.dwconv_r2.seq.1"],
          ["layer2.conv_r3.seq.0","layer2.conv_r3.seq.1","layer2.conv_r3.seq.2"],
          ["layer3.dwconv_l1.seq.0","layer3.dwconv_l1.seq.1"],
          ["layer3.conv_l2.seq.0","layer3.conv_l2.seq.1","layer3.conv_l2.seq.2"],
          ["layer3.conv_r1.seq.0","layer3.conv_r1.seq.1","layer3.conv_r1.seq.2"],
          ["layer3.dwconv_r2.seq.0","layer3.dwconv_r2.seq.1"],
          ["layer3.conv_r3.seq.0","layer3.conv_r3.seq.1","layer3.conv_r3.seq.2"]
          ]
    torch.quantization.fuse_modules(model,fuse_list, inplace=True)
    #print(model)

    backend = "qnnpack" # 若为 x86，否则为 'qnnpack' 
    model.qconfig = get_default_qconfig(backend)
    model_static_quantized = prepare(model)

    for (data,label) in data_loader_train:
        model_static_quantized(data)

    model_static_quantized_int8 = torch.quantization.convert(model_static_quantized)


    print(model_static_quantized_int8)
    traced_cell = torch.jit.trace(model_static_quantized_int8, torch.rand(1,1,32,32))
    #print(traced_cell)
    traced_cell.save('shufflenet_quant.pt')
    torch.save(model_static_quantized_int8.state_dict(),"./shufflenet_quant.pth")


    root_dir="../test_data"
    my_dataset=[]
    for (key,val) in label_dir.items():
        my_dataset+=myData(root_dir , key,val)
 
 
    test_data=my_dataset
    #print('train:', len(test_data),"label:",label)
 
    data_loader_test = DataLoader(test_data,batch_size=1, shuffle=True)
    
        #path="./shufflenet_"+str(i)+".pth"
        #print(path)
        #print(model)
    test_acc1 = 0
 
    for i_batch,(x,label) in enumerate(data_loader_test):
        x= x.to(device)
        label = label.to(device)
        y_ = model_static_quantized_int8(x)

        test_acc1 += get_acc(y_,label)
        
    print("test_acc:%f"%( test_acc1 / len(data_loader_test))) 
        #torch.jit.save(torch.jit.script(model_static_quantized_int8),'shufflenet_quant.pt')
    summary(model_static_quantized_int8,input_size=(1,32,32))
 


if __name__ == "__main__":
    main()
