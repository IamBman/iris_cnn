from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image
import torch
import os


class myData(Dataset):

    def __init__(self,root_dir,label_dir,label,transforms=transforms.Compose([transforms.Resize(32),transforms.Grayscale(),transforms.ToTensor()])):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.label=label
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.image_path=os.listdir(self.path)
        self.transforms=transforms

    def __getitem__(self,idx):
        image_name=self.image_path[idx]
        image_item_path=os.path.join(self.root_dir,self.label_dir,image_name)
        image=Image.open(image_item_path,"r")
        image=self.transforms(image)
        label=self.label
        label=torch.tensor(label)
        return image,label

    def __len__(self):
        return len(self.image_path)

#root_dir="./enrollment_data"
#label_dir="999_L"
#my_data=myData(root_dir, label_dir,497)
#img,lab=my_data[5]
#print(lab)

