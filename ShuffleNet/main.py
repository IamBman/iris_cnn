import sys
from shufflenet import ShuffleNet_v2
import numpy as np


sys.path.append("..") 
from my_train import train
from my_test import test

def main():
    train_dir="../enrollment_data"
    test_dir="../test_data"
    label_dir = np.load('../label_dir.npy',allow_pickle=True).item()
    
    model = ShuffleNet_v2()
    train(root_dir=train_dir,label_dir=label_dir,model=model,model_name="shufflenet_v2")
    test(root_dir=test_dir,label_dir=label_dir,model=model,model_name="shufflenet_v2")
        
if __name__ == "__main__":
    main()





