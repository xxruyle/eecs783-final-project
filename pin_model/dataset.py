import os
import PIL
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision

'''
Note: pin images were universalized with this:  
    mogrify -resize 200x200 -background white -gravity center -extent 200x200 *.png
'''

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)

class PinData(Dataset):
    def __init__(self, img_dir):      
        #Define dataset
        current_dir = os.path.join(os.getcwd(),'pin_model')
        self.dataset_dir = img_dir #
        
        #E.g. self.all_filenames = ['006.png','007.png','008.png'] when setname=='val'
        self.all_filenames = os.listdir(self.dataset_dir) # ['1b1.png']
        #print(f"{self.all_filenames}")
        self.all_labels = pd.read_csv(os.path.join(current_dir,'pin_labels.csv'),header=0,index_col=0)
        self.label_meanings = self.all_labels.columns.values.tolist()
    
    def __len__(self):
        return len(self.all_filenames)
        
    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        imagepil = PIL.Image.open(os.path.join(self.dataset_dir,selected_filename)).convert('RGB')
        
        #convert image to Tensor and normalize
        image = to_tensor_and_normalize(imagepil)
        
        #load label
        #print(self.dataset_dir)
        label = torch.Tensor([0.0,0.0])
        try:
            label = torch.Tensor(self.all_labels.loc[selected_filename,:].values)
        except:
            label = torch.Tensor([0.0,0.0])

        
        sample = {'data':image, #preprocessed image, for input into NN
                  'label':label,
                  'img_idx':idx,
                  'name':selected_filename}
        return sample


def to_tensor_and_normalize(imagepil):
    ChosenTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),])
    return ChosenTransforms(imagepil)