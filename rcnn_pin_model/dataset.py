import os
import PIL
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision
from pin.cv_detect import detect_edges
import cv2

#Set seeds
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)


# Custom Dataset class or using an existing one
class PinData(Dataset):
    def __init__(self, transforms=None):
        #Define dataset
        current_dir = os.path.join(os.getcwd(),'rcnn_pin_model')
        self.dataset_dir = os.path.join(os.getcwd(), 'ic-images-defects')
        
        self.all_filenames = os.listdir(self.dataset_dir)
        self.all_labels = pd.read_csv(os.path.join(current_dir,'pin_labels.csv'),header=0,index_col=0)
        self.label_meanings = self.all_labels.columns.values.tolist()

    def __getitem__(self, idx):
        selected_filename = self.all_filenames[idx]
        """ img = cv2.imread(os.path.join(self.dataset_dir,selected_filename))
        #TODO: investigate this
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 210, 255, cv2.THRESH_BINARY) """
        img = detect_edges(os.path.join(self.dataset_dir,selected_filename))


        imagepil = PIL.Image.fromarray(img).convert('RGB')
        
        #convert image to Tensor and normalize
        image = to_tensor_and_normalize(imagepil)
    
        # Load corresponding bounding boxes and labels

        boxes, labels = self.GetBoxesAndLabels(idx)

        # Create a target dictionary
        target = {
            "boxes": boxes,
            "labels": labels
        }

        return image, target, selected_filename
    
    def __len__(self):
        # Return the length of your dataset
        return len(self.all_filenames)
        
    def GetBoxesAndLabels(self, idx):
        selected_filename = self.all_filenames[idx]
        selected_boxes = self.all_labels.loc[selected_filename,:].values[0]
        boxes = []
        labels = []

        boxes_list = selected_boxes.split('|')
        for box in boxes_list:
            box_data = box[1:-1:].split('-')
            box_dim = []
            box_dim.append(int(box_data[0]))
            box_dim.append(int(box_data[1]))
            box_dim.append(box_dim[0] + int(box_data[2]))
            box_dim.append(box_dim[1] + int(box_data[3]))

            boxes.append(box_dim)
            labels.append(1)

        ret_boxes = torch.tensor(boxes, dtype=torch.float32)
        ret_labels = torch.tensor(labels, dtype=torch.int64)
        return ret_boxes, ret_labels

def to_tensor_and_normalize(imagepil): #Done with testing
    """Convert image to torch Tensor and normalize using the ImageNet training
    set mean and stdev taken from
    https://pytorch.org/docs/stable/torchvision/models.html.
    Why the ImageNet mean and stdev instead of the PASCAL VOC mean and stdev?
    Because we are using a model pretrained on ImageNet."""
    #TODO: edit normalization here
    ChosenTransforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),])
    return ChosenTransforms(imagepil)