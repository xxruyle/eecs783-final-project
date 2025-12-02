import os
import pandas as pd
import torch


current_dir = os.path.join(os.getcwd(),'rcnn_pin_model')
dataset_dir = os.path.join(os.getcwd(), 'defect-images')

all_filenames = os.listdir(dataset_dir)
all_labels = pd.read_csv(os.path.join(current_dir,'pin_labels.csv'),header=0,index_col=0)
label_meanings = all_labels.columns.values.tolist()

print(f"{all_filenames} \n|\n {all_labels} \n|\n {label_meanings}")
for i in range(len(all_labels)):
  selected_filename = all_filenames[i]
  selected_boxes = all_labels.loc[selected_filename,:].values[0]
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
  print(ret_boxes)
  degen_boxes = ret_boxes[:, 2:] <= ret_boxes[:, :2]
  print(degen_boxes)