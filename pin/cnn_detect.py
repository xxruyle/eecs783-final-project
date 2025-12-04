import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from util import defect_images
import cv2
from pin.depth_detect_pins import crop_from_boxes, only_get_boxes, run_depth_detect, show
import os

import torch.nn as nn
import torch.optim as optim

from pin_model.dataset import PinData
from pin_model.net import Net
from pin.fast_rcnn import do_fast_rcnn

BATCH_SIZE = 10
EPOCHS = 1000
TRAINED_EPOCHS = [1, 10, 50, 100, 1000]
classes = ('good', 'bad')
PATH = './pin_model/' + f"{EPOCHS}pin_net.pth"

def testPinData():
  data = PinData(os.path.join(os.path.join(os.getcwd(),'pin_model'), 'pin_images'))
  for i in range(len(data)):
    v_inputs, v_labels, v_idx, v_name = data[i].values()
    print(f"input size:{v_inputs.size()}\n")

def train(epochs):
  data = PinData(os.path.join(os.path.join(os.getcwd(),'pin_model'), 'pin_images'))
  trainloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

  net = Net()

  # Define a loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  print("Your network is ready for training!")

  print("Training...")
  for epoch in range(epochs):
      running_loss = 0.0
      for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1} of {epochs}", leave=True, ncols=80)):
          v_inputs, v_labels, v_idx, v_name = data.values()
          #print(f"{v_inputs}, {v_labels}")

          optimizer.zero_grad()
          outputs = net(v_inputs)
          loss = criterion(outputs, v_labels)
          loss.backward()
          optimizer.step()

  # Save our trained model
  path = './pin_model/' + f"{epochs}pin_net.pth"
  torch.save(net.state_dict(), path)

def simple_test():
  data = PinData(os.path.join(os.path.join(os.getcwd(),'pin_model'), 'pin_images'))
  trainloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=16)

  # Pick random photos from training set
  dataiter = iter(trainloader)
  images, labels, _, _ = next(dataiter).values()

  # Load our model
  net = Net()
  net.load_state_dict(torch.load(PATH))

  # Analyze images
  outputs = net(images)
  _, predicted = torch.max(outputs, 1)

  # Show results
  for i in range(BATCH_SIZE):
      # Add new subplot
      plt.subplot(2, int(BATCH_SIZE/2), i + 1)
      # Plot the image
      img = images[i]
      img = img / 2 + 0.5
      npimg = img.numpy()
      plt.imshow(np.transpose(npimg, (1, 2, 0)))
      plt.axis('off')
      # Add the image's label
      color = "green"
      label = classes[predicted[i]]
      #print(f" {(int)(labels[i][1])}, {predicted[i]}")
      if classes[(int)(labels[i][1])] != classes[predicted[i]]:
          color = "red"
          label = "(" + label + ")"
      plt.title(label, color=color)

  plt.suptitle('Objects Found by Model', size=20)
  plt.show()

def test(epochs):
  #stat tracking info
  count = 0
  true_neg = 0
  false_neg = 0
  true_pos = 0
  false_pos = 0

  data = PinData(os.path.join(os.path.join(os.getcwd(),'pin_model'), 'pin_images'))
  trainloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

  # Load our model
  net = Net()
  try:
    print("Loading " + './pin_model/' + f"{epochs}pin_net.pth" + "...")
    net.load_state_dict(torch.load('./pin_model/' + f"{epochs}pin_net.pth"))
  except:
    print(f"Can't find model with {epochs} epochs")
    return (0,0,0,0,0)

  # Pick random photos from training set
  dataiter = iter(trainloader)
  for data in dataiter:
    images, labels, _, name = data.values()

    # Analyze images
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    #use len here b/c can have less than batchsize in list returned
    for i in range(len(images)):
      label_val = (int)(labels[i][1])
      pred_val = predicted[i]
      print(f"{name[i]}: \tlabel: {classes[label_val]} \tpredicted: {classes[pred_val]}")
      count += 1
      if label_val == pred_val:
        if pred_val == 0:
           true_neg += 1
        else:
           true_pos += 1
      else:
        if pred_val == 0:
           false_neg += 1
        else:
           false_pos += 1

  print("="*80 + "\n" + "\t\tSTATS\n" + "="*80)
  print(f"Count: {count}\n\ttrue  negative: {true_neg}\ttrue  positive: {true_pos}\n\tfalse negative: {false_neg}\tfalse positive: {false_pos}")
  return (count, true_neg, true_pos, false_neg, false_pos)

def test_actual(epochs):
  only_get_boxes()
  #run_depth_detect()
  crop_from_boxes()

  #print(defect_images)

  data = PinData(os.path.join(os.getcwd(),'ic-crops'))
  trainloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

  # Load our model
  net = Net()
  try:
    print("Loading " + './pin_model/' + f"{epochs}pin_net.pth" + "...")
    net.load_state_dict(torch.load('./pin_model/' + f"{epochs}pin_net.pth"))
  except:
    print(f"Can't find model with {epochs} epochs")
    return (0,0,0,0,0)
  
  dataiter = iter(trainloader)
  for data in dataiter:
    images, labels, _, names = data.values()

    # Analyze images
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)
    for i in range(len(images)):
       pred_val = predicted[i]
       name = names[i]
       name_idencies = name[:-4].split('_')
       file_idx = int(name_idencies[0])
       box_idx = int(name_idencies[1])
       defect_images[file_idx][2][box_idx] = pred_val
  
  #print(defect_images)

def show_results():   
  for img_path, boxes, labels in defect_images:
    img = cv2.imread(img_path)
    for i in range(len(boxes)):
      if labels[i] == 1:
          color = (0, 0, 255)  # red for outlier
      else:
          color = (0, 255, 0)  # green bb 

      box = boxes[i]
      # Draw rectangle
      cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), color, 2)
    show(img, "temp")


def train_all():
  for epoch in TRAINED_EPOCHS:
    train(epoch)

def test_all():
  for epoch in TRAINED_EPOCHS:
    count, true_neg, true_pos, false_neg, false_pos = test(epoch)
    if (count == 0):
       continue
    print(f"Results for {epoch} epochs")
    print(f"\ttrue  negative: {(true_neg/count) * 100}\ttrue  positive: {(true_pos/count) * 100}\n\tfalse negative: {(false_neg/count) * 100}\tfalse positive: {(false_pos/count) * 100}")

def run_cnn_img_detect():
  #train_all()
  #test_all()
  test_actual(EPOCHS)
  show_results()