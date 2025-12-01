import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import torch.nn as nn
import torch.optim as optim

from pin_model.dataset import PinData
from pin_model.net import Net

BATCH_SIZE = 10
EPOCHS = 50
TRAINED_EPOCHS = [1, 10, 50, 100]
classes = ('good', 'bad')
PATH = './pin_model/' + f"{EPOCHS}pin_net.pth"

def testPinData():
  data = PinData()
  for i in range(len(data)):
    v_inputs, v_labels, v_idx, v_name = data[i].values()
    print(f"input size:{v_inputs.size()}\n")

def train():
  data = PinData()
  trainloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

  net = Net()

  # Define a loss function and optimizer
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

  print("Your network is ready for training!")

  print("Training...")
  for epoch in range(EPOCHS):
      running_loss = 0.0
      for i, data in enumerate(tqdm(trainloader, desc=f"Epoch {epoch + 1} of {EPOCHS}", leave=True, ncols=80)):
          v_inputs, v_labels, v_idx, v_name = data.values()
          #print(f"{v_inputs}, {v_labels}")

          optimizer.zero_grad()
          outputs = net(v_inputs)
          loss = criterion(outputs, v_labels)
          loss.backward()
          optimizer.step()

  # Save our trained model
  torch.save(net.state_dict(), PATH)

def simple_test():
  data = PinData()
  trainloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

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

  data = PinData()
  trainloader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

  # Load our model
  net = Net()
  try:
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
      pred_val = (int)(labels[i][1])
      #print(f"{name[i]}: \tlabel: {classes[label_val]} \tpredicted: {classes[pred_val]}")
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

  #print("="*80 + "\n" + "\t\tSTATS\n" + "="*80)
  #print(f"Count: {count}\n\ttrue  negative: {true_neg}\ttrue  positive: {true_pos}\n\tfalse negative: {false_neg}\tfalse positive: {false_pos}")
  return (count, true_neg, true_pos, false_neg, false_pos)

def run_cnn_img_detect():
  # TODO: implement this
  #train()
  for epoch in TRAINED_EPOCHS:
    count, true_neg, true_pos, false_neg, false_pos = test(epoch)
    if (count == 0):
       continue
    print(f"Results for {epoch} epochs")
    print(f"\ttrue  negative: {(true_neg/count) * 100}\ttrue  positive: {(true_pos/count) * 100}\n\tfalse negative: {(false_neg/count) * 100}\tfalse positive: {(false_pos/count) * 100}")
  #testPinData()