import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
#from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("tkagg")
import cv2

from rcnn_pin_model.dataset import PinData

#note: from running this, loss seems to have diminishing returns after 7 epochs
EPOCHS = 5
PATH = './rcnn_pin_model/' + f"{EPOCHS}pin_net.pth"

def show_boxes_on_img(img_path, boxes):
    img = cv2.imread(img_path)
    for box in boxes:
      cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (0,255,0), 2)
    plt.imshow(img)
    plt.show()

def train():
  # Load the pre-trained Faster R-CNN model with a ResNet-50 backbone
  #model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
  model = fasterrcnn_resnet50_fpn(pretrained=True)

  # Number of classes (your dataset classes + 1 for background)
  num_classes = 3  # For example, 2 classes + background

  # Get the number of input features for the classifier
  in_features = model.roi_heads.box_predictor.cls_score.in_features

  # Replace the head of the model with a new one (for the number of classes in your dataset)
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  # Load dataset
  dataset = PinData()
  # Split into train and validation sets
  indices = torch.randperm(len(dataset)).tolist()
  # Create data loaders
  train_loader = DataLoader(dataset, batch_size=4, shuffle=True, 
                                    collate_fn=lambda x: tuple(zip(*x)))
  # Move model to GPU if available
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)

  # Set up the optimizer
  params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, 
                                                    weight_decay=0.0005)
  # Learning rate scheduler
  lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, 
                                                                gamma=0.1)
  # Train the model
  model.train()
  for epoch in range(EPOCHS):
      train_loss = 0.0

    # Training loop
      for images, targets, _ in train_loader:
          images = list(image.to(device) for image in images)
          targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

          # Zero the gradients
          optimizer.zero_grad()

          # Forward pass
          loss_dict = model(images, targets)
          losses = sum(loss for loss in loss_dict.values())

          # Backward pass
          losses.backward()
          optimizer.step()
          train_loss += losses.item()

      # Update the learning rate
      lr_scheduler.step()
      print(f'Epoch: {epoch + 1}, Loss: {train_loss / len(train_loader)}')

  torch.save(model.state_dict(), PATH)
  print("Training complete!")

def test():
  model = fasterrcnn_resnet50_fpn(pretrained=True)
  #model = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
  num_classes = 3  # For example, 2 classes + background
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

  try:
    model.load_state_dict(torch.load(PATH))
  except:
    print(f"No model saved at {PATH}")
    return 
  
  dataset = PinData()
  train_loader = DataLoader(dataset, batch_size=4, shuffle=True, 
                                    collate_fn=lambda x: tuple(zip(*x)))
  # Move model to GPU if available
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  model.to(device)

  # Set the model to evaluation mode
  model.eval()
  # Test on a new image
  with torch.no_grad():
      for images, targets, files in train_loader:
          images = list(img.to(device) for img in images)
          predictions = model(images)
          # Example: print the bounding boxes and labels for the first image
          for i in range(len(predictions)):
            print(f"For file: {files[i]}")
            print(f"\t{predictions[i]['boxes']}")
            show_boxes_on_img(dataset.dataset_dir + "/" + files[i], targets[i]['boxes'])
            show_boxes_on_img(dataset.dataset_dir + "/" + files[i], predictions[i]['boxes'])


def testPinData():
  data = PinData()
  for i in range(len(data)):
    img, label_info = data[i]
    boxes, labels = label_info.values()
    print(f"img: {img}, boxes: {boxes}, labels: {labels}\n")
    print(f"img size: {img.size()}")

def do_fast_rcnn():
  #TODO: implement this
  #testPinData()
  train()
  test()