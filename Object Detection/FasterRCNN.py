import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torch.utils.data import Dataset, DataLoader

#Function to initialize weights (Xavier) 
def init_weights(m):
    if(type(m) == torch.nn.Linear or type(m) == torch.nn.Conv2d):
      torch.nn.init.xavier_uniform(m.weight)
      m.bias.data.fill_(0.01)

#Constructing the Faster RCNN. THe model is completely untrained at this point.
backbone = torchvision.models.alexnet().features
backbone.out_channels = 256
anchor_generator = AnchorGenerator(sizes = ((16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192),), aspect_ratios = ((0.25, 0.5, 1.0, 2.0, 4.0), ))
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names= ['0'], output_size = 7, sampling_ratio = 2)

#Initializing the weights of the model.
model = FasterRCNN(backbone, num_classes = 3, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler)
model.backbone.apply(init_weights)
model.rpn.apply(init_weights)
model.roi_heads.apply(init_weights)

#GPU stuff.
use_cuda = torch.cuda.is_available()

if (use_cuda):
  device = torch.device("cuda:0")
  torch.backends.cudnn.benchmark = True
  print("Using GPU")
else:
  device = "cpu"

model.to(torch.device(device))

#Working with the data.
classes = {"background" : 0 , "cat" : 1, "dog": 2} #Encoding the classes.

#Iterating over a directory that contains the anootations in an XML file. Return a dataframe of annotations.
def create_label_df(directory):
  labels = pd.DataFrame(columns = ["img", "size", "classes", "boxes"])
  labels = labels.astype("object")
  for file in os.listdir(directory):
    if (file.endswith("xml")):
      data = ET.parse(file)
      root = data.getroot()
      classes = []
      boxes = []
      for child in root:
        if (child.tag == "filename"):
          image = child.text
        elif (child.tag == "size"):
          size = []
          for sizes in child:
            size.append(sizes.text)
        elif (child.tag == "object"):
          for info in child:
            if (info.tag == "name"):
              classes.append(info.text)
            elif (info.tag == "bndbox"):
              bndbox = []
              for loc in info:
                bndbox.append(float(loc.text))
              boxes.append(bndbox)
      labels = labels.append(pd.DataFrame([[image, size, classes, boxes]], columns = labels.columns), ignore_index = True)
  return labels

#Custom dataset class. 
class image_data(Dataset):
  def __init__(self, df):
    self.data = df

  def __len__(self):
    return self.data.shape[0]

  def __getitem__(self, idx):
    img = self.data.loc[idx, "img"]
    labels = torch.as_tensor([classes[x] for x in self.data.loc[idx, "classes"]], dtype = torch.int64)
    boxes = torch.as_tensor(self.data.loc[idx, "boxes"], dtype = torch.float32)
    img_id = torch.tensor(idx, dtype = torch.int64)
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    iscrowd = torch.tensor([0] * len(labels), dtype = torch.uint8)

    targets = {"boxes": boxes, "labels": labels, "image_id": img_id, "area": area, "iscrowd": iscrowd}

    return torch.transpose(torch.transpose(torch.tensor(np.array(Image.open("/content/gdrive/My Drive/cats_dogs/" + img).convert("RGB")), dtype = torch.float32),0,2), 1, 2), targets

#Create a dataframe with all the annotations and shuffle it.
labels = create_label_df("./").sample(frac = 1).reset_index(drop = True)

#Setup train and validation data.
data = pascal_data(labels.iloc[0:800].reset_index(drop = True))
val_data = pascal_data(labels.iloc[800:1050, :].reset_index(drop = True))

#Need this when we stack images of different size in the train loader.
def collate_fn(batch):
  return tuple(zip(*batch))

#Setup train and val loader
train_loader = DataLoader(data, batch_size = 20, shuffle = True, num_workers = 4, collate_fn = collate_fn)
val_loader = DataLoader(val_data, batch_size = 55, shuffle = True, num_workers = 4, collate_fn = collate_fn)

#Returns average loss on the validation dataset.
def get_val_loss(model, val_loader):
  average_loss = []
  for images, targets in val_loader:
    images = [image.to(torch.device(device)) for image in list(images)]
    targets = [{k: v.to(torch.device(device)) for k, v in t.items()} for t in targets]
    loss_dict = model(images, targets)
    losses = sum(loss for loss in loss_dict.values())
    average_loss.append(losses)
  return sum(average_loss)/len(average_loss)

#Function to train model
def train(model, train_loader, val_loader, labels, num_epochs = 300, lr = 0.00003, momentum = 0.9):
  optimizer = torch.optim.SGD(model.parameters(), lr = lr)
  training_loss = []
  val_loss = []
  epochs = []
  val_epochs = []
  for epoch in range(num_epochs):
    epochs.append(epoch)
    print("EPOCH: " + str(epoch))
    average_loss = []
    for images, targets in train_loader:
      images = [image.to(torch.device(device)) for image in list(images)]
      targets = [{k: v.to(torch.device(device)) for k, v in t.items()} for t in targets]
      loss_dict = model(images, targets)
      losses = sum(loss for loss in loss_dict.values())
      optimizer.zero_grad()
      losses.backward()
      optimizer.step()
      average_loss.append(losses)
    print("Training Loss ", sum(average_loss)/len(average_loss))
    training_loss.append(sum(average_loss)/len(average_loss))
    print("Val Loss ", get_val_loss(model, val_loader))

    if(epoch == 150):
      #Decay learning rate
      lr *= 0.1
      optimizer = torch.optim.SGD(model.parameters(), lr = lr)
    if(epoch == 200):
      lr *= 0.5
      optimizer = torch.optim.SGD(model.parameters(), lr = lr)

  plt.plot(epochs, training_loss)

#Train and save the model
train(model, train_loader, val_loader, labels)
torch.save({"model_state_dict": model.state_dict()}, "./FasterRCNN.pt" )

#Function to display to image and the bounding boxes.
def plot_pred(model, labels, idx):
  img = np.array(Image.open("/content/gdrive/My Drive/cats_dogs/" + labels.loc[idx,"img"]).convert("RGB"))
  fig, ax = plt.subplots(1)
  ax.imshow(img)
  img = torch.transpose(torch.transpose(torch.tensor(img, dtype = torch.float32),0,2), 1, 2)
  model.eval()
  pred = model([img.to(torch.device(device))])
  keep = torchvision.ops.nms(pred[0]['boxes'], pred[0]['scores'], 0.1)
  
  for idx in keep:
    if(pred[0]['scores'][idx] == max(pred[0]['scores'])):
      box = pred[0]['boxes'][idx]
      print(pred)
      rect = patches.Rectangle((box[0],  box[1]), box[2] - box[0], box[3] - box[1],linewidth=2,edgecolor='r',facecolor='none')
      if (pred[0]["labels"][idx] == 1):
        text = "CAT"
      else:
        text = "DOG"
      ax.annotate(text, (0.5*(box[2] + box[0]), 0.5*(box[1] + box[3])), color = 'r')
      ax.add_patch(rect)

#Once trained, plot the predictions on different images.
plot_pred(model, labels, 1099) 