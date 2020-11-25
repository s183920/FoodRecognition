"""
Load data into a dataset class that can be used to train our model.
"""

import torchvision.datasets as datasets
import numpy as np
import torch
import torchvision.transforms as transforms
import os

# dicts for converting classes to labels
classes = ['__background__', 'boiled peas', 'boiled potatoes', 'chopped lettuce', 'fried egg',
        'glass of milk', 'glass of water', 'meatballs', 'plain rice', 'plain spaghetti',
        'slice of bread']

def cls_to_label(cls : str):
  cls_to_label_dict = {j:i for i,j in enumerate(classes)}
  return cls_to_label_dict[cls]
  
def labels_to_cls(label : int):
  label_to_cls_dict = {i:j for i,j in enumerate(classes)}
  return label_to_cls_dict[label]

class foodDataset(torch.utils.data.Dataset):
    """
    Class to store the food data
    """
    def __init__(self, data_root:str, datatype:str = "train", transforms = None):
      self.data_root = data_root
      self.transforms = transforms
      self.splitImgPath = data_root + "VOCdevkit/VOC2007/ImageSets/Main/" + datatype + ".txt"
      with open(self.splitImgPath, "r") as splitIdx:
        self.imgNames = splitIdx.readlines()

      # self.dataset = datasets.VOCDetection(data_root, year='2007', image_set = datatype, transform=self.transforms)
      self.dataset = datasets.VOCDetection(data_root, year='2007', image_set = datatype)

    def __getitem__(self, idx):
      img = self.dataset[idx][0]
      obs = self.dataset[idx][1]["annotation"]["object"]
      num_objs = len(obs)

      image_id = torch.tensor([int(os.path.splitext(self.dataset[idx][1]["annotation"]["filename"])[0])])

      boxes = []
      labels = torch.ones((num_objs,), dtype=torch.int64)
      for i in range(num_objs):
        xmin = int(obs[i]["bndbox"]["xmin"])
        xmax = int(obs[i]["bndbox"]["xmax"])
        ymin = int(obs[i]["bndbox"]["ymin"])
        ymax = int(obs[i]["bndbox"]["ymax"])
        boxes.append([xmin, ymin, xmax, ymax])
        cls = obs[i]["name"]
        try:
          labels[i] *= cls_to_label(cls)
        except KeyError:
          raise KeyError(f"Image {image_id} had an unavailable label: {cls} ")
      boxes = torch.as_tensor(boxes, dtype=torch.float32)

      area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

      # suppose all instances are not crowd
      iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
      
      target = {}
      target["boxes"] = boxes
      target["labels"] = labels
      target["area"] = area
      target["image_id"] = image_id
      target["iscrowd"] = iscrowd

      return (img, target) if self.transforms is None else self.transforms(img, target)

    def __len__(self):
      return len(self.imgNames)