#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:26:15 2020

@author: hso

Dataloader for Hedia food recognition project
"""
import os
import torch
import cv2
import torch.utils.data
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dic_path = '/home/hso/Documents/FoodRecognition/VOC2007/ImageSets/Main'
object_categories = ['boiled_peas', 'boiled_potatoes', 'chopped_lettuce', 'fried_egg',
        'glass_of_milk', 'glass_of_water', 'meatballs', 'plain_rice', 'plain_spaghetti',
        'slice_of_bread']
train_all = []
val_all = []

def read_all():
    files = os.listdir(dic_path)
    for fi in files:
        if('trainval' not in fi):
            num = 1
            for i,str in enumerate(object_categories,1):
                if (str in fi):
                    num = i
                    break
            if('train' in fi):
                f = open(dic_path+"/"+fi)
                iter_f = iter(f)
                for line in iter_f:
                    line = line[0:11]
                    train_all.append([line,num])
            else:
                f = open(dic_path+"/"+fi)
                iter_f = iter(f)
                for line in iter_f:
                    line = line[0:11]
                    val_all.append([line,num])
read_all()

class Data(torch.utils.data.Dataset):
    def __init__(self,li,transform=None,size=(224,224)):
        self.transform = transform
        self.size = size
        self.img = []
        self.lab = []
        for i in li:
            self.img.append(i[0])
            self.lab.append(int(i[1]))
    def __getitem__(self, index):
        img_path = '/home/hso/Documents/FoodRecognition/VOC2007/JPEGImages'+self.img[index]+'.jpg'
        image = cv2.imread(img_path)
        image = cv2.resize(image,self.size)
        image = self.transform(image)
        label = torch.LongTensor([self.lab[index]])
        return image,label
    def __len__(self):
        return len(self.img)
trainset = Data(train_all,transform)
train_loader = DataLoader(trainset,batch_size=40,shuffle=True,num_workers=0)
valset = Data(val_all,transform)
val_loader = DataLoader(valset,batch_size=40,shuffle=True,num_workers=0)

#plot a few examples
f, axarr = plt.subplots(4, 16, figsize=(16, 4))

# Load a batch of images into memory
images, labels = next(iter(train_loader))

for i, ax in enumerate(axarr.flat):
    ax.imshow(images[i].view(28, 28), cmap="binary_r")
    ax.axis('off')
    
plt.suptitle('Food recognition data')
plt.show()
