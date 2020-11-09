#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:26:15 2020

@author: hso

Dataloader for Hedia food recognition project
"""
# Import dependancies
import torchvision.datasets as datasets
import numpy as np
import torch
import torchvision.transforms as transforms

data_root = '/mnt/d/Git/GitHub/Studie/02456_Deep_learning/FoodRecognition/data/VOC'

# class foodData:
#     def __init__(self, data_root):


# Transform images
transform = transforms.Compose(
    [transforms.Resize([256,256]),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
    ]
)

# Import VOC-dataset
voc_trainset = datasets.VOCDetection(data_root, year='2007', image_set='train',transform=transform)
voc_testset = datasets.VOCDetection(data_root, year='2007', image_set='test',transform=transform)

# Import VOC-dataset
voc_trainvalset = datasets.VOCDetection(data_root, year='2007', image_set='trainval',transform=transform)
voc_valset = datasets.VOCDetection(data_root, year='2007', image_set='val',transform=transform)

print(voc_trainset[0])
# Print sizes
print('-'*40)
print('VOC2007-train')
print(len(voc_trainset))
print(voc_trainset)

print('-'*40)
print('VOC2007-test')
print(len(voc_testset))
print(voc_testset)

print('-'*40)
print('VOC2007-trainval')
print(len(voc_trainvalset))
print(voc_trainvalset)

print('-'*40)
print('VOC2007-val')
print(len(voc_valset))
print(voc_valset)


# Print shape of each image
# for i, sample in enumerate(voc_trainset, 1):
#     image, annotation = sample[0], sample[1]['annotation']
#     objects = annotation['object']
#     import matplotlib.pyplot as plt
#     show_image = np.array(image)
#     plt.imshow(show_image)
#     plt.show()
#     print('{} object:{}'.format(i, len(objects)))
#     print(show_image.shape)
#     if i == 2:
#         break


trainloader = torch.utils.data.DataLoader(voc_trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(voc_testset, batch_size=4,
                                         shuffle=True, num_workers=2)
print('trainloader')


train_data_iter = iter(trainloader)
test_data_iter = iter(testloader)


classes = ('boiled_peas', 'boiled_potatoes', 'chopped_lettuce', 'fried_egg',
        'glass_of_milk', 'glass_of_water', 'meatballs', 'plain_rice', 'plain_spaghetti',
        'slice_of_bread')
print('used classes:', classes)


print("# Training data")
print("Number of points:", len(voc_trainset))
x, y = next(iter(trainloader))
print("Batch dimension [B x C x H x W]:", x.shape)
# print(voc_trainset[1][1]["annotation"]["object"][0]["name"])
# print(np.shape(voc_trainset))
# print("Number of distinct labels:", len(set(voc_trainset[1])))
print("Number of distinct labels:", len(classes))


print("\n# Test data")
print("Number of points:", len(voc_testset))
x, y = next(iter(testloader))
print("Batch dimension [B x C x H x W]:", x.shape)
# print("Number of distinct labels:", len(set(voc_testset.targets)))
print("Number of distinct labels:", len(classes))
