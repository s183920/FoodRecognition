#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 15:26:15 2020

@author: hso

Dataloader class for Hedia food recognition project
"""
# Import dependancies
import torchvision.datasets as datasets
import numpy as np
import torch
import torchvision.transforms as transforms

class foodData:
    """
    Class to store the food data

    importDataset must be used after altering any values for the dataloader to reflect the changes
    """
    def __init__(self, data_root):
        self.root = data_root
        self.setTransform([256, 256])
        self.setClasses(['boiled_peas', 'boiled_potatoes', 'chopped_lettuce', 'fried_egg',
        'glass_of_milk', 'glass_of_water', 'meatballs', 'plain_rice', 'plain_spaghetti',
        'slice_of_bread'])

        self.setBatchSize(4)

        self.importDataset()

    def setTransform(self, img_resize : list):
        """
        Defines tranformations for the images
        """
        self.transform = transforms.Compose(
            [transforms.Resize(img_resize),transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
            ]
        )


    def importDataset(self):
        """
        Imports VOC-dataset
        """
        self.voc_trainset = datasets.VOCDetection(data_root, year='2007', image_set='train',transform=self.transform)
        self.voc_testset = datasets.VOCDetection(data_root, year='2007', image_set='test',transform=self.transform)

        # Import VOC-dataset
        self.voc_trainvalset = datasets.VOCDetection(data_root, year='2007', image_set='trainval',transform=self.transform)
        self.voc_valset = datasets.VOCDetection(data_root, year='2007', image_set='val',transform=self.transform)

        # data loader
        self.trainloader = torch.utils.data.DataLoader(self.voc_trainset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=2)

        self.testloader = torch.utils.data.DataLoader(self.voc_testset, batch_size=self.batch_size,
                                          shuffle=True, num_workers=2)

        # iterators
        self.train_data_iter = iter(self.trainloader)
        self.test_data_iter = iter(self.testloader)

    def setClasses(self, classes:list):
        """
        Sets the classes used
        """
        self.classes = classes

    def setBatchSize(self, batch_size:int):
        self.batch_size = batch_size

    def getLoader(self, train : bool = True, iter : bool = False):
        """
        returns a dataloader
        ---------------------
        train   :   true if trainloader should be returned and false if test loader should be returned
        iter    :   true if an iterator of the loader should be returned
        """
        return (self.train_data_iter if train else self.test_data_iter) if iter else (self.trainloader if train else self.testloader) 

    def getDataset(self, type:str):
        """
        returns a dataset
        -----------------
        type    :   type of data set, can be "train", "test", "trainval" or "val"
        """
        if type is "train": 
            return self.voc_trainset
        elif type is "test": 
            return self.voc_testset
        elif type is "trainval": 
            return self.voc_trainvalset
        elif type is "val": 
            return self.voc_valset
        else:
            raise AssertionError("Type must be either \"train\", \"test\", \"trainval\" or \"val\"")

    # class foodImage:
        # def __init__(self):
        #     super(foodData)
    def getImg(self, data_type:str, idx:int):
        """
        returns an image
        -----------------
        data_type   :   type of data set, can be "train", "test", "trainval" or "val"
        idx         :   index of the image
        """
        return self.getDataset(data_type)[idx][0]

    def getClass(self, data_type:str, idx:int):
        """
        returns the class of a specific index of a data set
        -----------------
        data_type   :   type of data set, can be "train", "test", "trainval" or "val"
        idx         :   index of the image
        """
        return self.getDataset(data_type)[idx][1]["annotation"]["object"][0]["name"]

    def saveImg(self, data_type:str, idx:int, filename:str, unnormalize : bool = True):
        """
        plots the image from a specific data set
        -----------------
        data_type   :   type of data set, can be "train", "test", "trainval" or "val"
        idx         :   index of the image
        filename    :   name of the saved image
        unnormalize :   whether to unnormalize before plotting
        """
        import matplotlib.pyplot as plt
        img = data.getImg(data_type, idx)
        # print("size of img: ", img.size())
        img = img / 2 + 0.5 if unnormalize else img   # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig(filename)



if __name__ == "__main__":
    data_root = '/mnt/d/Git/GitHub/Studie/02456_Deep_learning/FoodRecognition/data/VOC'
    data = foodData(data_root)

    voc_trainset = data.voc_trainset
    voc_testset = data.voc_testset
    voc_trainvalset = data.voc_trainvalset
    voc_valset = data.voc_valset


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

    print('used classes:', data.classes)

    trainloader = data.getLoader(train = True)
    testloader = data.getLoader(train = False)
    print('trainloader')

    train_data_iter = data.train_data_iter
    test_data_iter = data.test_data_iter



    print("# Training data")
    print("Number of points:", len(voc_trainset))
    x, y = next(iter(trainloader))
    print("Batch dimension [B x C x H x W]:", x.shape)
    # print(voc_trainset[1][1]["annotation"]["object"][0]["name"])
    # print(np.shape(voc_trainset))
    # print("Number of distinct labels:", len(set(voc_trainset[1])))
    print("Number of distinct labels:", len(data.classes))


    print("\n# Test data")
    print("Number of points:", len(voc_testset))
    # x, y = next(iter(testloader))
    x, y = next(data.test_data_iter)
    print("Batch dimension [B x C x H x W]:", x.shape)
    # print("Number of distinct labels:", len(set(voc_testset.targets)))
    print("Number of distinct labels:", len(data.classes))

    # test own functions
    print("Test example:", data.getImg("test", 1))
    print("Class of example: ", data.getClass("test", 1))
    data.saveImg("test", 1, "./plots/test.png")
    data.saveImg("test", 1, "./plots/test2.png", False)

    


    











