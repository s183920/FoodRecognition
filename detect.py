#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 00:20:09 2020

@author: hso
This .py-file follows the structure found in https://debuggercafe.com/faster-rcnn-object-detection-with-pytorch/

## Detect food in images - given pretrained model ##

"""

import torchvision
import numpy
import torch
import argparse
import cv2
import detect_utils
from PIL import Image

# Construct argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image/video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())

# Load model
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                    #min_size=args['min_size'])
model = torch.load("model_epoch18.pth",map_location=torch.device('cpu'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

image = Image.open(args['input'])
model.eval().to(device)
boxes, classes, labels = detect_utils.predict(image, model, device, 0.8)
image = detect_utils.draw_boxes(boxes, classes, labels, image)
cv2.imshow('Image', image)
save_name = "Pred_img"
cv2.imwrite(f"outputs/{save_name}.jpg", image)
cv2.waitKey(0)


