#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 00:18:42 2020

@author: hso

This .py-file follows the structure found in https://debuggercafe.com/faster-rcnn-object-detection-with-pytorch/
"""

import torchvision.transforms as transforms
import cv2
import numpy
import numpy as np
from coco_names import COCO_INSTANCE_CATEGORY_NAMES as coco_names

# Create differnt color for each class
COLORS = np.random.uniform(0, 255, size=(len(coco_names), 3))

# Define Torchvision image transforms
transform = transforms.Compose([
    transforms.ToTensor(),
])

def predict(image, model, device, detection_threshold):
    # transform the image to tensor
    image = transform(image).to(device)
    image = image.unsqueeze(0) # Add batch dimension
    outputs = model(image) # Get predictions on the image
    
    # Get predicited class names
    pred_classes = [coco_names[i] for i in outputs[0]['labels'].cpu().numpy()]
    # Get score for the predicted objects
    pred_scores = outputs[0]['scores'].detach().cpu().numpy()
    # Get the predicted bounding boxes
    pred_bboxes = outputs[0]['boxes'].detach().cpu().numpy()
    # Get boxes above the threshold score
    boxes = pred_bboxes[pred_scores >= detection_threshold].astype(np.int32)
    return boxes, pred_classes, outputs[0]['labels']

def draw_boxes(boxes, classes, labels, image):
    # Read the image with OpenCV
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
    for i, box in enumerate(boxes):
        color = COLORS[labels[i]]
        cv2.rectangle(
            image,
            (int(box[0]), int(box[1])),
            (int(box[2]), int(box[3])),
            color, 2
        )
        cv2.putText(image, classes[i], (int(box[0]), int(box[1]-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, 
                    lineType=cv2.LINE_AA)
    return image
