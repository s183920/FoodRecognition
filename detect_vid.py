#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 26 00:21:24 2020

@author: hso
This .py-file follows the structure found in https://debuggercafe.com/faster-rcnn-object-detection-with-pytorch/
"""

import torchvision
import cv2
import torch
import argparse
import time
import detect_utils
from PIL import Image

# Construct the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input video')
parser.add_argument('-m', '--min-size', dest='min_size', default=800, 
                    help='minimum input size for the FasterRCNN network')
args = vars(parser.parse_args())
# Load the model from disk
#model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True, 
                                                    #min_size=args['min_size'])
model = torch.load("model_epoch18.pth",map_location=torch.device('cpu'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

cap = cv2.VideoCapture(args['input'])
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# Get frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define codec and create VideoWriter object 
out = cv2.VideoWriter("predict.mp4", 
                      cv2.VideoWriter_fourcc(*'mp4v'), 30, 
                      (frame_width, frame_height))

frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# Load the model onto the computation device
model = model.eval().to(device)

# Read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            # Get predictions for the current frame
            boxes, classes, labels = detect_utils.predict(frame, model, device, 0.8)
        
        # Draw boxes and show current frame on screen
        image = detect_utils.draw_boxes(boxes, classes, labels, frame)
        # Get the end time
        end_time = time.time()
        # Get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # Increment frame count
        frame_count += 1
        # Press `q` to exit
        wait_time = max(1, int(fps/4))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) ##
        cv2.imshow('image', image)
        out.write(image)
        if cv2.waitKey(wait_time) & 0xFF == ord('q'):
            break
    else:
        break

# Release VideoCapture()
cap.release()
# Close all frames and video windows
cv2.destroyAllWindows()
# Calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")
