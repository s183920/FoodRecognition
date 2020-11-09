#!/bin/bash

### Make python environment
python3.6 -m venv FoodRecognition-env

. FoodRecognition-env/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==1.5.1 torchvision==0.6.1 numpy matplotlib


deactivate


