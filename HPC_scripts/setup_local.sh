#!/bin/bash

### Make python environment
python3.6 -m venv FoodRecognition-env

. FoodRecognition-env/bin/activate
python -m pip install --upgrade pip
# python -m pip install torch==1.5.1 torchvision==0.6.1 numpy matplotlib
# visdom scikit-image tqdm fire pprint cython torchnet
python -m pip install torch==1.1 torchvision==0.3.0
python -m pip install -r HPC_scripts/requirements.txt

deactivate


