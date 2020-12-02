install_cython=$1
install_coco=$2

if [ $install_cython -eq 1 ]
then 
  pip install cython
else
  echo "cython will not be installed"
fi

# Install pycocotools, the version by default in Colab
# has a bug fixed in https://github.com/cocodataset/cocoapi/pull/354
if [ $install_coco -eq 1 ]
then
  pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
  # pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI
else
  echo "pycocotools will not be installed!"
fi

# Download TorchVision repo to use some files from
# references/detection
if [ ! -d vision ]
then
  git clone https://github.com/pytorch/vision.git
  cd vision
  git checkout v0.3.0

  cp references/detection/utils.py ../
  cp references/detection/transforms.py ../
  cp references/detection/coco_eval.py ../
  cp references/detection/engine.py ../
  cp references/detection/coco_utils.py ../
else
  echo "torchvision repo already installed"
fi

if [ ! -d models ]
then 
  mkdir models
fi

if [ ! -d logs ]
then 
  mkdir logs
fi

if [ ! -d loss ]
then 
  mkdir loss
fi
