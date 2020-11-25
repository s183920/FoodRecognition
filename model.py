import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(pretrained = True : bool): 
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = len(classes)  # 9 classes  + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 