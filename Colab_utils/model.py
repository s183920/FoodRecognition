import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
classes = ['__background__', 'boiled peas', 'boiled potatoes', 'chopped lettuce', 'fried egg',
        'glass of milk', 'glass of water', 'meatballs', 'plain rice', 'plain spaghetti',
        'slice of bread']
def get_model(pretrained:bool = True): 
    # load a model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=pretrained)
    
    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    num_classes = len(classes)  # 9 classes  + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Device used is: {device}")
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)

    return model


