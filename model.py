import torch.nn as nn
import torchvision.models as models
from config import *

def build_model():

    if MODEL_NAME == "densenet":
        model = models.densenet121(weights="IMAGENET1K_V1")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, len(LABELS))

    elif MODEL_NAME == "resnet":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(LABELS))

    elif MODEL_NAME == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(LABELS))

    elif MODEL_NAME == "mobilenet":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(LABELS))

    else:
        raise ValueError("Unknown model")

    return model.to(DEVICE)
