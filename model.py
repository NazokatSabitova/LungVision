import torch.nn as nn
import torchvision.models as models
from config import *

def build_model(model_name=None):
    if model_name is None:
        model_name = MODEL_NAME  # from config
    
    if model_name == "densenet":
        model = models.densenet121(weights="IMAGENET1K_V1")
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, len(LABELS))

    elif model_name == "resnet":
        model = models.resnet50(weights="IMAGENET1K_V1")
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(LABELS))

    elif model_name == "efficientnet":
        model = models.efficientnet_b0(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(LABELS))

    elif model_name == "mobilenet":
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, len(LABELS))

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model.to(DEVICE)