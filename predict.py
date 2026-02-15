import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from model import build_model
from config import *
# Predict single image
def predict_image(model_path, img_path, top_k=5, threshold=None):

    model = build_model()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image = Image.open(img_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.sigmoid(outputs).cpu().numpy()[0]

    sorted_indices = np.argsort(probs)[::-1]

    results = []
    for i in sorted_indices[:top_k]:
        results.append({
            "class": LABELS[i],
            "probability": float(probs[i])
        })

    return results, probs
