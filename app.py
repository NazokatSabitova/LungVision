import os
import torch
import numpy as np
from flask import Flask, render_template, request
from PIL import Image
from torchvision import transforms

from config import *
from model import build_model

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load models
# ----------------------------
MODELS = {
    "MobileNet": "models/mobilenet.pth",
    "ResNet": "models/resnet.pth",
    "DenseNet": "models/densenet.pth",
    "EfficientNet": "models/efficientnet.pth"
}

def load_model(model_path, model_name):
    # Convert dropdown name to model name (e.g., "MobileNet" -> "mobilenet")
    model_type = model_name.lower()
    
    # Build model with specific type
    model = build_model(model_type)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


@app.route("/", methods=["GET", "POST"])
def index():

    image_path = None
    results = None
    selected_model = None
    critical_flag = False

    if request.method == "POST":

        file = request.files["file"]
        selected_model = request.form.get("model")

        if file:
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)
            image_path = f"uploads/{file.filename}"

            # Load selected model
            model = load_model(MODELS[selected_model], selected_model)

            image = Image.open(filepath).convert("RGB")
            image = transform(image).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                outputs = torch.sigmoid(model(image))
                probs = outputs.cpu().numpy()[0]

            # Sort predictions
            sorted_indices = np.argsort(probs)[::-1]

            results = []
            for i in sorted_indices:
                prob = float(probs[i])
                threshold = 0.5

                if prob > 0.7:
                    critical_flag = True

                results.append({
                    "class": LABELS[i],
                    "probability": prob,
                    "threshold": threshold
                })

    return render_template(
        "index.html",
        models=MODELS.keys(),
        image_path=image_path,
        results=results,
        model=selected_model,
        critical_flag=critical_flag
    )


if __name__ == "__main__":
    app.run(debug=True)