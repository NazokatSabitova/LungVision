import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-4

IMAGE_DIR = "images"

LABELS = [
"Atelectasis","Cardiomegaly","Consolidation","Edema",
"Effusion","Emphysema","Fibrosis","Hernia",
"Infiltration","Mass","Nodule","Pleural_Thickening",
"Pneumonia","Pneumothorax"
]
#Har xil model tanlash uchun
MODEL_NAME = "densenet"
#MODEL_NAME = "resnet"
#MODEL_NAME = "efficientnet"
#MODEL_NAME = "mobilenet"
