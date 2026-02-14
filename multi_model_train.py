import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    DenseNet121,
    ResNet50,
    EfficientNetB0,
    MobileNetV2
)

# ==========================
# GPU CHECK
# ==========================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU detected:", gpus)
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected. Running on CPU.")

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 5
IMAGE_DIR = "data/nih/images-small/"

TRAIN_CSV = "data/nih/train-small.csv"
VAL_CSV   = "data/nih/valid-small.csv"
TEST_CSV  = "data/nih/test.csv"

labels = [
"Atelectasis","Cardiomegaly","Consolidation","Edema",
"Effusion","Emphysema","Fibrosis","Hernia",
"Infiltration","Mass","Nodule","Pleural_Thickening",
"Pneumonia","Pneumothorax"
]

# ==========================
# LOAD DATA
# ==========================
train_df = pd.read_csv(TRAIN_CSV)
val_df   = pd.read_csv(VAL_CSV)
test_df  = pd.read_csv(TEST_CSV)

train_df[labels] = train_df[labels].astype("float32")
val_df[labels]   = val_df[labels].astype("float32")
test_df[labels]  = test_df[labels].astype("float32")

# ==========================
# DATA GENERATORS
# ==========================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_dataframe(
    train_df, IMAGE_DIR, "Image", labels,
    class_mode="raw",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_gen = val_datagen.flow_from_dataframe(
    val_df, IMAGE_DIR, "Image", labels,
    class_mode="raw",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_gen = val_datagen.flow_from_dataframe(
    test_df, IMAGE_DIR, "Image", labels,
    class_mode="raw",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# MODEL FACTORY
# ==========================
def build_model(name):

    if name == "DenseNet121":
        base = DenseNet121(weights="imagenet", include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif name == "ResNet50":
        base = ResNet50(weights="imagenet", include_top=False,
                        input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif name == "EfficientNetB0":
        base = EfficientNetB0(weights="imagenet", include_top=False,
                              input_shape=(IMG_SIZE, IMG_SIZE, 3))
    elif name == "MobileNetV2":
        base = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        raise ValueError("Unknown model")

    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(14, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(multi_label=True)]
    )

    return model

# ==========================
# EVALUATION FUNCTION
# ==========================
def evaluate_model(model):

    y_true = test_df[labels].values
    y_pred = model.predict(test_gen)

    y_pred_bin = (y_pred > 0.5).astype(int)

    acc = accuracy_score(y_true.flatten(), y_pred_bin.flatten())
    precision = precision_score(y_true, y_pred_bin, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred_bin, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred_bin, average="macro", zero_division=0)
    auc = roc_auc_score(y_true, y_pred, average="macro")

    return acc, precision, recall, f1, auc

# ==========================
# TRAIN & COMPARE
# ==========================
models_to_test = ["DenseNet121", "ResNet50", "EfficientNetB0", "MobileNetV2"]
results = {}

for name in models_to_test:

    print(f"\n========== Training {name} ==========")

    model = build_model(name)

    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS
    )

    acc, precision, recall, f1, auc = evaluate_model(model)

    results[name] = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "AUC": auc
    }

    print(f"{name} Results:")
    print("Accuracy :", round(acc, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1       :", round(f1, 4))
    print("AUC      :", round(auc, 4))

# ==========================
# RESULTS TABLE
# ==========================
results_df = pd.DataFrame(results).T
print("\n================ FINAL COMPARISON ================")
print(results_df)

results_df.to_csv("model_comparison_results.csv")
print("\nSaved results to model_comparison_results.csv")
