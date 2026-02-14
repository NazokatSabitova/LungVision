import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras import layers, models
from models import build_model
from eval import evaluate_model
import keras.backend as K

# ==========================
# CONFIG
# ==========================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 50
IMAGE_DIR = "data/nih/images-small"          # folder with images
CSV_PATH = "data/nih/train-small.csv"         # csv file

# ==========================
# LABELS
# ==========================
labels = [
"Atelectasis","Cardiomegaly","Consolidation","Edema",
"Effusion","Emphysema","Fibrosis","Hernia",
"Infiltration","Mass","Nodule","Pleural_Thickening",
"Pneumonia","Pneumothorax"
]

# ==========================
# LOAD DATA
# ==========================
df = pd.read_csv(CSV_PATH)

# ==========================
# PATIENT-LEVEL SPLIT
# ==========================
unique_patients = df["PatientId"].unique()

train_pat, temp_pat = train_test_split(
    unique_patients, test_size=0.3, random_state=42)

val_pat, test_pat = train_test_split(
    temp_pat, test_size=0.5, random_state=42)

train_df = df[df["PatientId"].isin(train_pat)]
val_df   = df[df["PatientId"].isin(val_pat)]
test_df  = df[df["PatientId"].isin(test_pat)]

print("Train:", len(train_df))
print("Val:", len(val_df))
print("Test:", len(test_df))

# ==========================
# DATA GENERATORS
# ==========================
datagen = ImageDataGenerator(rescale=1./255)

train_gen = datagen.flow_from_dataframe(
    train_df,
    directory=IMAGE_DIR,
    x_col="Image",
    y_col=labels,
    class_mode="raw",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_gen = datagen.flow_from_dataframe(
    val_df,
    directory=IMAGE_DIR,
    x_col="Image",
    y_col=labels,
    class_mode="raw",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_gen = datagen.flow_from_dataframe(
    test_df,
    directory=IMAGE_DIR,
    x_col="Image",
    y_col=labels,
    class_mode="raw",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=False
)

# ==========================
# CLASS IMBALANCE WEIGHTS
# ==========================
freq_pos = train_df[labels].mean().values
freq_neg = 1 - freq_pos

pos_weights = freq_neg
neg_weights = freq_pos

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):

    pos_weights = tf.constant(pos_weights, dtype=tf.float32)
    neg_weights = tf.constant(neg_weights, dtype=tf.float32)

    def weighted_loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        loss = - (
            pos_weights * y_true * tf.math.log(y_pred + epsilon) +
            neg_weights * (1 - y_true) * tf.math.log(1 - y_pred + epsilon)
        )

        return tf.reduce_mean(loss, axis=-1)

    return weighted_loss


# ==========================
# MODEL
# ==========================
models_to_test = ["densenet", "resnet", "efficientnet", "mobilenet"]

results = {}

for name in models_to_test:
    print(f"\nTraining {name.upper()}")

    model = build_model(name)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(multi_label=True)]
    )

    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS)

    acc, precision, recall, f1 = evaluate_model(model, test_gen, test_df, labels)

    results[name] = {
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }

# base_model = DenseNet121(
#     weights="imagenet",
#     include_top=False,
#     input_shape=(IMG_SIZE, IMG_SIZE, 3)
# )

# base_model.trainable = False

# model = models.Sequential([
#     base_model,
#     layers.GlobalAveragePooling2D(),
#     layers.Dense(14, activation="sigmoid")
# ])

# model.compile(
#     optimizer="adam",
#     loss=get_weighted_loss(pos_weights, neg_weights),
#     metrics=[tf.keras.metrics.AUC(multi_label=True)]
# )

# ==========================
# TRAIN
# ==========================
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ==========================
# FINE-TUNING
# ==========================
base_model.trainable = True

for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss=get_weighted_loss(pos_weights, neg_weights),
    metrics=[tf.keras.metrics.AUC(multi_label=True)]
)

model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=5
)

# ==========================
# EVALUATION (ROC-AUC per class)
# ==========================
y_true = test_df[labels].values
y_pred = model.predict(test_gen)

print("\nPer-class ROC-AUC:")
for i, label in enumerate(labels):
    auc = roc_auc_score(y_true[:, i], y_pred[:, i])
    print(f"{label}: {round(auc, 3)}")

print("\nMean AUC:", round(np.mean([
    roc_auc_score(y_true[:, i], y_pred[:, i])
    for i in range(len(labels))
]), 3))

# ==========================
# SAVE MODEL
# ==========================
model.save("chest_xray_14_model.h5")
print("Model saved!")
