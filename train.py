import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score
from config import *
from dataset import get_dataloaders
from model import build_model
from evaluation import evaluate_model

def train():

    print("Device:", DEVICE)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    train_loader, val_loader, test_loader = get_dataloaders()

    model = build_model().to(DEVICE)

    # ---------------- Weighted BCE ----------------
    train_df = pd.read_csv("train.csv")
    freq_pos = train_df[LABELS].mean().values
    pos_weight = torch.tensor(
        (1 - freq_pos) / (freq_pos + 1e-7),
        dtype=torch.float32
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    scaler = torch.cuda.amp.GradScaler()

    best_val_auc = 0

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):

        model.train()
        running_loss = 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=False)

        for images, labels in loop:

            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE, non_blocking=True)
                outputs = torch.sigmoid(model(images))

                val_preds.append(outputs.cpu().numpy())
                val_labels.append(labels.numpy())

        y_val_pred = np.vstack(val_preds)
        y_val_true = np.vstack(val_labels)

        val_auc = roc_auc_score(y_val_true, y_val_pred, average="macro")
        val_f1 = f1_score(
            y_val_true,
            (y_val_pred > 0.5).astype(int),
            average="macro",
            zero_division=0
        )

        print(f"Epoch {epoch+1}: "
              f"Train Loss={train_loss:.4f} | "
              f"Val AUC={val_auc:.4f} | "
              f"Val F1={val_f1:.4f}")

        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save(model.state_dict(), f"results/{MODEL_NAME}.pth")
            print("Model saved.")

    print("\nTraining finished.")
    print("Best Val AUC:", best_val_auc)

    # ---------------- FINAL TEST EVALUATION ----------------
    print("\n===== FINAL TEST EVALUATION =====")

    model.load_state_dict(torch.load(f"results/{MODEL_NAME}.pth"))
    model.eval()

    test_preds = []
    test_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = torch.sigmoid(model(images))

            test_preds.append(outputs.cpu().numpy())
            test_labels.append(labels.numpy())

    y_test_pred = np.vstack(test_preds)
    y_test_true = np.vstack(test_labels)


    evaluate_model(
        y_test_true,
        y_test_pred,
        LABELS,
        save_prefix=MODEL_NAME
    )

    print("Test evaluation complete.")
  




if __name__ == "__main__":
    train()
