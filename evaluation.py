import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)

def evaluate_model(y_true, y_pred, labels, save_prefix="model"):

    # ---------------------------
    # Threshold tuning per class
    # ---------------------------
    best_thresholds = []

    for i in range(len(labels)):
        best_f1 = 0
        best_t = 0.5

        for t in np.arange(0.1, 0.9, 0.05):
            preds = (y_pred[:, i] > t).astype(int)
            f1 = f1_score(y_true[:, i], preds, zero_division=0)

            if f1 > best_f1:
                best_f1 = f1
                best_t = t

        best_thresholds.append(best_t)

    best_thresholds = np.array(best_thresholds)
    y_bin = (y_pred > best_thresholds).astype(int)

    # ---------------------------
    # Global metrics
    # ---------------------------
    macro_auc = roc_auc_score(y_true, y_pred, average="macro")
    micro_auc = roc_auc_score(y_true, y_pred, average="micro")

    macro_f1 = f1_score(y_true, y_bin, average="macro", zero_division=0)
    micro_f1 = f1_score(y_true, y_bin, average="micro", zero_division=0)

    macro_precision = precision_score(y_true, y_bin, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_bin, average="macro", zero_division=0)

    print("\n=== GLOBAL METRICS ===")
    print("Macro AUC:", macro_auc)
    print("Micro AUC:", micro_auc)
    print("Macro F1:", macro_f1)
    print("Micro F1:", micro_f1)
    print("Macro Precision:", macro_precision)
    print("Macro Recall:", macro_recall)

    # ---------------------------
    # Per-class metrics
    # ---------------------------
    per_class_auc = roc_auc_score(y_true, y_pred, average=None)
    per_class_f1 = f1_score(y_true, y_bin, average=None, zero_division=0)

    results_df = pd.DataFrame({
        "Class": labels,
        "AUC": per_class_auc,
        "F1": per_class_f1,
        "Threshold": best_thresholds
    })

    print("\n=== PER CLASS METRICS ===")
    print(results_df)

    results_df.to_csv(f"{save_prefix}_per_class_metrics.csv", index=False)

    # ---------------------------
    # ROC Curves
    # ---------------------------
    plt.figure(figsize=(10, 8))

    for i in range(len(labels)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_pred[:, i])
        plt.plot(fpr, tpr, label=f"{labels[i]} (AUC={per_class_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Per Class")
    plt.legend(loc="lower right")
    plt.savefig(f"{save_prefix}_roc.png")
    plt.close()

    print("\nROC curves saved.")
