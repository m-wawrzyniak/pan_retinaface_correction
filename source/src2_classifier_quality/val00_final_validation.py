import torch
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import pandas as pd
from torchvision import transforms

from source.src0_dataset_creation.ImageDataset import CSVImageDataset
from source.src1_classifier.Classifier import FaceVerifierCNN

import config.P02_model_config as P02

def load_model(model_path, input_size):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FaceVerifierCNN(input_size=input_size).to(device)

    checkpoint = torch.load(model_path, map_location=device)

    # Your training script saves "model_state_dict"
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"model_state_dict missing in checkpoint! Keys: {checkpoint.keys()}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, device


def final_validate(model, val_loader, device, prob_threshold=None):
    """
    Validate a binary classifier (face / non-face) with full metrics and plots.

    Parameters
    ----------
    model : torch.nn.Module
        Trained model
    val_loader : DataLoader
        Validation dataset loader
    device : torch.device
    prob_threshold : float, optional
        Threshold for converting probabilities to class predictions.
        Defaults to P02.PROB_THRESHOLD
    """
    if prob_threshold is None:
        prob_threshold = P02.PROB_THRESHOLD

    model.eval()
    all_logits = []
    all_labels = []

    # Collect logits and labels
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).float()  # ensure float
            logits = model(imgs).view(-1)       # flatten to [B]
            all_logits.append(logits.cpu())
            all_labels.append(labels.view(-1).cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()

    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-all_logits))  # sigmoid
    preds = (probs >= prob_threshold).astype(int)

    # Compute metrics
    accuracy = accuracy_score(all_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, preds, average="binary")
    roc_auc = roc_auc_score(all_labels, probs)
    cm = confusion_matrix(all_labels, preds)

    # ROC and Precision-Recall curves
    fpr, tpr, roc_thresholds = roc_curve(all_labels, probs)
    precision_curve, recall_curve, _ = precision_recall_curve(all_labels, probs)

    # Best threshold via Youden's J statistic
    J = tpr - fpr
    best_thr = roc_thresholds[np.argmax(J)]

    # Print metrics
    print("========== FINAL VALIDATION ==========")
    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1 Score:       {f1:.4f}")
    print(f"ROC-AUC:        {roc_auc:.4f}")
    print(f"Best Threshold: {best_thr:.4f}")
    print("Confusion Matrix:\n", cm)

    # -------------------------------
    # PLOTS
    # -------------------------------

    # ROC Curve
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0,1], [0,1], '--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Precision-Recall Curve
    plt.figure(figsize=(6,5))
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.grid(True)
    plt.show()

    # Confusion Matrix Heatmap
    plt.figure(figsize=(5,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

    # Logit distributions
    plt.figure(figsize=(6,5))
    plt.hist(all_logits[all_labels==1], bins=50, alpha=0.6, label="Face")
    plt.hist(all_logits[all_labels==0], bins=50, alpha=0.6, label="Non-Face")
    plt.axvline(best_thr, color="red", linestyle="--", label=f"Best thr = {best_thr:.3f}")
    plt.xlabel("Logit")
    plt.ylabel("Count")
    plt.title("Logit Score Distribution")
    plt.legend()
    plt.grid(True)
    plt.show()

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "best_threshold": best_thr,
        "confusion_matrix": cm
    }


# ------------------------------------------------------------
# USAGE EXAMPLE
# ------------------------------------------------------------

if __name__ == "__main__":
    # Load model
    model, device = load_model(model_path="/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/data/classifiers/class_v00/best_model.pth", input_size=P02.CNN_INPUT_SIZE)
    model.eval()

    # Load validation DataLoader
    dataset_root = "/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/data/datasets/dummy"
    VAL_CSV   = f"{dataset_root}/val.csv"
    transform_eval = transforms.Compose([
        transforms.Resize((P02.CNN_INPUT_SIZE, P02.CNN_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    val_df = pd.read_csv(VAL_CSV)
    val_ds = CSVImageDataset(val_df, transform=transform_eval)
    val_loader = DataLoader(val_ds, batch_size=P02.BATCH_SIZE, shuffle=False,
                            num_workers=P02.NUM_WORKERS, pin_memory=True)

    results = final_validate(model, val_loader, device)
