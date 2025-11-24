import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# -------------------------
# Put your model class here
# -------------------------
# Copy the FaceVerifierCNN class you provided earlier here.
# For brevity, I'm reusing the same name — paste your model exactly.
class FaceVerifierCNN(nn.Module):
    def __init__(self, input_size=64):
        super(FaceVerifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout2d(p=0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.drop2 = nn.Dropout2d(p=0.15)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.drop3 = nn.Dropout2d(p=0.2)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.drop4 = nn.Dropout2d(p=0.2)

        final_dim = input_size // 16
        self.flatten_size = 256 * final_dim * final_dim

        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc_bn = nn.BatchNorm1d(512)
        self.fc_drop = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.drop1(x)
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.drop2(x)
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.drop3(x)
        x = self.pool4(torch.relu(self.bn4(self.conv4(x))))
        x = self.drop4(x)
        x = x.view(-1, self.flatten_size)
        x = torch.relu(self.fc_bn(self.fc1(x)))
        x = self.fc_drop(x)
        x = self.fc2(x)
        return x  # logits

# -------------------------
# Dataset that reads CSV with columns: path,is_face
# -------------------------
class CSVImageDataset(Dataset):
    def __init__(self, df, transform=None):
        """
        df: pandas DataFrame with columns ['path', 'is_face']
        transform: torchvision transform applied to PIL image
        """
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(row["path"])
        # load with PIL
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = float(row["is_face"])
        return img, torch.tensor(label, dtype=torch.float32)

# -------------------------
# Utility functions
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_confusion(y_true, y_pred_logits, threshold=0.5):
    """y_true: tensor or array of 0/1, y_pred_logits: logits -> apply sigmoid"""
    probs = torch.sigmoid(torch.tensor(y_pred_logits))
    preds = (probs >= threshold).int()
    y_true = torch.tensor(y_true).int()
    TP = int(((preds == 1) & (y_true == 1)).sum())
    TN = int(((preds == 0) & (y_true == 0)).sum())
    FP = int(((preds == 1) & (y_true == 0)).sum())
    FN = int(((preds == 0) & (y_true == 1)).sum())
    return {"TP": TP, "TN": TN, "FP": FP, "FN": FN}

# -------------------------
# Training / Evaluation loops
# -------------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    running_loss = 0.0
    all_logits = []
    all_labels = []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        if scaler:  # amp
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(labels.detach().cpu())

    epoch_loss = running_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits).squeeze().numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()
    return epoch_loss, all_labels, all_logits

def eval_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).unsqueeze(1)
            logits = model(imgs)
            loss = criterion(logits, labels)
            running_loss += loss.item() * imgs.size(0)
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

    epoch_loss = running_loss / len(loader.dataset)
    all_logits = torch.cat(all_logits).squeeze().numpy()
    all_labels = torch.cat(all_labels).squeeze().numpy()
    return epoch_loss, all_labels, all_logits

# -------------------------
# Main training function
# -------------------------
def train_model(
    train_csv,
    val_csv,
    test_csv,
    output_dir="output_model",
    input_size=64,
    batch_size=64,
    epochs=20,
    lr=1e-3,
    weight_decay=1e-4,
    num_workers=4,
    seed=42,
    use_amp=True,
    save_every=1
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])

    # Load CSVs
    train_df = pd.read_csv(train_csv)
    val_df   = pd.read_csv(val_csv)
    test_df  = pd.read_csv(test_csv)

    train_ds = CSVImageDataset(train_df, transform=transform_train)
    val_ds   = CSVImageDataset(val_df, transform=transform_eval)
    test_ds  = CSVImageDataset(test_df, transform=transform_eval)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    # Model, loss, optimizer
    model = FaceVerifierCNN(input_size=input_size).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, verbose=True)

    scaler = torch.cuda.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, epochs+1):
        t0 = time.time()
        train_loss, train_y, train_logits = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_y, val_logits = eval_epoch(model, val_loader, criterion, device)

        # scheduler step using val loss
        scheduler.step(val_loss)

        # metrics
        train_conf = compute_confusion(train_y, train_logits)
        val_conf = compute_confusion(val_y, val_logits)
        train_acc = (train_conf["TP"] + train_conf["TN"]) / max(1, len(train_ds))
        val_acc = (val_conf["TP"] + val_conf["TN"]) / max(1, len(val_ds))

        t1 = time.time()
        epoch_time = t1 - t0

        print(f"Epoch {epoch}/{epochs} — time: {epoch_time:.1f}s")
        print(f"  train_loss: {train_loss:.4f}  val_loss: {val_loss:.4f}")
        print(f"  train_acc: {train_acc:.4f}  val_acc: {val_acc:.4f}")
        print(f"  train_conf: {train_conf}  val_conf: {val_conf}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            **{f"train_{k}": v for k, v in train_conf.items()},
            **{f"val_{k}": v for k, v in val_conf.items()},
            "lr": optimizer.param_groups[0]["lr"]
        })

        # save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = out_dir / "best_model.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, best_path)
            print("  🔖 Saved best model:", best_path)

        # periodic checkpoint
        if epoch % save_every == 0:
            ckpt_path = out_dir / f"ckpt_epoch{epoch}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss
            }, ckpt_path)

    # save history CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_log.csv", index=False)
    print("Training complete. Log saved to:", out_dir / "training_log.csv")

    # final evaluation on test set
    test_loss, test_y, test_logits = eval_epoch(model, test_loader, criterion, device)
    test_conf = compute_confusion(test_y, test_logits)
    test_acc = (test_conf["TP"] + test_conf["TN"]) / max(1, len(test_ds))
    print("Test loss:", test_loss)
    print("Test conf:", test_conf)
    print("Test acc:", test_acc)

    # save final model
    final_path = out_dir / "final_model.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epochs,
        "test_loss": test_loss
    }, final_path)
    print("Saved final model to:", final_path)

    return model, hist_df, test_conf

# -------------------------
# Example usage (modify paths)
# -------------------------
if __name__ == "__main__":
    # paths to CSVs produced by split_dataset_with_ratios()
    dataset_root = "/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/data/datasets/dummy"
    TRAIN_CSV = f"{dataset_root}/train.csv"
    VAL_CSV   = f"{dataset_root}/val.csv"
    TEST_CSV  = f"{dataset_root}/test.csv"

    model_name = "class_v00"
    model, hist_df, test_conf = train_model(
        train_csv=TRAIN_CSV,
        val_csv=VAL_CSV,
        test_csv=TEST_CSV,
        output_dir=f"/home/mateusz-wawrzyniak/PycharmProjects/pan_retinaface_correction/data/classifiers/{model_name}",
        input_size=64,        # must match P02.CNN_INPUT_SIZE used when saving crops
        batch_size=128,
        epochs=25,
        lr=2e-4,
        weight_decay=1e-4,
        num_workers=6,
        seed=42,
        use_amp=True,
    )
