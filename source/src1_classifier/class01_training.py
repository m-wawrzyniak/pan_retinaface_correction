import time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from source.src0_dataset_creation.ImageDataset import CSVImageDataset
from source.src1_classifier.Classifier import FaceVerifierCNN
from source.src1_classifier import class00_utilities as class00

from config import P02_model_config as P02


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

def train_model(
        train_csv,
        val_csv,
        test_csv,
        input_size,
        batch_size,
        num_workers,
        lr,
        weight_decay,
        reduction_factor,
        patience,
        epochs,
        dec_threshold,
        seed,
        output_dir="output_model",
        use_amp=True
    ):
    class00.set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transforms
    '''
    Introduces heterogeneity in the training set.
    Resize just ensures proper format.
    RandomHorizontalFlip flips like mirror half the images.
    ColorJitter changes the colors a bit.
    Normalize is some normalization used by standard models.
    '''
    transform_train = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    '''
    We don't introduce heterogeneity in the evaluation set.
    '''
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=reduction_factor, patience=patience, verbose=True)

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
        train_conf = class00.compute_confusion(train_y, train_logits, threshold=dec_threshold)
        val_conf = class00.compute_confusion(val_y, val_logits, threshold=dec_threshold)
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

    # save history CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(out_dir / "training_log.csv", index=False)
    print("Training complete. Log saved to:", out_dir / "training_log.csv")

    # final evaluation on test set
    test_loss, test_y, test_logits = eval_epoch(model, test_loader, criterion, device)
    test_conf = class00.compute_confusion(test_y, test_logits, threshold=dec_threshold)
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


if __name__ == "__main__":
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
        input_size=P02.CNN_INPUT_SIZE,
        batch_size=P02.BATCH_SIZE,
        num_workers=P02.NUM_WORKERS,
        lr=P02.LEARNING_RATE,
        weight_decay=P02.OPTIMIZER_WEIGHT_DECAY,
        reduction_factor=P02.SCHEDULER_REDUCTION_FACTOR,
        patience=P02.SCHEDULER_PATIENCE,
        epochs=P02.EPOCHS,
        dec_threshold=P02.PROB_THRESHOLD,
        seed=P02.SEED,
        use_amp=P02.USE_AMP,
    )
