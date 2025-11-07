import torch
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from EyetrackerCNN import EyetrackerCNN

from config.P01_config import DATASET_DIR

# --- GPU setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- Transformations ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # converts to [0,1]
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet normalization
])

# --- Datasets ---
train_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform=transform)
val_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"), transform=transform)
test_dataset = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"), transform=transform)

# --- Dataloaders ---
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)


model = EyetrackerCNN().to(device)

# --- Loss and optimizer ---
criterion = nn.BCELoss()  # binary cross-entropy
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# --- Training loop ---
def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(inputs)
                preds = (outputs > 0.5).float()
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch + 1}/{epochs} - Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    print("✅ Training complete")
    return model


# --- Run training ---
trained_model = train_model(model, train_loader, val_loader, epochs=10)

# --- Save model ---
torch.save(trained_model.state_dict(), "face_classifier.pth")
print("Model saved to face_classifier.pth")
