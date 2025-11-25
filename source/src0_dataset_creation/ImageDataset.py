from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torch

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
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = float(row["is_face"])
        return img, torch.tensor(label, dtype=torch.float32)