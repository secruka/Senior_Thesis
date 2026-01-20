# simple_time_cnn.py
# 1-file baseline: clock image -> 144-class time classification (5-min steps)
# Uses clocks.csv columns: "class index", "filepaths", "data set", "labels"

from pathlib import Path
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# =========================
# Config (EDIT HERE ONLY)
# =========================
DATA_ROOT = Path("/Users/ruka/Senior_Thesis/clock_kaggle")   # contains train/ valid/ test/ and clocks.csv
CSV_PATH  = DATA_ROOT / "clocks.csv"

MODEL = "resnet18"   # "resnet18" or "small"
USE_IMAGENET_PRETRAIN = True   # only for resnet18

IMG_SIZE = 224
NUM_CLASSES = 144

EPOCHS = 20
BATCH_SIZE = 64
LR = 3e-4
NUM_WORKERS = 4

OUT_DIR = Path("./simple_runs")
RUN_NAME = "baseline"
SEED = 42
# =========================


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS for Mac
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ClockCSV(Dataset):
    def __init__(self, df: pd.DataFrame, data_root: Path, transform=None):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel = row["filepaths"]
        y = int(row["class index"])
        img_path = self.data_root / rel
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class SmallCNN(nn.Module):
    """Simple CNN baseline (fast, small; accuracy may be lower than ResNet)."""
    def __init__(self, num_classes=144):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 112
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 56
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 28
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def build_model(model_name: str, num_classes: int):
    if model_name == "small":
        return SmallCNN(num_classes=num_classes)

    if model_name == "resnet18":
        if USE_IMAGENET_PRETRAIN:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            m = models.resnet18(weights=weights)
        else:
            m = models.resnet18(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    raise ValueError(f"Unknown MODEL={model_name}")


def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


def train_one_epoch(model, loader, device, criterion, optimizer):
    model.train()
    total_loss, total_acc, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_loss / n, total_acc / n


def label_to_time_str(label: str) -> str:
    # label example: "1_00"
    h, m = label.split("_")
    return f"{int(h)}:{int(m):02d}"


@torch.no_grad()
def predict_image(model, img_path: Path, device, transform, class_to_label: dict, topk=5):
    model.eval()
    img = Image.open(img_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    logits = model(x)
    prob = torch.softmax(logits, dim=1)[0]
    vals, idxs = torch.topk(prob, k=topk)
    results = []
    for v, i in zip(vals.tolist(), idxs.tolist()):
        lab = class_to_label.get(i, str(i))
        results.append((i, v, lab, label_to_time_str(lab) if "_" in lab else lab))
    return results


def main():
    set_seed(SEED)
    device = get_device()
    print("device:", device)

    df = pd.read_csv(CSV_PATH)
    # Normalize column names just in case
    # (your CSV has: class index,filepaths,labels,data set)
    assert "class index" in df.columns
    assert "filepaths" in df.columns
    assert "data set" in df.columns
    assert "labels" in df.columns

    # class index -> label string (e.g., 0 -> "1_00")
    class_to_label = df.groupby("class index")["labels"].first().to_dict()

    df_train = df[df["data set"] == "train"].copy()
    df_valid = df[df["data set"] == "valid"].copy()
    df_test  = df[df["data set"] == "test"].copy()

    # Transform (NO rotation/flip; those break the label for clocks)
    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std  = (0.229, 0.224, 0.225)

    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
        transforms.RandomErasing(p=0.25),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(imagenet_mean, imagenet_std),
    ])

    train_ds = ClockCSV(df_train, DATA_ROOT, transform=train_tf)
    valid_ds = ClockCSV(df_valid, DATA_ROOT, transform=test_tf)
    test_ds  = ClockCSV(df_test,  DATA_ROOT, transform=test_tf)

    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=(device.type != "cpu"))
    valid_ld = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type != "cpu"))
    test_ld  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=(device.type != "cpu"))

    model = build_model(MODEL, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = OUT_DIR / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pt"

    best_val_acc = -1.0
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_ld, device, criterion, optimizer)
        va_loss, va_acc = evaluate(model, valid_ld, device, criterion)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} "
              f"| valid loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "model_name": MODEL,
                        "num_classes": NUM_CLASSES,
                        "class_to_label": class_to_label}, best_path)

    print("best valid acc:", best_val_acc)

    # Test
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    te_loss, te_acc = evaluate(model, test_ld, device, criterion)
    print(f"TEST | loss {te_loss:.4f} acc {te_acc:.4f}")

    # Single-image demo (edit if you want)
    demo_img = DATA_ROOT / "train/1-00/0.jpg"
    if demo_img.exists():
        preds = predict_image(model, demo_img, device, test_tf, class_to_label, topk=5)
        print("\nDemo:", demo_img)
        for i, p, lab, t in preds:
            print(f"  class {i:3d}  prob {p:.3f}  label {lab}  time {t}")


if __name__ == "__main__":
    main()
