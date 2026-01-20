from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision import transforms

try:
    from torchvision.models import (
        resnet18,
        resnet34,
        resnet50,
        ResNet18_Weights,
        ResNet34_Weights,
        ResNet50_Weights,
    )
    _HAS_WEIGHTS_API = True
except Exception:
    # Older torchvision
    from torchvision.models import resnet18, resnet34, resnet50

    _HAS_WEIGHTS_API = False

from .dataset import ClockTimeDataset


def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer_mps: bool = True) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        if _HAS_WEIGHTS_API:
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
        else:
            model = resnet18(pretrained=pretrained)
    elif model_name == "resnet34":
        if _HAS_WEIGHTS_API:
            weights = ResNet34_Weights.DEFAULT if pretrained else None
            model = resnet34(weights=weights)
        else:
            model = resnet34(pretrained=pretrained)
    elif model_name == "resnet50":
        if _HAS_WEIGHTS_API:
            weights = ResNet50_Weights.DEFAULT if pretrained else None
            model = resnet50(weights=weights)
        else:
            model = resnet50(pretrained=pretrained)
    else:
        raise ValueError("--model must be one of: resnet18, resnet34, resnet50")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)

    return loss_sum / max(total, 1), correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True, help="e.g., /Users/ruka/Senior_Thesis/clock_kaggle")
    ap.add_argument("--csv_path", type=str, required=True, help="path to clocks.csv")

    ap.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "resnet34", "resnet50"])
    ap.add_argument("--pretrained", action="store_true", help="use ImageNet pretrained backbone")

    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--run_name", type=str, default="", help="optional name; otherwise auto")
    ap.add_argument("--prefer_mps", action="store_true", help="prefer Apple MPS when CUDA not available")
    ap.add_argument("--amp", action="store_true", help="use automatic mixed precision (CUDA only)")

    args = ap.parse_args()

    seed_everything(args.seed)
    device = get_device(prefer_mps=args.prefer_mps)

    run_name = args.run_name.strip() or f"timecnn_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir = Path("runs") / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transforms: avoid flips/rotations because they change the time label.
    train_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.10), ratio=(0.3, 3.3), value="random"),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = ClockTimeDataset(args.csv_path, args.data_root, split="train", transform=train_tf)
    val_ds = ClockTimeDataset(args.csv_path, args.data_root, split="valid", transform=eval_tf)
    test_ds = ClockTimeDataset(args.csv_path, args.data_root, split="test", transform=eval_tf)

    num_classes = len(train_ds.class_map.idx_to_label)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    model = build_model(args.model, num_classes=num_classes, pretrained=args.pretrained)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    best_val_acc = -1.0

    # Save class map once
    class_map_path = out_dir / "class_map.json"
    with open(class_map_path, "w", encoding="utf-8") as f:
        json.dump({"idx_to_label": train_ds.class_map.idx_to_label}, f, ensure_ascii=False, indent=2)

    # Save config
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=2)

    log_path = out_dir / "log.csv"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,train_loss,train_acc,val_loss,val_acc,lr\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda" and scaler.is_enabled():
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += x.size(0)

            pbar.set_postfix({"loss": running_loss / max(total, 1), "acc": correct / max(total, 1)})

        scheduler.step()

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, device)
        lr = scheduler.get_last_lr()[0]

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch},{train_loss:.6f},{train_acc:.6f},{val_loss:.6f},{val_acc:.6f},{lr:.8f}\n")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt = {
                "model": args.model,
                "num_classes": num_classes,
                "state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "class_map": {"idx_to_label": train_ds.class_map.idx_to_label},
                "normalization": {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
            }
            torch.save(ckpt, out_dir / "best.pt")

    # Final test with best checkpoint
    best = torch.load(out_dir / "best.pt", map_location=device)
    model.load_state_dict(best["state_dict"])
    test_loss, test_acc = evaluate(model, test_loader, device)

    summary = {
        "best_val_acc": float(best_val_acc),
        "test_loss": float(test_loss),
        "test_acc": float(test_acc),
        "device": str(device),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Done ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
