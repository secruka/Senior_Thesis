from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image
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
    from torchvision.models import resnet18, resnet34, resnet50

    _HAS_WEIGHTS_API = False


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "resnet18":
        model = resnet18(weights=None) if _HAS_WEIGHTS_API else resnet18(pretrained=False)
    elif model_name == "resnet34":
        model = resnet34(weights=None) if _HAS_WEIGHTS_API else resnet34(pretrained=False)
    elif model_name == "resnet50":
        model = resnet50(weights=None) if _HAS_WEIGHTS_API else resnet50(pretrained=False)
    else:
        raise ValueError("Unsupported model")

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def label_to_timestr(label: str) -> str:
    h_str, m_str = label.split("_")
    return f"{int(h_str):02d}:{int(m_str):02d}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--img", type=str, required=True)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--prefer_mps", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model_name = ckpt.get("model", "resnet18")
    num_classes = ckpt["num_classes"]
    idx_to_label = ckpt["class_map"]["idx_to_label"]

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.prefer_mps and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    model = build_model(model_name, num_classes=num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    mean = ckpt.get("normalization", {}).get("mean", [0.485, 0.456, 0.406])
    std = ckpt.get("normalization", {}).get("std", [0.229, 0.224, 0.225])

    tf = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    img = Image.open(args.img).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze(0)

    topk = min(args.topk, num_classes)
    vals, idxs = torch.topk(probs, k=topk)

    print("\nImage:", args.img)
    print("Predictions:")
    for p, idx in zip(vals.tolist(), idxs.tolist()):
        label = idx_to_label[idx]
        print(f"  {idx:3d}  {label_to_timestr(label)}  (label={label})  p={p:.4f}")


if __name__ == "__main__":
    main()
