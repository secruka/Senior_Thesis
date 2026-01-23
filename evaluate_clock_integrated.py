#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""evaluate_clock_integrated.py

統合モデル（dial + hands + clock rules）を rotation.csv の test split で評価する。

- train_clock_integrated.py が保存する重み（dial/hour/minute_{TAG}.pth）を読み込み
- dial/hour_img/minute_img の確率から、Prolog(clock_integrated.pl)と同等の規則で
  P(time=H:M) を計算（4つの回転に対する和）
- top-1 / top-3 の time accuracy, hour accuracy, minute accuracy, dial accuracy などを出力

使い方例:
  python evaluate_clock_integrated.py \
    --data_root /Users/ruka/Senior_Thesis/clock_kaggle \
    --csv rotation.csv \
    --weights_dir weights \
    --tag after_time \
    --subset test \
    --batch_size 32

注意:
- rotation.csv の file 列が "train/..." "test/..." のように prefix で split されている前提。
- 回転の符号規約が逆の場合は、このスクリプトの mapping を Prolog と同じように直す必要あり。
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, List, Tuple

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ------------------------------
# Utils
# ------------------------------

def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rotation_deg_to_cls(rotation_deg: float) -> int:
    d = int(round(float(rotation_deg))) % 360
    return (d // 90) % 4


def angle_deg_clockwise_from_12(dx: float, dy: float) -> float:
    # 画像座標: x→右, y→下
    # 12時方向を0°、時計回り
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def angle_cls_12(hand_x: float, hand_y: float, cx: float, cy: float) -> int:
    dx = float(hand_x) - float(cx)
    dy = float(hand_y) - float(cy)
    deg = angle_deg_clockwise_from_12(dx, dy)
    return int((deg + 15.0) // 30.0) % 12


def hour_to_idx(hour: int) -> int:
    # 12 -> 0, 1..11 -> 1..11
    return int(hour) 


def idx_to_hour(idx: int) -> int:
    return int(idx)


def correct_idx(image_idx: int, steps: int) -> int:
    # CanonIdx = (ImageIdx - Steps) mod 12
    return (int(image_idx) - int(steps) + 120) % 12


# rot_cls -> 12-step offset
ROT_STEPS = {0: 0, 1: 3, 2: 6, 3: 9}


# ------------------------------
# Net
# ------------------------------

class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_f, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(next(self.parameters()).device)
        return self.backbone(x)


# ------------------------------
# Dataset
# ------------------------------

@dataclass
class RowLabels:
    hour_time: int
    minute_time: int
    rot_cls: int
    h_img_cls: int
    m_img_cls: int


def collate_fn_rowlabels(batch):
    """Custom collate function for batches containing RowLabels dataclass."""
    imgs = []
    labels_list = []
    rel_paths = []
    
    for img, labels, rel_path in batch:
        imgs.append(img)
        labels_list.append(labels)
        rel_paths.append(rel_path)
    
    # Stack images into a batch tensor
    imgs = torch.stack(imgs)
    
    # Convert labels dataclass to a list of dataclasses (keep as-is for unpacking later)
    return imgs, labels_list, rel_paths


class RotationCsvTorchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root: str,
        csv_path: str,
        subset: str,
        transform,
        max_samples: Optional[int] = None,
    ):
        self.data_root = data_root
        self.csv_path = csv_path
        self.subset = subset
        self.transform = transform

        df = pd.read_csv(csv_path)

        required = [
            "file",
            "time_h",
            "time_m",
            "rotation_deg",
            "status",
            "cx",
            "cy",
            "minute_x",
            "minute_y",
            "hour_x",
            "hour_y",
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError("rotation.csv missing columns: " + ", ".join(missing))

        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()
        prefix = subset.strip().lower() + "/"
        df = df[df["file"].astype(str).str.startswith(prefix)].copy()

        num_cols = [
            "time_h",
            "time_m",
            "rotation_deg",
            "cx",
            "cy",
            "minute_x",
            "minute_y",
            "hour_x",
            "hour_y",
        ]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["file"] + num_cols).reset_index(drop=True)

        df = df[(df["time_h"] >= 1) & (df["time_h"] <= 12)].copy()
        df = df[(df["time_m"] >= 0) & (df["time_m"] <= 59)].copy()
        df = df[df["time_m"] % 5 == 0].copy()
        df = df.reset_index(drop=True)

        if max_samples is not None:
            df = df.iloc[: int(max_samples)].reset_index(drop=True)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = str(row["file"])
        img_path = os.path.join(self.data_root, rel_path)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        hour_time = int(row["time_h"])
        minute_time = int(row["time_m"])
        rot_cls = rotation_deg_to_cls(row["rotation_deg"])

        cx, cy = float(row["cx"]), float(row["cy"])
        mx, my = float(row["minute_x"]), float(row["minute_y"])
        hx, hy = float(row["hour_x"]), float(row["hour_y"])

        m_img_cls = angle_cls_12(mx, my, cx, cy)
        h_img_cls = angle_cls_12(hx, hy, cx, cy)

        labels = RowLabels(
            hour_time=hour_time,
            minute_time=minute_time,
            rot_cls=rot_cls,
            h_img_cls=h_img_cls,
            m_img_cls=m_img_cls,
        )
        return img, labels, rel_path


# ------------------------------
# Time distribution computation (equivalent to clock_integrated.pl)
# ------------------------------

def build_time_index() -> Tuple[torch.Tensor, List[Tuple[int, int, int]]]:
    """Return (time_meta_tensor, time_list)

    time_list[t] = (Hour, MIdx, HPos)
      - Hour: 1..12
      - MIdx: 0..11 (Minute=5*MIdx)
      - HPos: 0..11 (canonical hour hand position index)

    HPos rule (matching clock_integrated.pl):
      if MIdx < 6:  hour_time = HPos
      else:         hour_time = prev(HPos)
      => HPos = hour_idx + (MIdx>=6 ? 1 : 0)  (mod 12)
    """
    time_list: List[Tuple[int, int, int]] = []
    for hour in range(1, 13):
        h_idx = hour_to_idx(hour)
        for midx in range(12):
            hpos = (h_idx + (1 if midx >= 6 else 0)) % 12
            time_list.append((hour, midx, hpos))

    # meta as tensor for convenience
    meta = torch.tensor(time_list, dtype=torch.long)  # [T,3]
    return meta, time_list


def precompute_latent_indices(time_meta: torch.Tensor, device: torch.device):
    """Precompute, for each rotation r, the required HImgIndex and MImgIndex per time t.

    Mapping (from Prolog):
      HPos = (HImg - Steps) mod 12  => HImg = (HPos + Steps) mod 12
      MIdx = (MImg - Steps) mod 12  => MImg = (MIdx + Steps) mod 12
    """
    T = time_meta.shape[0]
    # time_meta[t] = [Hour, MIdx, HPos]
    midx = time_meta[:, 1]
    hpos = time_meta[:, 2]

    h_img_idx = []
    m_img_idx = []
    for r in range(4):
        steps = ROT_STEPS[r]
        h_img_idx.append(((hpos + steps) % 12).to(device))
        m_img_idx.append(((midx + steps) % 12).to(device))

    # each is list of 4 tensors [T]
    return h_img_idx, m_img_idx


@torch.no_grad()
def time_distribution(
    p_dial: torch.Tensor,   # [B,4]
    p_hour: torch.Tensor,   # [B,12]  (hour_img)
    p_min: torch.Tensor,    # [B,12]  (minute_img)
    h_img_idx: List[torch.Tensor],
    m_img_idx: List[torch.Tensor],
) -> torch.Tensor:
    """Compute P(time) for each sample.

    P(time=t) = sum_{r} P(dial=r) * P(hour_img=HImg(r,t)) * P(min_img=MImg(r,t))

    returns: [B,T]
    """
    B = p_dial.shape[0]
    T = h_img_idx[0].shape[0]
    out = torch.zeros((B, T), device=p_dial.device)

    for r in range(4):
        # gather probs for required latent values
        h_idx = h_img_idx[r].view(1, T).expand(B, T)
        m_idx = m_img_idx[r].view(1, T).expand(B, T)
        ph = torch.gather(p_hour, 1, h_idx)
        pm = torch.gather(p_min, 1, m_idx)
        out += p_dial[:, r].view(B, 1) * ph * pm

    return out


def decode_time_argmax(time_meta: torch.Tensor, time_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # time_meta[t] = [Hour, MIdx, HPos]
    t_hat = torch.argmax(time_prob, dim=1)  # [B]
    hour_hat = time_meta[t_hat, 0]
    min_hat = time_meta[t_hat, 1] * 5
    return hour_hat, min_hat


# ------------------------------
# Metrics
# ------------------------------

@torch.no_grad()
def topk_time_hits(time_meta: torch.Tensor, time_prob: torch.Tensor, hour_gt: torch.Tensor, min_gt: torch.Tensor, k: int) -> torch.Tensor:
    """Return [B] boolean hit if (hour_gt,min_gt) is in top-k predicted times."""
    topk = torch.topk(time_prob, k=k, dim=1).indices  # [B,k]
    hours = time_meta[topk, 0]  # [B,k]
    mins = time_meta[topk, 1] * 5
    hit = (hours == hour_gt.view(-1, 1)) & (mins == min_gt.view(-1, 1))
    return hit.any(dim=1)


# ------------------------------
# Main
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--csv", type=str, default="rotations.csv")
    p.add_argument("--subset", type=str, default="test", choices=["train", "valid", "test"])
    p.add_argument("--weights_dir", type=str, default="weights")
    p.add_argument("--tag", type=str, default="after_time", help="dial_{tag}.pth の tag")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print_examples", type=int, default=5)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.data_root, csv_path)

    # transforms (match training)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    ds = RotationCsvTorchDataset(
        data_root=args.data_root,
        csv_path=csv_path,
        subset=args.subset,
        transform=tfm,
        max_samples=args.max_samples,
    )

    if len(ds) == 0:
        raise RuntimeError(
            f"No rows found for subset='{args.subset}'. rotation.csv の file 列の prefix を確認して下さい。"
        )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_fn_rowlabels,
    )

    # load nets
    dial_w = os.path.join(args.weights_dir, f"dial_{args.tag}.pth")
    hour_w = os.path.join(args.weights_dir, f"hour_{args.tag}.pth")
    min_w = os.path.join(args.weights_dir, f"minute_{args.tag}.pth")

    for pth in [dial_w, hour_w, min_w]:
        if not os.path.exists(pth):
            raise FileNotFoundError(f"weights not found: {pth}")

    net_dial = ResNetClassifier(4).to(device).eval()
    net_hour = ResNetClassifier(12).to(device).eval()
    net_min = ResNetClassifier(12).to(device).eval()

    net_dial.load_state_dict(torch.load(dial_w, map_location=device))
    net_hour.load_state_dict(torch.load(hour_w, map_location=device))
    net_min.load_state_dict(torch.load(min_w, map_location=device))

    # precompute time mapping
    time_meta, time_list = build_time_index()  # [T,3]
    time_meta = time_meta.to(device)
    h_img_idx, m_img_idx = precompute_latent_indices(time_meta, device)

    # metrics accumulators
    n = 0
    dial_ok = 0
    hourimg_ok = 0
    minimg_ok = 0

    time_ok = 0
    hour_ok = 0
    min_ok = 0

    top3_ok = 0

    printed = 0

    for imgs, labels_list, rel_paths in loader:
        imgs = imgs.to(device)

        # gt tensors - unpack from labels_list
        hour_gt = torch.tensor([lab.hour_time for lab in labels_list], device=device, dtype=torch.long)
        min_gt = torch.tensor([lab.minute_time for lab in labels_list], device=device, dtype=torch.long)
        rot_gt = torch.tensor([lab.rot_cls for lab in labels_list], device=device, dtype=torch.long)
        himg_gt = torch.tensor([lab.h_img_cls for lab in labels_list], device=device, dtype=torch.long)
        mimg_gt = torch.tensor([lab.m_img_cls for lab in labels_list], device=device, dtype=torch.long)

        p_dial = net_dial(imgs)   # [B,4]
        p_hour = net_hour(imgs)   # [B,12]
        p_min = net_min(imgs)     # [B,12]

        dial_pred = torch.argmax(p_dial, dim=1)
        himg_pred = torch.argmax(p_hour, dim=1)
        mimg_pred = torch.argmax(p_min, dim=1)
        
        if printed < 5:
            for j in range(min(5, imgs.shape[0])):
                print("DBG", rel_paths[j],
                "himg_gt", int(himg_gt[j]),
                "himg_pred", int(himg_pred[j]),
                "mimg_gt", int(mimg_gt[j]),
                "mimg_pred", int(mimg_pred[j]))

        dial_ok += int((dial_pred == rot_gt).sum().item())
        hourimg_ok += int((himg_pred == himg_gt).sum().item())
        minimg_ok += int((mimg_pred == mimg_gt).sum().item())

        # integrated time
        tprob = time_distribution(p_dial, p_hour, p_min, h_img_idx, m_img_idx)  # [B,T]
        hour_hat, min_hat = decode_time_argmax(time_meta, tprob)

        time_ok += int(((hour_hat == hour_gt) & (min_hat == min_gt)).sum().item())
        hour_ok += int((hour_hat == hour_gt).sum().item())
        min_ok += int((min_hat == min_gt).sum().item())

        top3_hit = topk_time_hits(time_meta, tprob, hour_gt, min_gt, k=3)
        top3_ok += int(top3_hit.sum().item())

        # print a few examples
        if printed < args.print_examples:
            B = imgs.shape[0]
            for j in range(B):
                if printed >= args.print_examples:
                    break
                # show top-5 times
                top5 = torch.topk(tprob[j], k=5).indices.detach().cpu().tolist()
                pred_str = ", ".join([f"{time_list[t][0]}:{time_list[t][1]*5:02d}" for t in top5])
                print(
                    f"[{printed}] {rel_paths[j]} | GT={int(hour_gt[j])}:{int(min_gt[j]):02d} | "
                    f"Pred={int(hour_hat[j])}:{int(min_hat[j]):02d} | top5={pred_str}"
                )
                printed += 1

        n += imgs.shape[0]

    def pct(x):
        return 100.0 * x / max(1, n)

    print("\n=== Results ===")
    print(f"N = {n}")
    print(f"dial acc        : {pct(dial_ok):.2f}%")
    print(f"hour_img acc    : {pct(hourimg_ok):.2f}%")
    print(f"minute_img acc  : {pct(minimg_ok):.2f}%")
    print("--- integrated time ---")
    print(f"time (exact) acc: {pct(time_ok):.2f}%")
    print(f"hour acc        : {pct(hour_ok):.2f}%")
    print(f"minute acc      : {pct(min_ok):.2f}%")
    print(f"top-3 time acc  : {pct(top3_ok):.2f}%")


if __name__ == "__main__":
    main()

#  python evaluate_clock_integrated.py --data_root clock_kaggle --csv rotations.csv --weights_dir weights --tag after_time --subset train --max_samples 100
# Using device: cuda
# [0] train/1-00/1.jpg | GT=1:00 | Pred=12:00 | top5=12:00, 3:15, 8:45, 1:00, 2:00
# [1] train/1-00/12.jpg | GT=1:00 | Pred=3:15 | top5=3:15, 8:45, 4:15, 5:30, 5:15
# [2] train/1-00/13.jpg | GT=1:00 | Pred=8:45 | top5=8:45, 3:15, 12:00, 9:45, 10:45
# [3] train/1-00/15.jpg | GT=1:00 | Pred=2:00 | top5=2:00, 9:00, 11:00, 12:00, 4:00
# [4] train/1-00/16.jpg | GT=1:00 | Pred=9:00 | top5=9:00, 2:00, 11:00, 8:00, 3:00

# === Results ===
# N = 100
# dial acc        : 68.00%
# hour_img acc    : 0.00%
# minute_img acc  : 63.00%
# --- integrated time ---
# time (exact) acc: 0.00%
# hour acc        : 4.00%
# minute acc      : 41.00%
# top-3 time acc  : 0.00%