import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# -------------------------
# label utils (train_clock_integrated.py と同等の意図)
# -------------------------
def rotation_deg_to_cls(rotation_deg: float) -> int:
    """0/90/180/270 -> 0..3 にする（近い90度に丸め）"""
    deg = float(rotation_deg) % 360.0
    # 0,90,180,270 に近いものへ
    cls = int(np.round(deg / 90.0)) % 4
    return cls

def angle_cls_12(hand_x: float, hand_y: float, cx: float, cy: float) -> int:
    """
    画像座標系（y下向き）で、
    - 12時方向を cls=0
    - 時計回りに cls が増える
    となるように 0..11 を返す（30度刻み最近傍）
    """
    dx = float(hand_x) - float(cx)
    dy = float(hand_y) - float(cy)

    # 12時を0度にしたいので、(dx,dy) を「上」を基準に角度化する
    # atan2 は (y, x) なので、ここでは "上" を基準にするために工夫する
    # 上方向ベクトルは (0, -1)。時計回りを正にしたい。
    angle = np.degrees(np.arctan2(dx, -dy))  # ここが肝：上=0, 右=90, 下=180, 左=270
    angle = (angle + 360.0) % 360.0

    cls = int(np.floor((angle + 15.0) / 30.0)) % 12  # 30度刻み最近傍
    return cls


# -------------------------
# Dataset: rotations.csv を使う（統合版の想定）
# -------------------------
class RotationCsvDataset(Dataset):
    def __init__(self, data_root: str, csv_path: str, split: str, task: str):
        """
        split: train / test / valid（file prefixで判定）
        task : dial / hands
        """
        self.data_root = data_root
        self.csv_path = csv_path
        self.split = split
        self.task = task

        df = pd.read_csv(csv_path)

        # 필터: status
        if "status" in df.columns:
            df = df[df["status"].astype(str) == "ok"].copy()

        # 必須列チェック（最低限）
        need_cols = ["file", "cx", "cy", "minute_x", "minute_y", "hour_x", "hour_y", "rotation_deg"]
        for c in need_cols:
            if c not in df.columns:
                raise ValueError(f"CSVに必要な列がありません: {c}")

        # 数値化してNaN落とし
        num_cols = ["cx","cy","minute_x","minute_y","hour_x","hour_y","rotation_deg"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=num_cols).copy()

        # 座標が負なら除外
        df = df[(df["minute_x"] >= 0) & (df["minute_y"] >= 0) & (df["hour_x"] >= 0) & (df["hour_y"] >= 0)].copy()

        # split（file prefix）
        prefix = f"{split}/"
        df = df[df["file"].astype(str).str.startswith(prefix)].copy()

        if len(df) == 0:
            raise ValueError(f"{split} に該当する行が0件です。fileのprefix '{prefix}' を確認してください。")

        self.df = df.reset_index(drop=True)

        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=ResNet18_Weights.IMAGENET1K_V1.transforms().mean,
                                 std=ResNet18_Weights.IMAGENET1K_V1.transforms().std),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img_path = os.path.join(self.data_root, r["file"])
        img = Image.open(img_path).convert("RGB")
        x = self.tf(img)

        cx, cy = r["cx"], r["cy"]
        mx, my = r["minute_x"], r["minute_y"]
        hx, hy = r["hour_x"], r["hour_y"]
        rot_deg = r["rotation_deg"]

        if self.task == "dial":
            y = rotation_deg_to_cls(rot_deg)
            return x, y, r["file"]
        elif self.task == "hands":
            h_cls = angle_cls_12(hx, hy, cx, cy)
            m_cls = angle_cls_12(mx, my, cx, cy)
            return x, h_cls, m_cls, r["file"]
        else:
            raise ValueError("task must be dial or hands")


# -------------------------
# Model: ResNet18 classifier（統合版と同系統）
# -------------------------
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_f, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.backbone(x)


@torch.no_grad()
def eval_dial(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    cm = np.zeros((4, 4), dtype=int)

    for x, y, _fname in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            cm[int(t), int(p)] += 1

    acc = correct / max(total, 1)
    return acc, cm


@torch.no_grad()
def eval_hands(hour_model, minute_model, loader, device):
    hour_model.eval()
    minute_model.eval()

    h_correct = 0
    m_correct = 0
    joint_correct = 0
    total = 0

    for x, h_y, m_y, _fname in loader:
        x = x.to(device)
        h_y = h_y.to(device)
        m_y = m_y.to(device)

        h_pred = torch.argmax(hour_model(x), dim=1)
        m_pred = torch.argmax(minute_model(x), dim=1)

        h_ok = (h_pred == h_y)
        m_ok = (m_pred == m_y)

        h_correct += h_ok.sum().item()
        m_correct += m_ok.sum().item()
        joint_correct += (h_ok & m_ok).sum().item()
        total += h_y.numel()

    return h_correct/total, m_correct/total, joint_correct/total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--csv", default="rotations.csv")
    ap.add_argument("--split", default="test", choices=["train","test","valid"])
    ap.add_argument("--dial_pth", default=None)
    ap.add_argument("--hour_pth", default=None)
    ap.add_argument("--minute_pth", default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    csv_path = args.csv if os.path.isabs(args.csv) else os.path.join(args.data_root, args.csv)

    # dial
    if args.dial_pth:
        dial_ds = RotationCsvDataset(args.data_root, csv_path, args.split, task="dial")
        dial_ld = DataLoader(dial_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        dial_model = ResNetClassifier(num_classes=4).to(device)
        dial_model.load_state_dict(torch.load(args.dial_pth, map_location=device))
        acc, cm = eval_dial(dial_model, dial_ld, device)
        print(f"[DIAL] split={args.split}  acc={acc:.4f}  (random=0.25)")
        print("[DIAL] confusion matrix (true row, pred col):")
        print(cm)

    # hands
    if args.hour_pth and args.minute_pth:
        hands_ds = RotationCsvDataset(args.data_root, csv_path, args.split, task="hands")
        hands_ld = DataLoader(hands_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        hour_model = ResNetClassifier(num_classes=12).to(device)
        minute_model = ResNetClassifier(num_classes=12).to(device)
        hour_model.load_state_dict(torch.load(args.hour_pth, map_location=device))
        minute_model.load_state_dict(torch.load(args.minute_pth, map_location=device))
        h_acc, m_acc, joint = eval_hands(hour_model, minute_model, hands_ld, device)
        print(f"[HANDS] split={args.split}  hour_acc={h_acc:.4f}  minute_acc={m_acc:.4f}  joint={joint:.4f}  (random=0.0833)")

    if (not args.dial_pth) and (not (args.hour_pth and args.minute_pth)):
        print("dial_pth か hour_pth+minute_pth を指定してください。")

if __name__ == "__main__":
    main()

# PS C:\Users\311\Downloads\Senior_Thesis> python evalu.py --data_root clock_kaggle --csv rotations.csv --split test --dial_pth weights/dial_after_dial.pth
# [DIAL] split=test  acc=0.6165  (random=0.25)
# [DIAL] confusion matrix (true row, pred col):
# [[363  11 124  22]
#  [ 26  92  14  21]
#  [ 42   3 146   0]
#  [ 90  17  40  58]]
# PS C:\Users\311\Downloads\Senior_Thesis> python evalu.py --data_root clock_kaggle --csv rotations.csv --split test --hour_pth weights/hour_after_hands.pth --minute_pth weights/minute_after_hands.pth
# [HANDS] split=test  hour_acc=0.2619  minute_acc=0.5706  joint=0.1646  (random=0.0833)

# python evalu.py  --data_root clock_kaggle --csv rotations_new.csv --split test --hour_pth weights_epoch10_csvnew_hands/hour_after_hands.pth --minute_pth weights_epoch10_csvnew_hands/minute_after_hands.pth
# [HANDS] split=test  hour_acc=0.1740  minute_acc=0.0730  joint=0.0196  (random=0.0833)