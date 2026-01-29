import os
import argparse
import math
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from typing import Tuple


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


class DialHeatmapRotationNet(nn.Module):
    """
    Dial rotation を「12時キーポイントのヒートマップ」経由で推定するネットワーク。
    """

    def __init__(
        self,
        heatmap_size: int = 56,
        softargmax_temp: float = 1.0,
        angle_softmax_k: float = 8.0,
        use_imagenet: bool = True,
        in_size: int = 224,
        center_xy: Tuple[float, float] = (112.0, 112.0),
    ):
        super().__init__()
        self.heatmap_size = int(heatmap_size)
        self.softargmax_temp = float(softargmax_temp)
        self.angle_softmax_k = float(angle_softmax_k)
        self.in_size = int(in_size)
        self.cx, self.cy = float(center_xy[0]), float(center_xy[1])

        if use_imagenet:
            res = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        else:
            res = models.resnet18(weights=None)

        self.backbone = nn.Sequential(
            res.conv1, res.bn1, res.relu, res.maxpool,
            res.layer1, res.layer2, res.layer3, res.layer4
        )

        self.head = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(14, 14), mode="bilinear", align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(size=(self.heatmap_size, self.heatmap_size), mode="bilinear", align_corners=False),

            nn.Conv2d(64, 1, 1),
        )

    def forward_heatmap_logits(self, x: torch.Tensor) -> torch.Tensor:
        f = self.backbone(x)
        hm = self.head(f)
        return hm

    @staticmethod
    def _softargmax_2d(hm_logits: torch.Tensor, temp: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = hm_logits.shape
        z = hm_logits.view(B, -1) / max(temp, 1e-8)
        p = nn.functional.softmax(z, dim=1)

        xs = torch.linspace(0, W - 1, W, device=hm_logits.device)
        ys = torch.linspace(0, H - 1, H, device=hm_logits.device)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        x = (p * xx[None, :]).sum(dim=1)
        y = (p * yy[None, :]).sum(dim=1)
        return x, y

    @staticmethod
    def _xy_to_angle_deg_top0(cx: float, cy: float, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dx = x - cx
        dy = y - cy
        ang = torch.atan2(dx, -dy) * (180.0 / math.pi)
        ang = torch.remainder(ang, 360.0)
        return ang

    @staticmethod
    def _angle_to_rot_probs(angle_deg: torch.Tensor, k: float = 8.0) -> torch.Tensor:
        targets = torch.tensor([0.0, 90.0, 180.0, 270.0], device=angle_deg.device)
        diff = torch.abs(angle_deg[:, None] - targets[None, :])
        delta = torch.minimum(diff, 360.0 - diff)
        logits = -k * delta
        return nn.functional.softmax(logits, dim=1)

    def forward(self, x):
        if isinstance(x, list):
            x = torch.stack([item.value for item in x])

        hm_logits = self.forward_heatmap_logits(x)
        xh, yh = self._softargmax_2d(hm_logits, temp=self.softargmax_temp)

        scale = float(self.in_size) / float(self.heatmap_size)
        x_img = xh * scale
        y_img = yh * scale

        angle = self._xy_to_angle_deg_top0(self.cx, self.cy, x_img, y_img)
        probs = self._angle_to_rot_probs(angle, k=self.angle_softmax_k)
        return probs


@torch.no_grad()
def eval_dial(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    cm = None

    for x, y, _fname in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        pred = torch.argmax(logits, dim=1)

        # initialize confusion matrix on first batch to match prediction width
        if cm is None:
            n_pred = logits.shape[1] if logits.ndim > 1 else 1
            cm = np.zeros((4, n_pred), dtype=int)

        correct += (pred == y).sum().item()
        total += y.numel()
        for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
            # expand columns if unseen prediction index appears
            if int(p) >= cm.shape[1]:
                new_cols = int(p) + 1
                new_cm = np.zeros((cm.shape[0], new_cols), dtype=int)
                new_cm[:, : cm.shape[1]] = cm
                cm = new_cm
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
        
        # Try to detect model type by loading weights and checking keys
        weights = torch.load(args.dial_pth, map_location=device)

        # If weights contain "head." keys, it's DialHeatmapRotationNet
        if any(k.startswith("head.") for k in weights.keys()):
            dial_model = DialHeatmapRotationNet().to(device)
        else:
            # Try to infer classifier output size from checkpoint (avoid size-mismatch)
            num_classes = 4
            # common possible keys for linear weight/bias inside our Sequential fc
            possible_keys = ("backbone.fc.0.weight", "backbone.fc.weight", "backbone.fc.0.bias", "backbone.fc.bias")
            for k in possible_keys:
                if k in weights:
                    w = weights[k]
                    try:
                        num_classes = int(w.shape[0])
                        break
                    except Exception:
                        pass
            dial_model = ResNetClassifier(num_classes=num_classes).to(device)

        dial_model.load_state_dict(weights)
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


# python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test --dial_pth weights_ep10_csvnew12_heatmap_hands/dial_after_dial.pth
# [DIAL] split=test  acc=0.9691  (random=0.25)
# [DIAL] confusion matrix (true row, pred col):
# [[510   0   9   1]
#  [  2 147   1   3]
#  [  5   0 186   0]
#  [  3   7   2 193]]
# PS C:\Users\311\Downloads\Senior_Thesis> python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test --dial_pth weights_ep10_csvnew12_heatmap_hands/hour_after_time.pth
# [DIAL] split=test  acc=0.0917  (random=0.25)
# [DIAL] confusion matrix (true row, pred col):
# [[ 62  38  30  59  29  48  42 100  30   0  44  38]
#  [ 25  11   5  21   7  13   9  28   9   0  13  12]
#  [ 30  16   9  15  16  17  15  45   7   1  21  13]]
# PS C:\Users\311\Downloads\Senior_Thesis> python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test --dial_pth weights_ep10_csvnew12_heatmap_hands/hour_after_dial.pth
# [DIAL] split=test  acc=0.1431  (random=0.25)
# [DIAL] confusion matrix (true row, pred col):
# [[  1 305  92  95   0   5   0  13   5   0   4   0]
#  [  0  84  29  24   0   6   0   6   2   0   2   0]
#  [  2  98  35  48   0   1   0   2   1   0   4   0]
#  [  0 114  34  33   0   3   0  13   2   0   6   0]]
# PS C:\Users\311\Downloads\Senior_Thesis> python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test --dial_pth weights_ep10_csvnew12_heatmap_hands/hour_after_hands.pth
# [DIAL] split=test  acc=0.1160  (random=0.25)
# [DIAL] confusion matrix (true row, pred col):
# [[  0 382  16   0   0   2   5   2  41  71   0   1]
#  [  0 119   2   0   0   0   1   0   9  21   0   1]
#  [  0 145   5   0   0   0   1   0  13  26   0   1]
#  [  0 155   4   0   0   1   2   0  12  31   0   0]]
# PS C:\Users\311\Downloads\Senior_Thesis> python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test --dial_pth weights_ep10_csvnew12_heatmap_hands/minute_after_time.pth 
# [DIAL] split=test  acc=0.0674  (random=0.25)
# [DIAL] confusion matrix (true row, pred col):
# [[  0 109  46  41  46  54  89  23   0   0  38  74]
#  [  0  32  15  19   7  13  26   8   0   0  10  23]
#  [  0  45  27  13  17  30  19   6   0   0  12  22]
#  [  0  41  22  13  19  24  36   0   0   0  17  33]]
# PS C:\Users\311\Downloads\Senior_Thesis> python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test --dial_pth weights_ep10_csvnew12_heatmap_hands/minute_after_dial.pth
# [DIAL] split=test  acc=0.0327  (random=0.25)
# [DIAL] confusion matrix (true row, pred col):
# [[ 28   2   0   0   8   0 123   1   0 279  54  25]
#  [  6   7   0   0   8   0  24   0   0  80  26   2]
#  [ 11   1   0   0   8   0  34   1   0 107  28   1]
# #  [ 14   3   1   0   3   0  48   0   0 105  22   9]]
# python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test --hour_pth  weights_ep10_csvnew12_heatmap_hands/hour_after_hands.pth --minute_pth  weights_ep10_csvnew12_heatmap_hands/minute_after_hands.pth
# [HANDS] split=test  hour_acc=0.2404  minute_acc=0.3302  joint=0.0814  (random=0.0833)

# ruka@rukanoMacBook-Air Senior_Thesis % python evalu.py --data_root clock_kaggle --csv rotations.csv --split test \
#   --hour_pth weights_ep10_csvnew12_heatmap_hands/hour_after_hands.pth --minute_pth weights_ep10_csvnew12_heatmap_hands/minute_after_hands.pth

# [HANDS] split=test  hour_acc=0.2161  minute_acc=0.3302  joint=0.0730  (random=0.0833)
# ruka@rukanoMacBook-Air Senior_Thesis % python evalu.py --data_root clock_kaggle --csv rotations.csv --split test \
#   --hour_pth weights_ep10_csvnew12_heatmap_hands/hour_after_time.pth --minute_pth weights_ep10_csvnew12_heatmap_hands/minute_after_time.pth

# [HANDS] split=test  hour_acc=0.5014  minute_acc=0.4752  joint=0.2208  (random=0.0833)
# ruka@rukanoMacBook-Air Senior_Thesis % python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test \  --hour_pth weights_ep10_csvnew12_heatmap_hands/hour_after_time.pth --minute_pth weights_ep10_csvnew12_heatmap_hands/minute_after_time.pth

# [HANDS] split=test  hour_acc=0.6193  minute_acc=0.4771  joint=0.2825  (random=0.0833)
# ruka@rukanoMacBook-Air Senior_Thesis % python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split train\
#   --hour_pth weights_ep10_csvnew12_heatmap_hands/hour_after_time.pth --minute_pth weights_ep10_csvnew12_heatmap_hands/minute_after_time.pth

# [HANDS] split=train  hour_acc=0.6179  minute_acc=0.5033  joint=0.2914  (random=0.0833)

# python train_clock_integrated_heatmap_softlabel.py --data_root clock_kaggle --csv rotations_new.csv --run pretrain_and_finetune --epochs 10 --batch_size 8 --hands_softlabel_sigma 12 --finetune_dial --dial_data_ro
# ot clock_kaggle --save_dir weights_softlabel --dial_arch heatmap --dial_kp_csv rotations_new_12.csv 
# /home/user/.pyenv/versions/3.10.0/lib/python3.10/site-packages/deepproblog/engines/__init__.py:6: UserWarning: ApproximateEngine is not available as PySwip could not be found
#   warnings.warn("ApproximateEngine is not available as PySwip could not be found")
# Using device: cuda
# === Stage 1: dial pretrain (NOHANDS) ===
# [dial-kp][ep 1] iter 100 loss=0.0787 acc=0.6913
# [dial-kp][ep 1] iter 200 loss=0.0559 acc=0.7712
# [dial-kp][ep 1] iter 300 loss=0.0470 acc=0.8079
# [dial-kp][ep 1] iter 400 loss=0.0423 acc=0.8294
# [dial-kp][ep 1] iter 500 loss=0.0392 acc=0.8460
# [dial-kp][ep 1] iter 600 loss=0.0369 acc=0.8619
# [dial-kp][ep 1] iter 700 loss=0.0353 acc=0.8725
# [dial-kp][ep 1] iter 800 loss=0.0339 acc=0.8817
# [dial-kp][ep 1] iter 900 loss=0.0329 acc=0.8890
# [dial-kp][ep 1] iter 1000 loss=0.0321 acc=0.8942
# [dial-kp][ep 1] train_loss=0.0319 train_acc=0.8959  test_loss=0.0245 test_acc=0.9383
# [dial-kp][ep 2] iter 100 loss=0.0240 acc=0.9513
# [dial-kp][ep 2] iter 200 loss=0.0239 acc=0.9544
# [dial-kp][ep 2] iter 300 loss=0.0237 acc=0.9571
# [dial-kp][ep 2] iter 400 loss=0.0237 acc=0.9587
# [dial-kp][ep 2] iter 500 loss=0.0236 acc=0.9603
# [dial-kp][ep 2] iter 600 loss=0.0236 acc=0.9590
# [dial-kp][ep 2] iter 700 loss=0.0237 acc=0.9571
# [dial-kp][ep 2] iter 800 loss=0.0237 acc=0.9575
# [dial-kp][ep 2] iter 900 loss=0.0237 acc=0.9567
# [dial-kp][ep 2] iter 1000 loss=0.0237 acc=0.9566
# [dial-kp][ep 2] train_loss=0.0237 train_acc=0.9573  test_loss=0.0239 test_acc=0.9373
# [dial-kp][ep 3] iter 100 loss=0.0227 acc=0.9637
# [dial-kp][ep 3] iter 200 loss=0.0226 acc=0.9688
# [dial-kp][ep 3] iter 300 loss=0.0227 acc=0.9658
# [dial-kp][ep 3] iter 400 loss=0.0227 acc=0.9659
# [dial-kp][ep 3] iter 500 loss=0.0226 acc=0.9665
# [dial-kp][ep 3] iter 600 loss=0.0226 acc=0.9665
# [dial-kp][ep 3] iter 700 loss=0.0227 acc=0.9661
# [dial-kp][ep 3] iter 800 loss=0.0226 acc=0.9675
# [dial-kp][ep 3] iter 900 loss=0.0226 acc=0.9685
# [dial-kp][ep 3] iter 1000 loss=0.0226 acc=0.9688
# [dial-kp][ep 3] train_loss=0.0227 train_acc=0.9679  test_loss=0.0230 test_acc=0.9542
# [dial-kp][ep 4] iter 100 loss=0.0221 acc=0.9738
# [dial-kp][ep 4] iter 200 loss=0.0223 acc=0.9744
# [dial-kp][ep 4] iter 300 loss=0.0224 acc=0.9700
# [dial-kp][ep 4] iter 400 loss=0.0224 acc=0.9716
# [dial-kp][ep 4] iter 500 loss=0.0224 acc=0.9710
# [dial-kp][ep 4] iter 600 loss=0.0224 acc=0.9715
# [dial-kp][ep 4] iter 700 loss=0.0223 acc=0.9705
# [dial-kp][ep 4] iter 800 loss=0.0223 acc=0.9700
# [dial-kp][ep 4] iter 900 loss=0.0223 acc=0.9708
# [dial-kp][ep 4] iter 1000 loss=0.0223 acc=0.9716
# [dial-kp][ep 4] train_loss=0.0223 train_acc=0.9722  test_loss=0.0230 test_acc=0.9579
# [dial-kp][ep 5] iter 100 loss=0.0220 acc=0.9750
# [dial-kp][ep 5] iter 200 loss=0.0220 acc=0.9769
# [dial-kp][ep 5] iter 300 loss=0.0220 acc=0.9775
# [dial-kp][ep 5] iter 400 loss=0.0221 acc=0.9762
# [dial-kp][ep 5] iter 500 loss=0.0221 acc=0.9775
# [dial-kp][ep 5] iter 600 loss=0.0221 acc=0.9783
# [dial-kp][ep 5] iter 700 loss=0.0220 acc=0.9788
# [dial-kp][ep 5] iter 800 loss=0.0220 acc=0.9784
# [dial-kp][ep 5] iter 900 loss=0.0220 acc=0.9774
# [dial-kp][ep 5] iter 1000 loss=0.0221 acc=0.9762
# [dial-kp][ep 5] train_loss=0.0221 train_acc=0.9759  test_loss=0.0226 test_acc=0.9626
# [dial-kp][ep 6] iter 100 loss=0.0219 acc=0.9788
# [dial-kp][ep 6] iter 200 loss=0.0219 acc=0.9819
# [dial-kp][ep 6] iter 300 loss=0.0219 acc=0.9796
# [dial-kp][ep 6] iter 400 loss=0.0218 acc=0.9816
# [dial-kp][ep 6] iter 500 loss=0.0217 acc=0.9828
# [dial-kp][ep 6] iter 600 loss=0.0218 acc=0.9831
# [dial-kp][ep 6] iter 700 loss=0.0218 acc=0.9823
# [dial-kp][ep 6] iter 800 loss=0.0218 acc=0.9817
# [dial-kp][ep 6] iter 900 loss=0.0218 acc=0.9824
# [dial-kp][ep 6] iter 1000 loss=0.0218 acc=0.9819
# [dial-kp][ep 6] train_loss=0.0218 train_acc=0.9817  test_loss=0.0228 test_acc=0.9532
# [dial-kp][ep 7] iter 100 loss=0.0215 acc=0.9862
# [dial-kp][ep 7] iter 200 loss=0.0215 acc=0.9844
# [dial-kp][ep 7] iter 300 loss=0.0216 acc=0.9829
# [dial-kp][ep 7] iter 400 loss=0.0216 acc=0.9831
# [dial-kp][ep 7] iter 500 loss=0.0218 acc=0.9828
# [dial-kp][ep 7] iter 600 loss=0.0218 acc=0.9817
# [dial-kp][ep 7] iter 700 loss=0.0219 acc=0.9823
# [dial-kp][ep 7] iter 800 loss=0.0218 acc=0.9830
# [dial-kp][ep 7] iter 900 loss=0.0219 acc=0.9828
# [dial-kp][ep 7] iter 1000 loss=0.0219 acc=0.9821
# [dial-kp][ep 7] train_loss=0.0219 train_acc=0.9819  test_loss=0.0226 test_acc=0.9588
# [dial-kp][ep 8] iter 100 loss=0.0215 acc=0.9875
# [dial-kp][ep 8] iter 200 loss=0.0213 acc=0.9906
# [dial-kp][ep 8] iter 300 loss=0.0214 acc=0.9875
# [dial-kp][ep 8] iter 400 loss=0.0214 acc=0.9875
# [dial-kp][ep 8] iter 500 loss=0.0214 acc=0.9878
# [dial-kp][ep 8] iter 600 loss=0.0214 acc=0.9885
# [dial-kp][ep 8] iter 700 loss=0.0214 acc=0.9896
# [dial-kp][ep 8] iter 800 loss=0.0214 acc=0.9891
# [dial-kp][ep 8] iter 900 loss=0.0215 acc=0.9889
# [dial-kp][ep 8] iter 1000 loss=0.0215 acc=0.9892
# [dial-kp][ep 8] train_loss=0.0215 train_acc=0.9890  test_loss=0.0229 test_acc=0.9607
# [dial-kp][ep 9] iter 100 loss=0.0216 acc=0.9925
# [dial-kp][ep 9] iter 200 loss=0.0216 acc=0.9938
# [dial-kp][ep 9] iter 300 loss=0.0216 acc=0.9912
# [dial-kp][ep 9] iter 400 loss=0.0217 acc=0.9900
# [dial-kp][ep 9] iter 500 loss=0.0217 acc=0.9908
# [dial-kp][ep 9] iter 600 loss=0.0217 acc=0.9894
# [dial-kp][ep 9] iter 700 loss=0.0217 acc=0.9891
# [dial-kp][ep 9] iter 800 loss=0.0217 acc=0.9898
# [dial-kp][ep 9] iter 900 loss=0.0216 acc=0.9903
# [dial-kp][ep 9] iter 1000 loss=0.0216 acc=0.9899
# [dial-kp][ep 9] train_loss=0.0217 train_acc=0.9896  test_loss=0.0235 test_acc=0.9579
# [dial-kp][ep 10] iter 100 loss=0.0219 acc=0.9900
# [dial-kp][ep 10] iter 200 loss=0.0216 acc=0.9925
# [dial-kp][ep 10] iter 300 loss=0.0215 acc=0.9933
# [dial-kp][ep 10] iter 400 loss=0.0215 acc=0.9912
# [dial-kp][ep 10] iter 500 loss=0.0215 acc=0.9910
# [dial-kp][ep 10] iter 600 loss=0.0214 acc=0.9915
# [dial-kp][ep 10] iter 700 loss=0.0215 acc=0.9911
# [dial-kp][ep 10] iter 800 loss=0.0215 acc=0.9906
# [dial-kp][ep 10] iter 900 loss=0.0215 acc=0.9912
# [dial-kp][ep 10] iter 1000 loss=0.0215 acc=0.9912
# [dial-kp][ep 10] train_loss=0.0215 train_acc=0.9913  test_loss=0.0228 test_acc=0.9645
# [saved] after_dial -> weights_softlabel/
# === Stage 2: hands pretrain ===
# [hands-soft] Training for 10 epoch(s)  sigma=12.0deg  batch=8
#   [ep01 step0100] loss=3.2400 hour_acc=0.4575 min_acc=0.6138
#   [ep01 step0200] loss=2.3827 hour_acc=0.6225 min_acc=0.7769
#   [ep01 step0300] loss=2.0750 hour_acc=0.6817 min_acc=0.8325
#   [ep01 step0400] loss=1.8803 hour_acc=0.7200 min_acc=0.8653
#   [ep01 step0500] loss=1.7731 hour_acc=0.7395 min_acc=0.8835
#   [ep01 step0600] loss=1.6820 hour_acc=0.7573 min_acc=0.8977
#   [ep01 step0700] loss=1.6215 hour_acc=0.7729 min_acc=0.9062
#   [ep01 step0800] loss=1.5692 hour_acc=0.7870 min_acc=0.9136
#   [ep01 step0900] loss=1.5351 hour_acc=0.7949 min_acc=0.9179
#   [ep01 step1000] loss=1.5038 hour_acc=0.8011 min_acc=0.9209
# [ep01] train_loss=1.4903 train_hour_acc=0.8043 train_min_acc=0.9226 | test_loss=1.1351 test_hour_acc=0.9046 test_min_acc=0.9560
#   [ep02 step0100] loss=1.1300 hour_acc=0.9025 min_acc=0.9637
#   [ep02 step0200] loss=1.1162 hour_acc=0.8988 min_acc=0.9663
#   [ep02 step0300] loss=1.1186 hour_acc=0.8933 min_acc=0.9675
#   [ep02 step0400] loss=1.1106 hour_acc=0.8953 min_acc=0.9688
#   [ep02 step0500] loss=1.1242 hour_acc=0.8882 min_acc=0.9657
#   [ep02 step0600] loss=1.1196 hour_acc=0.8919 min_acc=0.9660
#   [ep02 step0700] loss=1.1238 hour_acc=0.8900 min_acc=0.9654
#   [ep02 step0800] loss=1.1226 hour_acc=0.8900 min_acc=0.9656
#   [ep02 step0900] loss=1.1243 hour_acc=0.8901 min_acc=0.9650
#   [ep02 step1000] loss=1.1234 hour_acc=0.8909 min_acc=0.9640
# [ep02] train_loss=1.1283 train_hour_acc=0.8905 train_min_acc=0.9635 | test_loss=1.1263 test_hour_acc=0.8952 test_min_acc=0.9588
#   [ep03 step0100] loss=1.0623 hour_acc=0.8950 min_acc=0.9650
#   [ep03 step0200] loss=1.0724 hour_acc=0.9012 min_acc=0.9663
#   [ep03 step0300] loss=1.0895 hour_acc=0.8967 min_acc=0.9592
#   [ep03 step0400] loss=1.0774 hour_acc=0.8972 min_acc=0.9628
#   [ep03 step0500] loss=1.0744 hour_acc=0.8978 min_acc=0.9633
#   [ep03 step0600] loss=1.0745 hour_acc=0.9002 min_acc=0.9646
#   [ep03 step0700] loss=1.0728 hour_acc=0.9016 min_acc=0.9645
#   [ep03 step0800] loss=1.0749 hour_acc=0.9009 min_acc=0.9642
#   [ep03 step0900] loss=1.0683 hour_acc=0.9012 min_acc=0.9658
#   [ep03 step1000] loss=1.0659 hour_acc=0.9036 min_acc=0.9674
# [ep03] train_loss=1.0665 train_hour_acc=0.9031 train_min_acc=0.9674 | test_loss=1.1519 test_hour_acc=0.8877 test_min_acc=0.9532
#   [ep04 step0100] loss=1.0445 hour_acc=0.9125 min_acc=0.9650
#   [ep04 step0200] loss=1.0401 hour_acc=0.9062 min_acc=0.9688
#   [ep04 step0300] loss=1.0474 hour_acc=0.9054 min_acc=0.9671
#   [ep04 step0400] loss=1.0414 hour_acc=0.9078 min_acc=0.9684
#   [ep04 step0500] loss=1.0442 hour_acc=0.9095 min_acc=0.9690
#   [ep04 step0600] loss=1.0410 hour_acc=0.9121 min_acc=0.9696
#   [ep04 step0700] loss=1.0395 hour_acc=0.9116 min_acc=0.9695
#   [ep04 step0800] loss=1.0378 hour_acc=0.9128 min_acc=0.9692
#   [ep04 step0900] loss=1.0380 hour_acc=0.9121 min_acc=0.9694
#   [ep04 step1000] loss=1.0381 hour_acc=0.9137 min_acc=0.9694
# [ep04] train_loss=1.0380 train_hour_acc=0.9141 train_min_acc=0.9697 | test_loss=1.0844 test_hour_acc=0.9027 test_min_acc=0.9626
#   [ep05 step0100] loss=1.0212 hour_acc=0.9213 min_acc=0.9688
#   [ep05 step0200] loss=1.0127 hour_acc=0.9231 min_acc=0.9712
#   [ep05 step0300] loss=1.0061 hour_acc=0.9213 min_acc=0.9721
#   [ep05 step0400] loss=1.0063 hour_acc=0.9203 min_acc=0.9725
#   [ep05 step0500] loss=1.0060 hour_acc=0.9235 min_acc=0.9730
#   [ep05 step0600] loss=1.0009 hour_acc=0.9248 min_acc=0.9744
#   [ep05 step0700] loss=1.0042 hour_acc=0.9248 min_acc=0.9736
#   [ep05 step0800] loss=1.0057 hour_acc=0.9242 min_acc=0.9730
#   [ep05 step0900] loss=1.0124 hour_acc=0.9197 min_acc=0.9721
#   [ep05 step1000] loss=1.0113 hour_acc=0.9195 min_acc=0.9729
# [ep05] train_loss=1.0118 train_hour_acc=0.9198 train_min_acc=0.9731 | test_loss=1.0936 test_hour_acc=0.9308 test_min_acc=0.9579
#   [ep06 step0100] loss=0.9777 hour_acc=0.9363 min_acc=0.9812
#   [ep06 step0200] loss=0.9897 hour_acc=0.9319 min_acc=0.9775
#   [ep06 step0300] loss=0.9844 hour_acc=0.9325 min_acc=0.9771
#   [ep06 step0400] loss=0.9882 hour_acc=0.9337 min_acc=0.9762
#   [ep06 step0500] loss=0.9899 hour_acc=0.9310 min_acc=0.9760
#   [ep06 step0600] loss=0.9893 hour_acc=0.9290 min_acc=0.9756
#   [ep06 step0700] loss=0.9854 hour_acc=0.9313 min_acc=0.9764
#   [ep06 step0800] loss=0.9859 hour_acc=0.9297 min_acc=0.9767
#   [ep06 step0900] loss=0.9876 hour_acc=0.9286 min_acc=0.9767
#   [ep06 step1000] loss=0.9882 hour_acc=0.9290 min_acc=0.9768
# [ep06] train_loss=0.9881 train_hour_acc=0.9284 train_min_acc=0.9771 | test_loss=1.1263 test_hour_acc=0.8999 test_min_acc=0.9626
#   [ep07 step0100] loss=0.9818 hour_acc=0.9175 min_acc=0.9875
#   [ep07 step0200] loss=0.9773 hour_acc=0.9206 min_acc=0.9831
#   [ep07 step0300] loss=0.9795 hour_acc=0.9175 min_acc=0.9779
#   [ep07 step0400] loss=0.9703 hour_acc=0.9241 min_acc=0.9803
#   [ep07 step0500] loss=0.9704 hour_acc=0.9235 min_acc=0.9792
#   [ep07 step0600] loss=0.9693 hour_acc=0.9267 min_acc=0.9792
#   [ep07 step0700] loss=0.9762 hour_acc=0.9257 min_acc=0.9777
#   [ep07 step0800] loss=0.9745 hour_acc=0.9275 min_acc=0.9777
#   [ep07 step0900] loss=0.9774 hour_acc=0.9285 min_acc=0.9779
#   [ep07 step1000] loss=0.9771 hour_acc=0.9297 min_acc=0.9782
# [ep07] train_loss=0.9754 train_hour_acc=0.9302 train_min_acc=0.9784 | test_loss=1.1068 test_hour_acc=0.9205 test_min_acc=0.9579
#   [ep08 step0100] loss=0.9598 hour_acc=0.9313 min_acc=0.9862
#   [ep08 step0200] loss=0.9675 hour_acc=0.9269 min_acc=0.9844
#   [ep08 step0300] loss=0.9556 hour_acc=0.9346 min_acc=0.9862
#   [ep08 step0400] loss=0.9543 hour_acc=0.9366 min_acc=0.9850
#   [ep08 step0500] loss=0.9560 hour_acc=0.9380 min_acc=0.9850
#   [ep08 step0600] loss=0.9552 hour_acc=0.9385 min_acc=0.9850
#   [ep08 step0700] loss=0.9552 hour_acc=0.9382 min_acc=0.9834
#   [ep08 step0800] loss=0.9578 hour_acc=0.9372 min_acc=0.9831
#   [ep08 step0900] loss=0.9583 hour_acc=0.9389 min_acc=0.9828
#   [ep08 step1000] loss=0.9572 hour_acc=0.9390 min_acc=0.9828
# [ep08] train_loss=0.9566 train_hour_acc=0.9392 train_min_acc=0.9828 | test_loss=1.1096 test_hour_acc=0.9065 test_min_acc=0.9616
#   [ep09 step0100] loss=0.9538 hour_acc=0.9187 min_acc=0.9875
#   [ep09 step0200] loss=0.9381 hour_acc=0.9344 min_acc=0.9856
#   [ep09 step0300] loss=0.9329 hour_acc=0.9392 min_acc=0.9854
#   [ep09 step0400] loss=0.9253 hour_acc=0.9434 min_acc=0.9866
#   [ep09 step0500] loss=0.9316 hour_acc=0.9430 min_acc=0.9852
#   [ep09 step0600] loss=0.9355 hour_acc=0.9425 min_acc=0.9838
#   [ep09 step0700] loss=0.9366 hour_acc=0.9430 min_acc=0.9845
#   [ep09 step0800] loss=0.9395 hour_acc=0.9431 min_acc=0.9839
#   [ep09 step0900] loss=0.9428 hour_acc=0.9425 min_acc=0.9833
#   [ep09 step1000] loss=0.9459 hour_acc=0.9410 min_acc=0.9829
# [ep09] train_loss=0.9459 train_hour_acc=0.9408 train_min_acc=0.9828 | test_loss=1.1478 test_hour_acc=0.9177 test_min_acc=0.9542
#   [ep10 step0100] loss=0.9609 hour_acc=0.9237 min_acc=0.9850
#   [ep10 step0200] loss=0.9545 hour_acc=0.9244 min_acc=0.9869
#   [ep10 step0300] loss=0.9423 hour_acc=0.9275 min_acc=0.9883
#   [ep10 step0400] loss=0.9429 hour_acc=0.9334 min_acc=0.9869
#   [ep10 step0500] loss=0.9410 hour_acc=0.9350 min_acc=0.9872
#   [ep10 step0600] loss=0.9394 hour_acc=0.9373 min_acc=0.9865
#   [ep10 step0700] loss=0.9366 hour_acc=0.9393 min_acc=0.9868
#   [ep10 step0800] loss=0.9378 hour_acc=0.9408 min_acc=0.9869
#   [ep10 step0900] loss=0.9376 hour_acc=0.9408 min_acc=0.9865
#   [ep10 step1000] loss=0.9352 hour_acc=0.9413 min_acc=0.9869
# [ep10] train_loss=0.9354 train_hour_acc=0.9415 train_min_acc=0.9870 | test_loss=1.1265 test_hour_acc=0.9177 test_min_acc=0.9682
# [hands-soft] done.
# [saved] after_hands -> weights_softlabel/
# === Stage 3: time finetune (integrated constraints) ===
# Training  for 10 epoch(s)
# Epoch 1
# Iteration:  100         s:229.5644      Average Loss:  1.8358038902282714
# Iteration:  200         s:232.1502      Average Loss:  0.7112592706084251
# Iteration:  300         s:231.2096      Average Loss:  0.4629359680227935
# Iteration:  400         s:231.2432      Average Loss:  0.4110844030417502
# Iteration:  500         s:232.8177      Average Loss:  0.3654044604860246
# Iteration:  600         s:231.4080      Average Loss:  0.25209031712263824
# Iteration:  700         s:232.1810      Average Loss:  0.297737050158903
# Iteration:  800         s:232.5467      Average Loss:  0.23281547564081848
# Iteration:  900         s:232.9666      Average Loss:  0.24884631423279643
# Iteration:  1000        s:232.7407      Average Loss:  0.23826448954641818
# Epoch time:  2391.8252432346344
# Epoch 2
# Iteration:  1100        s:227.5355      Average Loss:  0.2048162423307076
# Iteration:  1200        s:231.6847      Average Loss:  0.24206312900874763
# Iteration:  1300        s:232.1449      Average Loss:  0.21993839372880758
# Iteration:  1400        s:232.1763      Average Loss:  0.2176829977100715
# Iteration:  1500        s:231.5056      Average Loss:  0.273145020538941
# Iteration:  1600        s:231.1552      Average Loss:  0.22441269097384065
# Iteration:  1700        s:230.1478      Average Loss:  0.21776473643723876
# Iteration:  1800        s:230.4849      Average Loss:  0.159334939152468
# Iteration:  1900        s:231.6221      Average Loss:  0.19214737601345405
# Iteration:  2000        s:227.8397      Average Loss:  0.31094364538555963
# Epoch time:  2379.011196374893
# Epoch 3
# Iteration:  2100        s:231.3950      Average Loss:  0.27508545726537703
# Iteration:  2200        s:231.1386      Average Loss:  0.13209947236580774
# Iteration:  2300        s:231.1271      Average Loss:  0.1908679467800539
# Iteration:  2400        s:231.0483      Average Loss:  0.2130412914045155
# Iteration:  2500        s:231.4701      Average Loss:  0.15438559839269148
# Iteration:  2600        s:230.5459      Average Loss:  0.24543893086141907
# Iteration:  2700        s:230.1897      Average Loss:  0.2277687744551804
# Iteration:  2800        s:229.3172      Average Loss:  0.16469258562778122
# Iteration:  2900        s:230.9601      Average Loss:  0.19592023846576923
# Iteration:  3000        s:230.5269      Average Loss:  0.14963654942344873
# Epoch time:  2378.384711742401
# Epoch 4
# Iteration:  3100        s:230.2458      Average Loss:  0.15407185171963647
# Iteration:  3200        s:230.7273      Average Loss:  0.1644745152129326
# Iteration:  3300        s:230.9638      Average Loss:  0.1708317422051914
# Iteration:  3400        s:230.4766      Average Loss:  0.14923919631226454
# Iteration:  3500        s:230.9868      Average Loss:  0.1871169764676597
# Iteration:  3600        s:230.5685      Average Loss:  0.17910877617483492
# Iteration:  3700        s:232.3329      Average Loss:  0.19767647713655606
# Iteration:  3800        s:230.7225      Average Loss:  0.1109512415289646
# Iteration:  3900        s:233.0641      Average Loss:  0.18330488311650697
# Iteration:  4000        s:231.9618      Average Loss:  0.2057502680073958
# Iteration:  4100        s:230.4928      Average Loss:  0.1969737197633367
# Epoch time:  2382.7749013900757
# Epoch 5
# Iteration:  4200        s:230.8867      Average Loss:  0.06684450466709677
# Iteration:  4300        s:231.7260      Average Loss:  0.04194013441214338
# Iteration:  4400        s:230.6182      Average Loss:  0.13717159842781257
# Iteration:  4500        s:228.0841      Average Loss:  0.303422552018892
# Iteration:  4600        s:228.8170      Average Loss:  0.2042250999197131
# Iteration:  4700        s:230.7438      Average Loss:  0.27602074734808413
# Iteration:  4800        s:231.8116      Average Loss:  0.1406346252176445
# Iteration:  4900        s:229.9601      Average Loss:  0.19863674022664782
# Iteration:  5000        s:232.3543      Average Loss:  0.0752627853717422
# Iteration:  5100        s:231.0251      Average Loss:  0.1266425884165801
# Epoch time:  2377.6019167900085
# Epoch 6
# Iteration:  5200        s:228.9164      Average Loss:  0.16588976583676412
# Iteration:  5300        s:228.1250      Average Loss:  0.15209531005530152
# Iteration:  5400        s:230.1149      Average Loss:  0.13498537886800477
# Iteration:  5500        s:231.1786      Average Loss:  0.13317713359778283
# Iteration:  5600        s:231.2193      Average Loss:  0.10586412175209262
# Iteration:  5700        s:230.7438      Average Loss:  0.13805498561559942
# Iteration:  5800        s:229.6138      Average Loss:  0.04731634440191556
# Iteration:  5900        s:227.2239      Average Loss:  0.05459578408772359
# Iteration:  6000        s:231.1264      Average Loss:  0.2116833132901229
# Iteration:  6100        s:231.2357      Average Loss:  0.07433111453719903
# Epoch time:  2369.554041624069
# Epoch 7
# Iteration:  6200        s:227.8905      Average Loss:  0.20188933032157366
# Iteration:  6300        s:231.1106      Average Loss:  0.1335042147547938
# Iteration:  6400        s:232.2083      Average Loss:  0.12788796981913036
# Iteration:  6500        s:231.1781      Average Loss:  0.04235969773231773
# Iteration:  6600        s:228.5134      Average Loss:  0.1325814851708128
# Iteration:  6700        s:230.6569      Average Loss:  0.07638594428513898
# Iteration:  6800        s:230.8797      Average Loss:  0.12854794407612644
# Iteration:  6900        s:232.0003      Average Loss:  0.09967127848853125
# Iteration:  7000        s:231.5073      Average Loss:  0.14180184677185026
# Iteration:  7100        s:232.5844      Average Loss:  0.041849411286821125
# Iteration:  7200        s:231.1435      Average Loss:  0.11721514475939329
# Epoch time:  2381.4916577339172
# Epoch 8
# Iteration:  7300        s:230.8634      Average Loss:  0.16491009074932664
# Iteration:  7400        s:231.6438      Average Loss:  0.10979991360218264
# Iteration:  7500        s:232.6382      Average Loss:  0.11650483279605396
# Iteration:  7600        s:230.5383      Average Loss:  0.08073399377404712
# Iteration:  7700        s:231.8104      Average Loss:  0.06696705133421346
# Iteration:  7800        s:230.6280      Average Loss:  0.1528552494820906
# Iteration:  7900        s:230.3812      Average Loss:  0.03165956576151075
# Iteration:  8000        s:230.5753      Average Loss:  0.09243668719864218
# Iteration:  8100        s:230.3783      Average Loss:  0.08167654024116927
# Iteration:  8200        s:232.5842      Average Loss:  0.14947340787417487
# Epoch time:  2384.5437140464783
# Epoch 9
# Iteration:  8300        s:230.2006      Average Loss:  0.06944539243500912
# Iteration:  8400        s:231.5556      Average Loss:  0.0428476556710666
# Iteration:  8500        s:231.6580      Average Loss:  0.07005943526426563
# Iteration:  8600        s:227.6003      Average Loss:  0.06530063713391428
# Iteration:  8700        s:229.7262      Average Loss:  0.06288665354877594
# Iteration:  8800        s:232.6964      Average Loss:  0.1723686534643639
# Iteration:  8900        s:230.9174      Average Loss:  0.0749960803607246
# Iteration:  9000        s:230.3911      Average Loss:  0.11383625935559394
# Iteration:  9100        s:232.0627      Average Loss:  0.0637679902210948
# Iteration:  9200        s:231.1418      Average Loss:  0.13537639829344697
# Epoch time:  2377.5407507419586
# Epoch 10
# Iteration:  9300        s:228.2314      Average Loss:  0.03561508073893492
# Iteration:  9400        s:230.6081      Average Loss:  0.023490750579803717
# Iteration:  9500        s:231.4436      Average Loss:  0.1385774758446496
# Iteration:  9600        s:228.2278      Average Loss:  0.05031829178755288
# Iteration:  9700        s:232.3525      Average Loss:  0.11334912760794395
# Iteration:  9800        s:232.3100      Average Loss:  0.1094729079309036
# Iteration:  9900        s:230.2069      Average Loss:  0.029650174233684084
# Iteration:  10000       s:230.4876      Average Loss:  0.04906142769323196
# Iteration:  10100       s:231.0834      Average Loss:  0.06192839431518223
# Iteration:  10200       s:232.2472      Average Loss:  0.11893014409666648
# Iteration:  10300       s:229.9763      Average Loss:  0.10233323110136552
# Epoch time:  2379.038364171982
# [saved] after_time -> weights_softlabel/
# Done.


# python evalu.py --data_root clock_kaggle --csv rotations_new.csv --split test  --minute_pth weights_softlabel/minute_after_time.pth --hour_pth weights_softlabel/hour_after_time.pth
# /home/user/Senior_Thesis/evalu.py:354: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   hour_model.load_state_dict(torch.load(args.hour_pth, map_location=device))
# /home/user/Senior_Thesis/evalu.py:355: FutureWarning: You are using `torch.load` with `weights_only=False` (the current default value), which uses the default pickle module implicitly. It is possible to construct malicious pickle data which will execute arbitrary code during unpickling (See https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). In a future release, the default value for `weights_only` will be flipped to `True`. This limits the functions that could be executed during unpickling. Arbitrary objects will no longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the user via `torch.serialization.add_safe_globals`. We recommend you start setting `weights_only=True` for any use case where you don't have full control of the loaded file. Please open an issue on GitHub for any issues related to this experimental feature.
#   minute_model.load_state_dict(torch.load(args.minute_pth, map_location=device))
# [HANDS] split=test  hour_acc=0.9364  minute_acc=0.9616  joint=0.9289  (random=0.0833)