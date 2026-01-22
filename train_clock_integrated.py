"""train_clock_integrated.py

DeepProbLog 統合版（dial + hands + clock rules）

このスクリプトがやること:
  1) rotation.csv（1つ）から
      - dial rotation (0/90/180/270 -> rot_cls 0..3)
      - hour/minute hand angle class (0..11)  を生成
      - time (Hour, Minute) を取り出す
  2) 3つの学習タスクを順番に回せる
      - dial:  dial(X, RotCls)
      - hands: hands(X, HImgCls, MImgCls)
      - time:  time(X, Hour, Minute)  ※ clock_integrated.pl の制約で統合

前提:
  - models/clock_integrated.pl が存在
  - rotation.csv の file 列が 例: train/1-00/0.jpg のような相対パス
  - data_root が 例: /Users/ruka/Senior_Thesis/clock_kaggle

実行例:
  python train_clock_integrated.py \
    --data_root /Users/ruka/Senior_Thesis/clock_kaggle \
    --csv rotation.csv \
    --run pretrain_and_finetune

※ DeepProbLog / ProbLog のバージョン差があるので、
   既存のあなたの動く環境（model_hands2.pyが動く環境）に合わせています。
"""

import argparse
import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

from deepproblog.train import train_model
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.dataset import DataLoader, Dataset
from deepproblog.engines import ExactEngine

from problog.logic import Term, Constant, Var
from deepproblog.query import Query


# ------------------------------
# 0) Utils
# ------------------------------

def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rotation_deg_to_cls(rotation_deg: float) -> int:
    """rotation_deg（0/90/180/270） -> rot_cls（0..3）"""
    d = int(round(float(rotation_deg))) % 360
    return (d // 90) % 4


def angle_deg_clockwise_from_12(dx: float, dy: float) -> float:
    """画像座標 (x→右, y→下) で、12時方向を0°として時計回り[0,360)"""
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def angle_cls_12(hand_x: float, hand_y: float, cx: float, cy: float) -> int:
    """(hand_x,hand_y) から 30°刻み最近傍クラス (0..11)"""
    dx = float(hand_x) - float(cx)
    dy = float(hand_y) - float(cy)
    deg = angle_deg_clockwise_from_12(dx, dy)
    return int((deg + 15.0) // 30.0) % 12


# ------------------------------
# 1) Nets
# ------------------------------

class ResNetClassifier(nn.Module):
    """ResNet18 -> Softmax。DeepProbLog は確率を期待することが多いので Softmax を含める。"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_f = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_f, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # DeepProbLog から Constant(list) が来ることがあるため吸収
        if isinstance(x, list):
            x = torch.stack([item.value for item in x])

        x = x.to(next(self.parameters()).device)
        return self.backbone(x)


# ------------------------------
# 2) Torch Dataset (rotation.csv)
# ------------------------------

@dataclass
class RowLabels:
    hour_time: int
    minute_time: int
    rot_cls: int
    h_img_cls: int
    m_img_cls: int


class RotationCsvTorchDataset(torch.utils.data.Dataset):
    """rotation.csv から統合用ラベルを作って返す。

    必須列:
      file,time_h,time_m,rotation_deg,status,cx,cy,minute_x,minute_y,hour_x,hour_y
    """

    def __init__(
        self,
        data_root: str,
        csv_path: str,
        subset: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = data_root
        self.csv_path = csv_path
        self.subset = self._normalize_subset(subset)
        self.transform = transform

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

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

        # status == ok
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()

        # subset
        prefix = self.subset + "/"
        df = df[df["file"].astype(str).str.startswith(prefix)].copy()

        # numeric coercion
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

        # time filters (minute should be 0..55 step5; hour 1..12)
        df = df[(df["time_h"] >= 1) & (df["time_h"] <= 12)].copy()
        df = df[(df["time_m"] >= 0) & (df["time_m"] <= 59)].copy()
        df = df[df["time_m"] % 5 == 0].copy()
        df = df.reset_index(drop=True)

        if max_samples is not None:
            df = df.iloc[: int(max_samples)].reset_index(drop=True)

        self.df = df

    @staticmethod
    def _normalize_subset(subset: str) -> str:
        s = (subset or "train").strip().lower()
        if s in {"val", "valid", "vaild", "validation"}:
            return "valid"
        if s in {"train", "test", "valid"}:
            return s
        return s

    def __len__(self):
        return len(self.df)

    def _get_labels(self, row) -> RowLabels:
        hour_time = int(row["time_h"])
        minute_time = int(row["time_m"])
        rot_cls = rotation_deg_to_cls(row["rotation_deg"])

        cx, cy = float(row["cx"]), float(row["cy"])
        mx, my = float(row["minute_x"]), float(row["minute_y"])
        hx, hy = float(row["hour_x"]), float(row["hour_y"])

        m_img_cls = angle_cls_12(mx, my, cx, cy)
        h_img_cls = angle_cls_12(hx, hy, cx, cy)

        return RowLabels(
            hour_time=hour_time,
            minute_time=minute_time,
            rot_cls=rot_cls,
            h_img_cls=h_img_cls,
            m_img_cls=m_img_cls,
        )

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = str(row["file"])
        img_path = os.path.join(self.data_root, rel_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        labels = self._get_labels(row)
        return img, labels


# ------------------------------
# 3) DeepProbLog Datasets
# ------------------------------

class DPBDialDataset(Dataset):
    """Query: dial(X, RotCls)."""

    def __init__(self, torch_ds: RotationCsvTorchDataset):
        self.ds = torch_ds
        self.cache = {}

    def __len__(self):
        return len(self.ds)

    def to_query(self, i: int):
        if i not in self.cache:
            img, lab = self.ds[i]
            q = Term("dial", Var("X"), Constant(int(lab.rot_cls)))
            self.cache[i] = Query(q, {Var("X"): Constant(img)})
        return self.cache[i]


class DPBHandsDataset(Dataset):
    """Query: hands(X, HImgCls, MImgCls)."""

    def __init__(self, torch_ds: RotationCsvTorchDataset):
        self.ds = torch_ds
        self.cache = {}

    def __len__(self):
        return len(self.ds)

    def to_query(self, i: int):
        if i not in self.cache:
            img, lab = self.ds[i]
            q = Term(
                "hands",
                Var("X"),
                Constant(int(lab.h_img_cls)),
                Constant(int(lab.m_img_cls)),
            )
            self.cache[i] = Query(q, {Var("X"): Constant(img)})
        return self.cache[i]


class DPBTimeDataset(Dataset):
    """Query: time(X, Hour, Minute)."""

    def __init__(self, torch_ds: RotationCsvTorchDataset):
        self.ds = torch_ds
        self.cache = {}

    def __len__(self):
        return len(self.ds)

    def to_query(self, i: int):
        if i not in self.cache:
            img, lab = self.ds[i]
            q = Term(
                "time",
                Var("X"),
                Constant(int(lab.hour_time)),
                Constant(int(lab.minute_time)),
            )
            self.cache[i] = Query(q, {Var("X"): Constant(img)})
        return self.cache[i]


# ------------------------------
# 4) Build model
# ------------------------------

def build_deepproblog_model(
    prolog_file: str,
    device: torch.device,
    lr_dial: float,
    lr_hands: float,
    weights_dial: Optional[str] = None,
    weights_hour: Optional[str] = None,
    weights_minute: Optional[str] = None,
) -> Tuple[Model, ResNetClassifier, ResNetClassifier, ResNetClassifier]:
    """Build Model + 3 networks (dial/hour/minute)."""

    cnn_dial = ResNetClassifier(num_classes=4).to(device)
    cnn_hour = ResNetClassifier(num_classes=12).to(device)
    cnn_minute = ResNetClassifier(num_classes=12).to(device)

    if weights_dial:
        cnn_dial.load_state_dict(torch.load(weights_dial, map_location=device))
    if weights_hour:
        cnn_hour.load_state_dict(torch.load(weights_hour, map_location=device))
    if weights_minute:
        cnn_minute.load_state_dict(torch.load(weights_minute, map_location=device))

    net_dial = Network(cnn_dial, "net_dial", batching=True)
    net_hour = Network(cnn_hour, "net_hour", batching=True)
    net_minute = Network(cnn_minute, "net_minute", batching=True)

    net_dial.optimizer = torch.optim.Adam(cnn_dial.parameters(), lr=lr_dial)
    net_hour.optimizer = torch.optim.Adam(cnn_hour.parameters(), lr=lr_hands)
    net_minute.optimizer = torch.optim.Adam(cnn_minute.parameters(), lr=lr_hands)

    model = Model(prolog_file, [net_dial, net_hour, net_minute])
    model.set_engine(ExactEngine(model))

    return model, cnn_dial, cnn_hour, cnn_minute


# ------------------------------
# 5) Training orchestration
# ------------------------------

def run_training(
    args,
    model: Model,
    train_torch: RotationCsvTorchDataset,
    test_torch: RotationCsvTorchDataset,
):
    if args.task == "dial":
        train_ds = DPBDialDataset(train_torch)
        test_ds = DPBDialDataset(test_torch)
    elif args.task == "hands":
        train_ds = DPBHandsDataset(train_torch)
        test_ds = DPBHandsDataset(test_torch)
    elif args.task == "time":
        train_ds = DPBTimeDataset(train_torch)
        test_ds = DPBTimeDataset(test_torch)
    else:
        raise ValueError(f"Unknown task: {args.task}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    if len(train_loader) == 0:
        raise RuntimeError("Training loader is empty. Check data_root/csv/subset filtering.")

    train_model(
        model,
        train_loader,
        args.epochs,
        test_iter=test_loader,
        log_iter=args.log_iter,
        profile=0,
    )


# ------------------------------
# 6) Main
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--data_root", type=str, required=True, help="例: /Users/ruka/Senior_Thesis/clock_kaggle")
    p.add_argument("--csv", type=str, default="rotation.csv", help="data_root配下のCSV名 or 絶対パス")

    p.add_argument("--prolog", type=str, default="models/clock_integrated.pl")

    p.add_argument("--max_train", type=int, default=None)
    p.add_argument("--max_test", type=int, default=None)

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--log_iter", type=int, default=100)

    p.add_argument("--lr_dial", type=float, default=1e-4)
    p.add_argument("--lr_hands", type=float, default=1e-4)

    p.add_argument("--task", type=str, choices=["dial", "hands", "time"], default="time")

    p.add_argument(
        "--run",
        type=str,
        choices=["dial", "hands", "time", "pretrain_and_finetune"],
        default="pretrain_and_finetune",
        help="dial→hands→time の順に回すなら pretrain_and_finetune",
    )

    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save_dir", type=str, default="weights")

    # load pretrained
    p.add_argument("--load_dial", type=str, default=None)
    p.add_argument("--load_hour", type=str, default=None)
    p.add_argument("--load_minute", type=str, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # resolve CSV path
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.data_root, csv_path)

    # transforms
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_torch = RotationCsvTorchDataset(
        data_root=args.data_root,
        csv_path=csv_path,
        subset="train",
        transform=tfm,
        max_samples=args.max_train,
    )
    test_torch = RotationCsvTorchDataset(
        data_root=args.data_root,
        csv_path=csv_path,
        subset="test",
        transform=tfm,
        max_samples=args.max_test,
    )

    if len(train_torch) == 0:
        raise RuntimeError("No training rows after filtering. Check your rotation.csv and subset paths.")

    os.makedirs(args.save_dir, exist_ok=True)

    # build model (possibly loading pretrained)
    model, cnn_dial, cnn_hour, cnn_minute = build_deepproblog_model(
        prolog_file=args.prolog,
        device=device,
        lr_dial=args.lr_dial,
        lr_hands=args.lr_hands,
        weights_dial=args.load_dial,
        weights_hour=args.load_hour,
        weights_minute=args.load_minute,
    )

    def save_all(tag: str):
        torch.save(cnn_dial.state_dict(), os.path.join(args.save_dir, f"dial_{tag}.pth"))
        torch.save(cnn_hour.state_dict(), os.path.join(args.save_dir, f"hour_{tag}.pth"))
        torch.save(cnn_minute.state_dict(), os.path.join(args.save_dir, f"minute_{tag}.pth"))
        print(f"[saved] {tag} -> {args.save_dir}/")

    if args.run == "dial":
        args.task = "dial"
        run_training(args, model, train_torch, test_torch)
        save_all("dial")
        return

    if args.run == "hands":
        args.task = "hands"
        run_training(args, model, train_torch, test_torch)
        save_all("hands")
        return

    if args.run == "time":
        args.task = "time"
        run_training(args, model, train_torch, test_torch)
        save_all("time")
        return

    # pretrain_and_finetune
    print("=== Stage 1: dial pretrain ===")
    args.task = "dial"
    run_training(args, model, train_torch, test_torch)
    save_all("after_dial")

    print("=== Stage 2: hands pretrain ===")
    args.task = "hands"
    run_training(args, model, train_torch, test_torch)
    save_all("after_hands")

    print("=== Stage 3: time finetune (integrated constraints) ===")
    args.task = "time"
    run_training(args, model, train_torch, test_torch)
    save_all("after_time")

    print("Done.")


if __name__ == "__main__":
    main()


# python train_clock_integrated.py \
#   --data_root /Users/ruka/Senior_Thesis/clock_kaggle \
#   --csv rotation.csv \
#   --run pretrain_and_finetune \
#   --epochs 10 \
#   --batch_size 8