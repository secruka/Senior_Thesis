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
        # hour座標が欠損（-1など）を落とす（hands/time用途）
        df = df[(df["hour_x"] >= 0) & (df["hour_y"] >= 0)].copy()
        df = df[(df["minute_x"] >= 0) & (df["minute_y"] >= 0)].copy()
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


#python train_clock_integrated.py --data_root clock_kaggle --csv rotations.csv --run pretrain_and_finetune --epochs 10 --batch_size 8        


# C:\Users\311\Downloads\Senior_Thesis> python train_clock_integrated.py --data_root clock_kaggle --csv rotations.csv --run pretrain_and_finetune --epochs 10 --batch_size 8
# C:\Users\311\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\site-packages\deepproblog\engines\__init__.py:6: UserWarning: ApproximateEngine is not available as PySwip could not be found
#   warnings.warn("ApproximateEngine is not available as PySwip could not be found")
# Using device: cuda
# === Stage 1: dial pretrain ===
# Training  for 10 epoch(s)
# Epoch 1
# Iteration:  600         s:12.3603       Average Loss:  0.3005255340831354
# Iteration:  700         s:12.4094       Average Loss:  0.23301976066082716
# Iteration:  800         s:12.5013       Average Loss:  0.21267994648311286
# Iteration:  900         s:12.3862       Average Loss:  0.25517314955359327
# Iteration:  1000        s:12.3810       Average Loss:  0.1898390742088668
# Epoch time:  128.92878699302673
# Epoch 2
# Iteration:  1100        s:12.4459       Average Loss:  0.18604897907120177
# Iteration:  1200        s:12.3134       Average Loss:  0.17410544612444936
# Iteration:  1300        s:12.2854       Average Loss:  0.22623921706574038
# Iteration:  1400        s:12.2273       Average Loss:  0.20904642681591212
# Iteration:  1500        s:12.2478       Average Loss:  0.1621265614242293
# Iteration:  1600        s:12.2894       Average Loss:  0.20527967456029728
# Iteration:  1700        s:12.3298       Average Loss:  0.19862552723148838
# Iteration:  1800        s:12.3067       Average Loss:  0.15660108840325848
# Iteration:  1900        s:12.3043       Average Loss:  0.23201091592898593
# Iteration:  2000        s:12.3161       Average Loss:  0.2195628464582842
# Epoch time:  126.85726857185364
# Epoch 3
# Iteration:  2100        s:12.2576       Average Loss:  0.15081972405081615
# Iteration:  2200        s:12.3066       Average Loss:  0.19279154291492887
# Iteration:  2300        s:12.3060       Average Loss:  0.1944828718167264
# Iteration:  2400        s:12.3328       Average Loss:  0.15849123219843023
# Iteration:  2500        s:12.3198       Average Loss:  0.12646463504061103
# Iteration:  2600        s:12.3125       Average Loss:  0.13967044101678766
# Iteration:  2700        s:12.3380       Average Loss:  0.12472282778820955
# Iteration:  2800        s:12.2911       Average Loss:  0.1310234331752872
# Iteration:  2900        s:12.2988       Average Loss:  0.12823569514672273
# Iteration:  3000        s:12.2851       Average Loss:  0.12298206744599156
# Epoch time:  126.85453701019287
# Epoch 4
# Iteration:  3100        s:12.2529       Average Loss:  0.1629923569454695
# Iteration:  3200        s:12.3643       Average Loss:  0.100348094208166
# Iteration:  3300        s:12.3019       Average Loss:  0.10410144887806382
# Iteration:  3400        s:12.2955       Average Loss:  0.1068086678700638
# Iteration:  3500        s:12.3180       Average Loss:  0.12929760466038714
# Iteration:  3600        s:12.3043       Average Loss:  0.13309641423111315
# Iteration:  3700        s:12.3024       Average Loss:  0.11553027778572869
# Iteration:  3800        s:12.3014       Average Loss:  0.10707724917127052
# Iteration:  3900        s:12.2777       Average Loss:  0.13168266994995065
# Iteration:  4000        s:12.2833       Average Loss:  0.08481211595411878
# Iteration:  4100        s:12.2648       Average Loss:  0.11036333292373456
# Epoch time:  126.81171917915344
# Epoch 5
# Iteration:  4200        s:12.2739       Average Loss:  0.09873681203345769
# Iteration:  4300        s:12.3129       Average Loss:  0.10682237564935348
# Iteration:  4400        s:12.2903       Average Loss:  0.0848726556650945
# Iteration:  4500        s:12.2951       Average Loss:  0.10010368582763476
# Iteration:  4600        s:12.2942       Average Loss:  0.06802650743629783
# Iteration:  4700        s:12.2934       Average Loss:  0.08764025370037416
# Iteration:  4800        s:12.2919       Average Loss:  0.05545644681958947
# Iteration:  4900        s:12.3018       Average Loss:  0.053128012903471244
# Iteration:  5000        s:12.3348       Average Loss:  0.06886913090827874
# Iteration:  5100        s:12.3455       Average Loss:  0.07553447937883902
# Epoch time:  126.81300139427185
# Epoch 6
# Iteration:  5200        s:12.2577       Average Loss:  0.07752964375380543
# Iteration:  5300        s:12.3122       Average Loss:  0.04924394679706893
# Iteration:  5400        s:12.4823       Average Loss:  0.07359493983880384
# Iteration:  5500        s:12.4096       Average Loss:  0.09243472051311982
# Iteration:  5600        s:12.3590       Average Loss:  0.09383144193940098
# Iteration:  5700        s:12.3052       Average Loss:  0.09829048601473915
# Iteration:  5800        s:12.3036       Average Loss:  0.09739731382447644
# Iteration:  5900        s:12.3094       Average Loss:  0.07317498849792173
# Iteration:  6000        s:12.3049       Average Loss:  0.09129703806960605
# Iteration:  6100        s:12.3152       Average Loss:  0.09730219142089482
# Epoch time:  127.15173625946045
# Epoch 7
# Iteration:  6200        s:12.2276       Average Loss:  0.0647526060842938
# Iteration:  6300        s:12.3098       Average Loss:  0.06952241562248673
# Iteration:  6400        s:12.2942       Average Loss:  0.049452435646089726
# Iteration:  6500        s:12.3165       Average Loss:  0.047899600587115856
# Iteration:  6600        s:12.3359       Average Loss:  0.04231041098042624
# Iteration:  6700        s:12.3066       Average Loss:  0.055262255520065084
# Iteration:  6800        s:12.2894       Average Loss:  0.0756112389206828
# Iteration:  6900        s:12.2750       Average Loss:  0.055528706653494735
# Iteration:  7000        s:12.3285       Average Loss:  0.0592881380081235
# Iteration:  7100        s:12.2908       Average Loss:  0.042896182980985034
# Iteration:  7200        s:12.3096       Average Loss:  0.05756192940592882
# Epoch time:  126.81766295433044
# Epoch 8
# Iteration:  7300        s:12.2618       Average Loss:  0.05279337810279685
# Iteration:  7400        s:12.2926       Average Loss:  0.04119592405062576
# Iteration:  7500        s:12.3163       Average Loss:  0.04113994478946552
# Iteration:  7600        s:12.3653       Average Loss:  0.06466675529001804
# Iteration:  7700        s:12.2958       Average Loss:  0.04896210243590758
# Iteration:  7800        s:12.3053       Average Loss:  0.06088159863946203
# Iteration:  7900        s:12.1730       Average Loss:  0.03311232233478222
# Iteration:  8000        s:12.1565       Average Loss:  0.04502686794075998
# Iteration:  8100        s:12.2059       Average Loss:  0.04765146984609601
# Iteration:  8200        s:12.1682       Average Loss:  0.05537362982591731
# Epoch time:  126.28331446647644
# Epoch 9
# Iteration:  8300        s:12.1032       Average Loss:  0.020934459320924362
# Iteration:  8400        s:12.1648       Average Loss:  0.03379596251805197
# Iteration:  8500        s:12.1618       Average Loss:  0.03022972049097007
# Iteration:  8600        s:12.1700       Average Loss:  0.038331072518849395
# Iteration:  8700        s:12.1872       Average Loss:  0.052558883530145976
# Iteration:  8800        s:12.1785       Average Loss:  0.057114760108233895
# Iteration:  8900        s:12.1751       Average Loss:  0.028279207621817477
# Iteration:  9000        s:12.2150       Average Loss:  0.051862848754026344
# Iteration:  9100        s:12.1680       Average Loss:  0.04272945125092519
# Iteration:  9200        s:12.1743       Average Loss:  0.029267400150456525
# Epoch time:  125.45627903938293
# Epoch 10
# Iteration:  9300        s:12.0968       Average Loss:  0.048146289743672244
# Iteration:  9400        s:12.1839       Average Loss:  0.02968601358035812
# Iteration:  9500        s:12.1646       Average Loss:  0.03294695198725094
# Iteration:  9600        s:12.1749       Average Loss:  0.047611603500954516
# Iteration:  9700        s:12.1453       Average Loss:  0.05157498244770977
# Iteration:  9800        s:12.1525       Average Loss:  0.04695062369974039
# Iteration:  9900        s:12.1609       Average Loss:  0.035719475503574356
# Iteration:  10000       s:12.2023       Average Loss:  0.043273556205167554
# Iteration:  10100       s:12.1517       Average Loss:  0.03734559972308489
# Iteration:  10200       s:12.1455       Average Loss:  0.04095874754839315
# Iteration:  10300       s:12.1463       Average Loss:  0.0414955158531302
# Epoch time:  125.33947777748108
# [saved] after_dial -> weights/
# === Stage 2: hands pretrain ===
# Training  for 10 epoch(s)
# Epoch 1
# Iteration:  100         s:21.8279       Average Loss:  4.871247878074646
# Iteration:  200         s:22.0074       Average Loss:  3.792937454581261
# Iteration:  300         s:22.0301       Average Loss:  2.2358438447117805
# Iteration:  400         s:22.1027       Average Loss:  1.5410690131038427
# Iteration:  500         s:22.0353       Average Loss:  1.2849774558097125
# Iteration:  600         s:22.0346       Average Loss:  1.0314877497404813
# Iteration:  700         s:22.0298       Average Loss:  0.9956052562221884
# Iteration:  800         s:22.0331       Average Loss:  0.9447463393583894
# Iteration:  900         s:22.0101       Average Loss:  0.9490890617668629
# Iteration:  1000        s:22.0121       Average Loss:  0.8313125491142273
# Epoch time:  227.03587675094604
# Epoch 2
# Iteration:  1100        s:21.8484       Average Loss:  0.7210253721475601
# Iteration:  1200        s:21.9917       Average Loss:  0.7665545176900923
# Iteration:  1300        s:22.2074       Average Loss:  0.7260002382565289
# Iteration:  1400        s:21.9753       Average Loss:  0.6873670604638755
# Iteration:  1500        s:21.9533       Average Loss:  0.7528618803340942
# Iteration:  1600        s:21.9346       Average Loss:  0.7304276859760285
# Iteration:  1700        s:21.9746       Average Loss:  0.6626950171776116
# Iteration:  1800        s:21.9679       Average Loss:  0.6308848836645484
# Iteration:  1900        s:22.1004       Average Loss:  0.5628983044903726
# Iteration:  2000        s:21.9815       Average Loss:  0.6335205740481615
# Epoch time:  226.74337100982666
# Epoch 3
# Iteration:  2100        s:21.8673       Average Loss:  0.5570838964171707
# Iteration:  2200        s:22.0044       Average Loss:  0.5105724831111729
# Iteration:  2300        s:22.0009       Average Loss:  0.5382858805079013
# Iteration:  2400        s:22.0051       Average Loss:  0.4636361847165972
# Iteration:  2500        s:22.0082       Average Loss:  0.5436797537794337
# Iteration:  2600        s:21.9791       Average Loss:  0.5333515589218587
# Iteration:  2700        s:21.9957       Average Loss:  0.5214985033310949
# Iteration:  2800        s:22.0474       Average Loss:  0.5236842309357599
# Iteration:  2900        s:22.0393       Average Loss:  0.5477282986836508
# Iteration:  3000        s:21.9748       Average Loss:  0.5707148905470967
# Epoch time:  226.66356229782104
# Epoch 4
# Iteration:  3100        s:21.7868       Average Loss:  0.5125267196074128
# Iteration:  3200        s:22.0302       Average Loss:  0.4302526714699343
# Iteration:  3300        s:21.9937       Average Loss:  0.4298795850155875
# Iteration:  3400        s:22.0689       Average Loss:  0.46960981383919714
# Iteration:  3500        s:22.0160       Average Loss:  0.4014309411821887
# Iteration:  3600        s:22.0103       Average Loss:  0.46118632030906154
# Iteration:  3700        s:21.9813       Average Loss:  0.4402459097607061
# Iteration:  3800        s:22.0134       Average Loss:  0.49126854436472056
# Iteration:  3900        s:22.0309       Average Loss:  0.41634716173401104
# Iteration:  4000        s:21.9930       Average Loss:  0.4178155492199585
# Iteration:  4100        s:22.1011       Average Loss:  0.4109425492794253
# Epoch time:  226.95966362953186
# Epoch 5
# Iteration:  4200        s:21.8924       Average Loss:  0.35027613912010563
# Iteration:  4300        s:21.9865       Average Loss:  0.29845320707652717
# Iteration:  4400        s:22.0085       Average Loss:  0.3472173188207671
# Iteration:  4500        s:22.9237       Average Loss:  0.4177396588306874
# Iteration:  4600        s:22.3767       Average Loss:  0.3639322116784751
# Iteration:  4700        s:22.3108       Average Loss:  0.394833305447828
# Iteration:  4800        s:22.2715       Average Loss:  0.42370584519347176
# Iteration:  4900        s:22.2586       Average Loss:  0.30868261885363607
# Iteration:  5000        s:22.2979       Average Loss:  0.3986519536585547
# Iteration:  5100        s:22.2725       Average Loss:  0.3720091887214221
# Epoch time:  229.5388045310974
# Epoch 6
# Iteration:  5200        s:22.1502       Average Loss:  0.3548469302523881
# Iteration:  5300        s:22.2741       Average Loss:  0.2639233655645512
# Iteration:  5400        s:22.3189       Average Loss:  0.3628826129226945
# Iteration:  5500        s:22.2907       Average Loss:  0.340998045981396
# Iteration:  5600        s:22.2383       Average Loss:  0.27869193722959607
# Iteration:  5700        s:22.2834       Average Loss:  0.2748339677043259
# Iteration:  5800        s:22.3041       Average Loss:  0.2794326115900185
# Iteration:  5900        s:22.2254       Average Loss:  0.2810757161688525
# Iteration:  6000        s:22.2540       Average Loss:  0.2946718664269429
# Iteration:  6100        s:22.3095       Average Loss:  0.37103722042171283
# Epoch time:  229.57602977752686
# Epoch 7
# Iteration:  6200        s:22.1356       Average Loss:  0.3420111286302563
# Iteration:  6300        s:22.2716       Average Loss:  0.21445877972524613
# Iteration:  6400        s:22.4084       Average Loss:  0.2414134906930849
# Iteration:  6500        s:22.2192       Average Loss:  0.2462127706129104
# Iteration:  6600        s:22.3135       Average Loss:  0.2532858745928388
# Iteration:  6700        s:22.2686       Average Loss:  0.2531253383809235
# Iteration:  6800        s:22.2631       Average Loss:  0.23505295592942274
# Iteration:  6900        s:22.3105       Average Loss:  0.26369253160897643
# Iteration:  7000        s:22.2527       Average Loss:  0.25216449742903935
# Iteration:  7100        s:22.2676       Average Loss:  0.2511390157265123
# Iteration:  7200        s:22.2892       Average Loss:  0.31592945350683294
# Epoch time:  229.61107110977173
# Epoch 8
# Iteration:  7300        s:22.1465       Average Loss:  0.17404163120198063
# Iteration:  7400        s:22.2389       Average Loss:  0.23269395798211917
# Iteration:  7500        s:22.3063       Average Loss:  0.15953556213295086
# Iteration:  7600        s:22.3374       Average Loss:  0.2012022107222583
# Iteration:  7700        s:21.9860       Average Loss:  0.1867074642353691
# Iteration:  7800        s:22.0291       Average Loss:  0.23086614584433846
# Iteration:  7900        s:21.9993       Average Loss:  0.22187708999263123
# Iteration:  8000        s:22.0189       Average Loss:  0.20890871409676037
# Iteration:  8100        s:21.9962       Average Loss:  0.2685255859210156
# Iteration:  8200        s:21.9969       Average Loss:  0.26960397135815584
# Epoch time:  227.82261729240417
# Epoch 9
# Iteration:  8300        s:21.8808       Average Loss:  0.15449543646012898
# Iteration:  8400        s:22.0029       Average Loss:  0.1080425112217199
# Iteration:  8500        s:22.0117       Average Loss:  0.16526074299472385
# Iteration:  8600        s:21.9819       Average Loss:  0.18383960076665973
# Iteration:  8700        s:22.0825       Average Loss:  0.16021071991184727
# Iteration:  8800        s:22.0076       Average Loss:  0.18020036548259669
# Iteration:  8900        s:22.6204       Average Loss:  0.2061184271343518
# Iteration:  9000        s:22.2989       Average Loss:  0.1972598459350411
# Iteration:  9100        s:22.2408       Average Loss:  0.24944427283480763
# Iteration:  9200        s:22.2822       Average Loss:  0.17744315739837474
# Epoch time:  228.43637204170227
# Epoch 10
# Iteration:  9300        s:22.1239       Average Loss:  0.22700181619205978
# Iteration:  9400        s:22.2582       Average Loss:  0.11596272522467188
# Iteration:  9500        s:22.3004       Average Loss:  0.1047351374826394
# Iteration:  9600        s:22.2877       Average Loss:  0.13031336022133475
# Iteration:  9700        s:22.2498       Average Loss:  0.1673341518867528
# Iteration:  9800        s:22.2681       Average Loss:  0.14503494084754492
# Iteration:  9900        s:22.3085       Average Loss:  0.14664581056451426
# Iteration:  10000       s:22.2793       Average Loss:  0.1973245738859987
# Iteration:  10100       s:22.3035       Average Loss:  0.16583573120296932
# Iteration:  10200       s:22.2325       Average Loss:  0.17127514856285417
# Iteration:  10300       s:22.2702       Average Loss:  0.16136948755476624
# Epoch time:  229.52293944358826
# [saved] after_hands -> weights/
# === Stage 3: time finetune (integrated constraints) ===
# Training  for 10 epoch(s)
# Epoch 1
# Iteration:  100         s:351.4028      Average Loss:  0.5956122138048522
# Iteration:  200         s:351.0450      Average Loss:  0.30398821756592953
# Iteration:  300         s:351.5426      Average Loss:  0.19978371614124626
# Iteration:  400         s:351.1396      Average Loss:  0.1790921819017967
# Iteration:  500         s:351.2704      Average Loss:  0.1503791984298732
# Iteration:  600         s:351.0612      Average Loss:  0.10097538311150857
# Iteration:  700         s:350.9717      Average Loss:  0.06978370326571166
# Iteration:  800         s:350.6975      Average Loss:  0.0724368971871445
# Iteration:  900         s:350.9967      Average Loss:  0.03617307170468848
# Iteration:  1000        s:350.2345      Average Loss:  0.030320433957967907
# Epoch time:  3620.90612244606
# Epoch 2
# Iteration:  1100        s:350.2213      Average Loss:  0.024664879223855678
# Iteration:  1200        s:352.2292      Average Loss:  0.02575487370515475
# Iteration:  1300        s:348.7906      Average Loss:  0.012959663251531311
# Iteration:  1400        s:350.3314      Average Loss:  0.011461554484558292
# Iteration:  1500        s:353.0961      Average Loss:  0.007631397615768947
# Iteration:  1600        s:354.4942      Average Loss:  0.01655855873628752
# Iteration:  1700        s:352.9186      Average Loss:  0.009889538385905325
# Iteration:  1800        s:352.6634      Average Loss:  0.005811003847629764
# Iteration:  1900        s:351.8488      Average Loss:  0.00477022429346107
# Iteration:  2000        s:350.2170      Average Loss:  0.008045339609961956
# Epoch time:  3626.1432621479034
# Epoch 3
# Iteration:  2100        s:349.9362      Average Loss:  0.009856183149095159


