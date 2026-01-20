import os
import math
from dataclasses import dataclass

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset as TorchDataset
from torchvision import models, transforms
from PIL import Image

from deepproblog.train import train_model
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.dataset import DataLoader as DPBDataLoader, Dataset as DPBDataset
from deepproblog.engines import ExactEngine
from deepproblog.query import Query
from problog.logic import Term, Constant, Var


# =========================
# 設定（ここだけ触ればOK）
# =========================
@dataclass
class CFG:
    # clock_kaggle のルート（labels_points.csv と train/ がある場所）
    root_dir: str = "clock_kaggle"
    csv_name: str = "labels_points.csv"

    # dialを「nohands画像」で学習したい場合：
    # 例）clock_kaggle/images_nohands/train/... という構造があるなら "images_nohands"
    # Noneなら clock_kaggle/train/... を使う（file列が train/... を含むため）
    img_base_dir: str | None = None  # "images_nohands" など

    # 角度列候補（CSVに入っている方が自動採用される）
    angle_col_candidates = ("theta12_deg", "rotation_deg")
    only_ok: bool = True

    # 12クラス（0,30,...,330）
    bin_size_deg: int = 30
    labels_deg = (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330)

    # DeepProbLog program
    prolog_file: str = "models/dial.pl"

    # train
    epochs: int = 10
    batch_size: int = 16
    lr: float = 1e-4

CFG = CFG()
# =========================


def norm360(deg: float) -> float:
    return (deg % 360.0 + 360.0) % 360.0


def pick_angle_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"None of angle columns found in CSV: {candidates}")


def deg_to_label_12bin(deg: float, bin_size: int = 30) -> int:
    """
    連続角度(deg)を 0,30,...,330 のどれかに丸める（最近傍bin）
    """
    d = norm360(deg)
    k = int((d + bin_size / 2) // bin_size) % (360 // bin_size)
    return k * bin_size  # 0..330


class DialRotationDatasetCSV(TorchDataset):
    """
    return: (img_tensor, rot_label_deg)
      rot_label_deg は {0,30,...,330} の実値
    """
    def __init__(self, root_dir: str, subset: str, csv_name: str,
                 angle_col: str, only_ok: bool, img_base_dir: str | None,
                 transform=None, bin_size_deg: int = 30):
        self.root_dir = root_dir
        self.subset = subset
        self.transform = transform
        self.bin_size_deg = bin_size_deg

        csv_path = os.path.join(root_dir, csv_name)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if only_ok and "status" in df.columns:
            df = df[df["status"].astype(str) == "ok"].copy()

        if "file" not in df.columns:
            raise ValueError("CSV must have 'file' column like 'train/1-00/0.jpg'.")

        if angle_col not in df.columns:
            raise ValueError(f"CSV must have angle column '{angle_col}'.")

        # subsetの行だけ（file が "train/..." の形式なので先頭一致でフィルタ）
        prefix = f"{subset}/"
        df = df[df["file"].astype(str).str.startswith(prefix)].copy()
        df = df.dropna(subset=["file", angle_col]).reset_index(drop=True)

        # 画像ベースディレクトリ
        # img_base_dir が相対なら root_dir からの相対として解釈
        if img_base_dir is None:
            img_base = root_dir
        else:
            img_base = img_base_dir
            if not os.path.isabs(img_base):
                img_base = os.path.join(root_dir, img_base)

        self.samples = []
        for _, r in df.iterrows():
            rel = str(r["file"]).strip()  # 例: "train/1-00/0.jpg"
            img_path = os.path.join(img_base, rel)
            if not os.path.exists(img_path):
                continue

            deg = float(r[angle_col])
            rot_label = deg_to_label_12bin(deg, bin_size=bin_size_deg)  # 0..330
            self.samples.append((img_path, int(rot_label)))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples found for subset='{subset}'.\n"
                f"Check:\n"
                f"  csv={csv_path}\n"
                f"  img_base={img_base}\n"
                f"  file entries like '{subset}/...'\n"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, rot_label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, rot_label


class DeepProbLogDialDataset(DPBDataset):
    """
    rot(X, R). を返す
    X は substitution で画像テンソルを渡す（hands側と同じ方式）
    """
    def __init__(self, pytorch_dataset):
        self.dataset = pytorch_dataset

    def __len__(self):
        return len(self.dataset)

    def to_query(self, i):
        img, rot_label = self.dataset[i]
        q_term = Term("rot", Var("X"), Constant(int(rot_label)))
        substitution = {Var("X"): Constant(img)}
        return Query(q_term, substitution)


class DialNet(nn.Module):
    """
    12クラス分類（0,30,...,330）
    DeepProbLog nn/4 でそのまま扱える Softmax 出力
    """
    def __init__(self, num_classes=12):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_dim = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_dim, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        # DeepProbLog batching=True だと list[Constant(tensor)] で来るので吸収
        if isinstance(x, list):
            x = torch.stack([item.value for item in x])
        x = x.to(next(self.parameters()).device)
        return self.resnet(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # 角度列の自動決定
    df0 = pd.read_csv(os.path.join(CFG.root_dir, CFG.csv_name))
    angle_col = pick_angle_col(df0, CFG.angle_col_candidates)
    print("using angle column:", angle_col)
    print("labels:", list(CFG.labels_deg))

    # hands側と同じ標準transform
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # Torch dataset
    pt_train = DialRotationDatasetCSV(
        root_dir=CFG.root_dir,
        subset="train",
        csv_name=CFG.csv_name,
        angle_col=angle_col,
        only_ok=CFG.only_ok,
        img_base_dir=CFG.img_base_dir,
        transform=tfm,
        bin_size_deg=CFG.bin_size_deg,
    )
    pt_test = DialRotationDatasetCSV(
        root_dir=CFG.root_dir,
        subset="test",
        csv_name=CFG.csv_name,
        angle_col=angle_col,
        only_ok=CFG.only_ok,
        img_base_dir=CFG.img_base_dir,
        transform=tfm,
        bin_size_deg=CFG.bin_size_deg,
    )

    # DeepProbLog dataset + loader
    train_ds = DeepProbLogDialDataset(pt_train)
    test_ds = DeepProbLogDialDataset(pt_test)

    train_loader = DPBDataLoader(train_ds, batch_size=CFG.batch_size)
    test_loader = DPBDataLoader(test_ds, batch_size=CFG.batch_size)

    print(f"Train examples: {len(train_ds)}, Test examples: {len(test_ds)}")

    # Network + DeepProbLog model
    cnn_rot = DialNet(num_classes=12).to(device)
    net_r = Network(cnn_rot, "net_rot", batching=True)
    net_r.optimizer = torch.optim.Adam(cnn_rot.parameters(), lr=CFG.lr)

    model = Model(CFG.prolog_file, [net_r])
    model.set_engine(ExactEngine(model))

    train_model(
        model,
        train_loader,
        CFG.epochs,
        test_iter=test_loader,
        log_iter=100,
        profile=0
    )

    os.makedirs("dialnet_ckpt", exist_ok=True)
    torch.save(cnn_rot.state_dict(), "dialnet_ckpt/rot_model.pth")
    print("Saved: dialnet_ckpt/rot_model.pth")
    # # 追加：rot用モデル
    # cnn_rot = ClockNet(num_classes=12).to(device)
    # cnn_rot.load_state_dict(torch.load("dialnet_ckpt/rot_model.pth", map_location=device))

    # net_r = Network(cnn_rot, "net_rot", batching=True)
    # net_r.optimizer = torch.optim.Adam(cnn_rot.parameters(), lr=1e-5)  # 微調整なら小さめ推奨

    # # Modelのprologを統合版に変更
    # model = Model("models/clock_integrated.pl", [net_h, net_m, net_r])

if __name__ == "__main__":
    main()
