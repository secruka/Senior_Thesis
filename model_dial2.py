"""
model_dial.py

目的:
- 12時キーポイント付きCSV（例: dial_keypoints.csv）を使って、
  文字盤の回転（0/90/180/270）を 4分類（rot_cls: 0..3）で学習・推論する。

前提CSV（例: dial_keypoints.csv）の想定カラム:
file,cx,cy,twelve_x,twelve_y,rot_deg,rot_cls
※ status 列があってもOK（status==ok のみ利用）

ここでの出力:
- rot_cls ∈ {0,1,2,3}
  0->0°, 1->90°, 2->180°, 3->270°

注意:
- このモデルは derotation（画像の回転補正）をしない。
  画像内で観測される回転をそのまま4分類するのが役割。
"""

import os
from typing import Optional

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


# =========================
# ここを自分の環境に合わせて編集
# =========================
ROOT_DIR = "clock_kaggle"  # 画像パスのroot（file列が train/... のように相対ならここをrootに）
CSV_PATH = "clock_kaggle/dial_keypoints.csv"  # 12時キーポイント付きCSV

# DeepProbLog 側の Prolog ファイル（あなたの構成に合わせて）
PROLOG_FILE = "models/dial.pl"  # 例: "clock_integrated.pl" などでもOK（dial/2 を含む前提）

# 学習ハイパーパラメータ（必要なら調整）
EPOCHS = 10
LR = 1e-4
BATCH_SIZE = 32
# =========================


# ------------------------------
# 1) CNN (4-class) for dial rotation
# ------------------------------
class DialNet(nn.Module):
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, num_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        # DeepProbLog から Constant(list) が来ることがあるため吸収
        if isinstance(x, list):
            x = torch.stack([item.value for item in x])

        # GPU対応: モデルと同じデバイスに転送
        x = x.to(next(self.parameters()).device)
        return self.resnet(x)


# -----------------------------------------
# 2) Torch Dataset: CSV -> (img, rot_cls)
# -----------------------------------------
class DialCsvTorchDataset(torch.utils.data.Dataset):
    """
    dial_keypoints.csv から rot_cls(0..3) を読み、画像と一緒に返すDataset。

    - subset は file 列の先頭 ("train/", "test/", "valid/") でフィルタ。
    - status 列がある場合は status == 'ok' のみ使用（無ければそのまま）
    """

    def __init__(
        self,
        root_dir: str,
        csv_path: str,
        subset: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.root_dir = root_dir
        self.csv_path = csv_path
        self.subset = self._normalize_subset(subset)
        self.transform = transform

        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # 必須列チェック（最小）
        required_cols = ["file", "rot_cls"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(missing)
                + f"\nCSV path: {self.csv_path}"
            )

        # status列があれば ok のみ
        if "status" in df.columns:
            df = df[df["status"].astype(str).str.lower().eq("ok")].copy()

        # subset フィルタ: file が "train/..." のように入っている想定
        prefix = self.subset + "/"
        df = df[df["file"].astype(str).str.startswith(prefix)].copy()

        # rot_cls を数値化 & 欠損排除
        df["rot_cls"] = pd.to_numeric(df["rot_cls"], errors="coerce")
        df = df.dropna(subset=["file", "rot_cls"]).reset_index(drop=True)

        # rot_cls の範囲チェック（0..3）
        df = df[(df["rot_cls"] >= 0) & (df["rot_cls"] <= 3)].reset_index(drop=True)

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

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        rel_path = str(row["file"])
        img_path = os.path.join(self.root_dir, rel_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        rot_cls = int(row["rot_cls"])
        return img, rot_cls


# -----------------------------------------
# 3) DeepProbLog Dataset wrapper
# -----------------------------------------
class DeepProbLogDataset(Dataset):
    """
    Query: dial(X, RotCls).
    - X は画像テンソル
    - RotCls は 0..3 の定数
    """

    def __init__(self, pytorch_dataset: torch.utils.data.Dataset):
        self.dataset = pytorch_dataset
        self._query_cache = {}

    def to_query(self, i):
        if i not in self._query_cache:
            img, rot_cls = self.dataset[i]

            # クエリ: dial(X, RotCls).
            q_term = Term("dial", Var("X"), Constant(int(rot_cls)))
            substitution = {Var("X"): Constant(img)}

            self._query_cache[i] = Query(q_term, substitution)
        return self._query_cache[i]

    def __len__(self):
        return len(self.dataset)


# -----------------------------------------
# 4) Train entry point
# -----------------------------------------
def main():
    print("Setting up the model (dial rotation classification: 4 classes)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn_dial = DialNet(num_classes=4).to(device)
    net_dial = Network(cnn_dial, "net_dial", batching=True)

    # DeepProbLog model
    model = Model(PROLOG_FILE, [net_dial])
    model.set_engine(ExactEngine(model))
    
    # Set optimizer on the network
    net_dial.optimizer = torch.optim.Adam(cnn_dial.parameters(), lr=LR)

    # Transforms（必要最小限）
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
    )

    # Torch datasets
    train_torch = DialCsvTorchDataset(ROOT_DIR, CSV_PATH, subset="train", transform=transform)
    valid_torch = DialCsvTorchDataset(ROOT_DIR, CSV_PATH, subset="valid", transform=transform)
    test_torch  = DialCsvTorchDataset(ROOT_DIR, CSV_PATH, subset="test",  transform=transform)

    # DeepProbLog datasets
    train_data = DeepProbLogDataset(train_torch)
    valid_data = DeepProbLogDataset(valid_torch)
    test_data  = DeepProbLogDataset(test_torch)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=BATCH_SIZE, shuffle=False)

    # Train
    print("Start training...")
    train_model(
        model,
        train_loader,
        EPOCHS,
        test_iter=valid_loader,
        log_iter=100,
    )

    # Save weights
    save_path = "model_dial_rotcls.pth"
    torch.save(cnn_dial.state_dict(), save_path)
    print(f"[OK] saved weights: {save_path}")

    # (任意) テスト評価を回したい場合は、あなたの評価関数に合わせて追加してください
    print("Done.")


if __name__ == "__main__":
    main()
