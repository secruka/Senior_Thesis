"""model_hands.py

目的:
- rotation.csv (または同等のCSV) から、画像内座標系での
  長針・短針の角度(30度刻み=12分類)を教師ありで学習する。

前提CSVの想定カラム例:
file,time_h,time_m,cx,cy,minute_x,minute_y,hour_x,hour_y,rotation_deg,status

このファイルでは **status == 'ok'** の行のみ使用する。
学習ラベルは「時刻」ではなく「角度クラス(0..11)」:
- 0: 12時方向(真上)
- 3: 3時方向(右)
- 6: 6時方向(下)
- 9: 9時方向(左)

注意:
- Prolog側の述語名は従来の time/3 ではなく hands/3 にしています。
  既存の clock.pl が time/3 を想定している場合は、合わせて修正してください。
"""

import math
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


# ------------------------------
# 1) CNN (12-class) for each hand
# ------------------------------
class ClockNet(nn.Module):
    def __init__(self, num_classes: int = 12):
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
# 2) Torch Dataset: CSV -> (img, h_cls, m_cls)
# -----------------------------------------
class ClockHandsCsvTorchDataset(torch.utils.data.Dataset):
    """rotation.csv から角度クラス(12分類)を生成して返すDataset。

    - subset は file 列の先頭 ("train/", "test/", "valid/") でフィルタ。
    - status == 'ok' のみ使用。
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

        # 必須列チェック
        required_cols = [
            "file",
            "cx",
            "cy",
            "minute_x",
            "minute_y",
            "hour_x",
            "hour_y",
            "status",
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(
                "CSV is missing required columns: "
                + ", ".join(missing)
                + f"\nCSV path: {self.csv_path}"
            )

        # status == ok のみ
        df = df[df["status"].astype(str) == "ok"].copy()

        # subset フィルタ: file が "train/..." のように入っている想定
        prefix = self.subset + "/"
        df = df[df["file"].astype(str).str.startswith(prefix)].copy()

        # 欠損/非数の行を落とす（no_hour / fail 等を欠損扱いする意図に合わせる）
        coord_cols = ["cx", "cy", "minute_x", "minute_y", "hour_x", "hour_y"]
        for c in coord_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=coord_cols + ["file"]).reset_index(drop=True)

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
        # 予期しない値でも file プレフィックスにそのまま使えるよう返す
        return s

    @staticmethod
    def _angle_deg_clockwise_from_12(dx: float, dy: float) -> float:
        """画像座標系 (x→右, y→下) で、12時方向を0°として時計回りの角度[0,360) を返す。"""
        # 12時方向(上)を基準にするため、atan2(dx, -dy)
        deg = math.degrees(math.atan2(dx, -dy)) % 360.0
        return deg

    @classmethod
    def _angle_cls_12(cls, hand_x: float, hand_y: float, cx: float, cy: float) -> int:
        dx = float(hand_x) - float(cx)
        dy = float(hand_y) - float(cy)
        deg = cls._angle_deg_clockwise_from_12(dx, dy)
        # 30度刻みの最近傍（±15度以内）
        k = int((deg + 15.0) // 30.0) % 12
        return k

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

        cx, cy = float(row["cx"]), float(row["cy"])
        mx, my = float(row["minute_x"]), float(row["minute_y"])
        hx, hy = float(row["hour_x"]), float(row["hour_y"])

        m_cls = self._angle_cls_12(mx, my, cx, cy)
        h_cls = self._angle_cls_12(hx, hy, cx, cy)

        return img, h_cls, m_cls


# -----------------------------------------
# 3) DeepProbLog Dataset wrapper
# -----------------------------------------
class DeepProbLogDataset(Dataset):
    def __init__(self, pytorch_dataset: torch.utils.data.Dataset):
        self.dataset = pytorch_dataset
        self._query_cache = {}  # Lazy-load queries

    def to_query(self, i):
        if i not in self._query_cache:
            img, h_cls, m_cls = self.dataset[i]

            # クエリ: hands(X, Hcls, Mcls).
            q_term = Term("hands", Var("X"), Constant(int(h_cls)), Constant(int(m_cls)))

            # 「変数 X の中身は 画像データ(tensor) です」という辞書
            substitution = {Var("X"): Constant(img)}

            self._query_cache[i] = Query(q_term, substitution)
        return self._query_cache[i]

    def __len__(self):
        return len(self.dataset)


# -----------------------------------------
# 4) Train entry point
# -----------------------------------------
def main():
    print("Setting up the model (hands angle classification)...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2つのネット: hour / minute を別々に 12分類
    cnn_hour = ClockNet(num_classes=12).to(device)
    cnn_minute = ClockNet(num_classes=12).to(device)

    net_h = Network(cnn_hour, "net_hour", batching=True)
    net_m = Network(cnn_minute, "net_minute", batching=True)

    net_h.optimizer = torch.optim.Adam(cnn_hour.parameters(), lr=1e-4)
    net_m.optimizer = torch.optim.Adam(cnn_minute.parameters(), lr=1e-4)

    # Prolog: models/clock.pl 内で hands/3 を定義しておくこと
    model = Model("models/clock.pl", [net_h, net_m])
    model.set_engine(ExactEngine(model))

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # データ設定
    data_path = "clock_kaggle"  # 例: clock_kaggle/train/... が存在する想定
    csv_name = "rotation.csv"   # ここを rotations.csv 等に合わせて変更
    csv_path = os.path.join(data_path, csv_name)

    # Dataset（必要なら max_samples を外してフルで学習）
    pt_train = ClockHandsCsvTorchDataset(
        root_dir=data_path,
        csv_path=csv_path,
        subset="train",
        transform=transform,
        max_samples=1000,
    )
    pt_test = ClockHandsCsvTorchDataset(
        root_dir=data_path,
        csv_path=csv_path,
        subset="test",
        transform=transform,
        max_samples=500,
    )

    train_dataset = DeepProbLogDataset(pt_train)
    test_dataset = DeepProbLogDataset(pt_test)

    train_loader = DataLoader(train_dataset, batch_size=4)
    test_loader = DataLoader(test_dataset, batch_size=4)

    print(f"Start training with {len(train_dataset)} examples...")
    if len(train_loader) == 0:
        print("Error: Training data not found. Please check the data path / csv path.")
        return

    train_model(
        model,
        train_loader,
        10,
        test_iter=test_loader,
        log_iter=100,
        profile=0,
    )

    torch.save(cnn_hour.state_dict(), "hour_model.pth")
    torch.save(cnn_minute.state_dict(), "minute_model.pth")
    print("Training finished and models saved.")


if __name__ == "__main__":
    main()
