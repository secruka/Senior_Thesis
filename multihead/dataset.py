"""
dataset.py
==========
Dataset classes for the multi-head clock recognition model.

Classes:
  - ClockDataset: PyTorch Dataset loading annotations.csv
  - DPBComponentDataset: DeepProbLog Dataset for Stage 2 (component training)
  - DPBTimeDataset: DeepProbLog Dataset for Stage 3 (time integration)
  - LatentClockDataset: Lightweight PyTorch Dataset from folder structure (no annotations.csv)
  - DPBLatentTimeDataset: DeepProbLog Dataset for latent training (time labels only)
"""

import math
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from PIL import Image

import torch
from torchvision import transforms

from deepproblog.dataset import Dataset
from deepproblog.query import Query
from problog.logic import Term, Constant, Var


# ---------------------------------------------------------------------------
# Image transforms
# ---------------------------------------------------------------------------

def get_transform(train: bool = True):
    """Standard ImageNet-normalised transforms for 224x224 clock images."""
    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.0),  # no flip for clocks
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])


# ---------------------------------------------------------------------------
# Helper: compute angle class from keypoint coordinates
# ---------------------------------------------------------------------------

def angle_deg_cw_from_12(dx: float, dy: float) -> float:
    """Angle in degrees clockwise from 12 o'clock direction.

    Image coords: x->right, y->down.  12 o'clock = -Y direction.
    """
    return math.degrees(math.atan2(dx, -dy)) % 360.0


def angle_to_cls12(hand_x: float, hand_y: float, cx: float, cy: float) -> int:
    """Convert hand tip coordinates to 12-class angle bin (30-degree bins).

    Returns class 0..11 where 0=12 o'clock, 3=3 o'clock, etc.
    """
    dx = hand_x - cx
    dy = hand_y - cy
    deg = angle_deg_cw_from_12(dx, dy)
    return int((deg + 15.0) // 30.0) % 12


# ---------------------------------------------------------------------------
# Labels dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClockLabels:
    time_h: int         # 1-12
    time_m: int         # 0-55 (5-min steps)
    rot_cls: int        # 0-3
    hour_img_cls: int   # 0-11 (hour hand angle in image coords)
    minute_img_cls: int # 0-11 (minute hand angle in image coords)
    class_index: int    # 0-143 (time class for warmup)


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class ClockDataset(torch.utils.data.Dataset):
    """PyTorch Dataset loading annotations.csv.

    Each item returns (image_tensor, ClockLabels).
    """

    def __init__(
        self,
        data_root: str,
        annotations_csv: str,
        subset: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.data_root = data_root
        self.transform = transform or get_transform(train=(subset == "train"))

        df = pd.read_csv(annotations_csv)

        # Filter by subset (train/valid/test) based on file path prefix
        subset = self._normalize_subset(subset)
        prefix = subset + "/"
        df = df[df["file"].astype(str).str.startswith(prefix)].copy()
        df = df.reset_index(drop=True)

        if max_samples is not None:
            df = df.iloc[:int(max_samples)].reset_index(drop=True)

        self.df = df

    @staticmethod
    def _normalize_subset(subset: str) -> str:
        s = (subset or "train").strip().lower()
        if s in {"val", "valid", "validation"}:
            return "valid"
        return s

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        rel_path = str(row["file"])
        img_path = os.path.join(self.data_root, rel_path)
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        time_h = int(row["time_h"])
        time_m = int(row["time_m"])
        rot_cls = int(row["rot_cls"])

        # Compute angle classes from keypoint coordinates
        cx, cy = float(row["cx"]), float(row["cy"])
        hour_img_cls = angle_to_cls12(
            float(row["hour_x"]), float(row["hour_y"]), cx, cy
        )
        minute_img_cls = angle_to_cls12(
            float(row["minute_x"]), float(row["minute_y"]), cx, cy
        )

        # Time class index for warmup (0-143)
        class_index = (time_h % 12) * 12 + time_m // 5

        labels = ClockLabels(
            time_h=time_h,
            time_m=time_m,
            rot_cls=rot_cls,
            hour_img_cls=hour_img_cls,
            minute_img_cls=minute_img_cls,
            class_index=class_index,
        )
        return img, labels


# ---------------------------------------------------------------------------
# DeepProbLog Datasets
# ---------------------------------------------------------------------------

class DPBComponentDataset(Dataset):
    """DeepProbLog Dataset for Stage 2: component(X, R, H, M).

    Trains all three heads simultaneously with ground-truth labels
    derived from the annotations.
    """

    def __init__(self, torch_ds: ClockDataset):
        self.ds = torch_ds
        self.cache = {}

    def __len__(self):
        return len(self.ds)

    def to_query(self, i: int):
        if i not in self.cache:
            img, lab = self.ds[i]
            q = Term(
                "component",
                Var("X"),
                Constant(int(lab.rot_cls)),
                Constant(int(lab.hour_img_cls)),
                Constant(int(lab.minute_img_cls)),
            )
            self.cache[i] = Query(q, {Var("X"): Constant(img)})
        return self.cache[i]


class DPBTimeDataset(Dataset):
    """DeepProbLog Dataset for Stage 3: time(X, Hour, Minute).

    Uses ProbLog's rotation correction and hour-minute constraints
    for integrated training.
    """

    def __init__(self, torch_ds: ClockDataset):
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
                Constant(int(lab.time_h)),
                Constant(int(lab.time_m)),
            )
            self.cache[i] = Query(q, {Var("X"): Constant(img)})
        return self.cache[i]


# ---------------------------------------------------------------------------
# Lightweight dataset for latent-variable training (no annotations.csv)
# ---------------------------------------------------------------------------

@dataclass
class LatentClockLabels:
    """Minimal labels: only time and warmup class index."""
    time_h: int         # 1-12
    time_m: int         # 0-55 (5-min steps)
    class_index: int    # 0-143 (time class for warmup)


def _scan_folder_images(
    data_root: str, subset: str
) -> List[Tuple[str, int, int]]:
    """Scan {data_root}/{subset}/{h}-{m}/ for images.

    Returns list of (image_path, hour, minute).
    """
    subset_dir = os.path.join(data_root, subset)
    entries = []
    for folder_name in sorted(os.listdir(subset_dir)):
        parts = folder_name.split("-")
        if len(parts) != 2:
            continue
        try:
            h, m = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        folder_path = os.path.join(subset_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue
        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                entries.append((os.path.join(folder_path, fname), h, m))
    return entries


class LatentClockDataset(torch.utils.data.Dataset):
    """PyTorch Dataset loading images from folder structure.

    No annotations.csv needed — time labels come from folder names.
    Folder structure: {data_root}/{subset}/{hour}-{minute}/image.jpg
    """

    def __init__(
        self,
        data_root: str,
        subset: str = "train",
        transform=None,
        max_samples: Optional[int] = None,
    ):
        self.transform = transform or get_transform(train=(subset == "train"))
        subset = self._normalize_subset(subset)
        self.entries = _scan_folder_images(data_root, subset)
        if max_samples is not None:
            self.entries = self.entries[:int(max_samples)]

    @staticmethod
    def _normalize_subset(subset: str) -> str:
        s = (subset or "train").strip().lower()
        if s in {"val", "valid", "validation"}:
            return "valid"
        return s

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx: int):
        img_path, time_h, time_m = self.entries[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        labels = LatentClockLabels(
            time_h=time_h,
            time_m=time_m,
            class_index=(time_h % 12) * 12 + time_m // 5,
        )
        return img, labels


class DPBLatentTimeDataset(Dataset):
    """DeepProbLog Dataset for latent training: time(X, Hour, Minute).

    Uses only time labels as supervision. No component labels needed.
    Works with either LatentClockDataset or ClockDataset.
    """

    def __init__(self, torch_ds):
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
                Constant(int(lab.time_h)),
                Constant(int(lab.time_m)),
            )
            self.cache[i] = Query(q, {Var("X"): Constant(img)})
        return self.cache[i]
