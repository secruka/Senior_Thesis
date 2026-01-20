from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset


def parse_label_to_time(label: str) -> Tuple[int, int]:
    """Convert label like '1_05' to (hour, minute)."""
    h_str, m_str = label.split("_")
    return int(h_str), int(m_str)


@dataclass
class ClassMap:
    """Mapping between class index and (label, hour, minute)."""

    idx_to_label: List[str]

    def idx_to_time(self, idx: int) -> Tuple[int, int]:
        return parse_label_to_time(self.idx_to_label[idx])

    def idx_to_timestr(self, idx: int) -> str:
        h, m = self.idx_to_time(idx)
        return f"{h:02d}:{m:02d}"


class ClockTimeDataset(Dataset):
    def __init__(
        self,
        csv_path: str | Path,
        data_root: str | Path,
        split: str,
        transform: Optional[Callable] = None,
    ):
        """Dataset for the Kaggle clock dataset using clocks.csv.

        Args:
            csv_path: path to clocks.csv
            data_root: root that contains train/valid/test folders
            split: one of {'train','valid','test'} (matches CSV 'data set')
            transform: torchvision transforms applied to PIL image
        """
        self.csv_path = Path(csv_path)
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform

        df = pd.read_csv(self.csv_path)
        # normalize split names to match CSV exactly
        df_split = df[df["data set"].astype(str).str.lower() == split.lower()].copy()
        if df_split.empty:
            raise ValueError(
                f"Split '{split}' not found in CSV column 'data set'. "
                f"Available: {sorted(df['data set'].unique())}"
            )

        # Store paths and labels
        self.rel_paths: List[str] = df_split["filepaths"].tolist()
        self.targets: List[int] = df_split["class index"].astype(int).tolist()
        self.labels: List[str] = df_split["labels"].astype(str).tolist()

        # Build a stable class map from the whole CSV (0..143)
        df_all = df[["class index", "labels"]].drop_duplicates().sort_values("class index")
        max_idx = int(df_all["class index"].max())
        idx_to_label = [""] * (max_idx + 1)
        for idx, lab in zip(df_all["class index"].astype(int), df_all["labels"].astype(str)):
            idx_to_label[int(idx)] = lab
        if any(l == "" for l in idx_to_label):
            missing = [i for i, l in enumerate(idx_to_label) if l == ""]
            raise ValueError(f"Missing labels for class indices: {missing[:10]}...")
        self.class_map = ClassMap(idx_to_label=idx_to_label)

    def __len__(self) -> int:
        return len(self.rel_paths)

    def __getitem__(self, i: int):
        rel = self.rel_paths[i]
        img_path = self.data_root / rel
        y = self.targets[i]

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(y, dtype=torch.long)
