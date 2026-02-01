import os
from pathlib import Path

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from deepproblog.query import Query
from problog.logic import Term, Constant, Var


class ClockDataset(Dataset):
    def __init__(self, root_dir, subset='train', transform=None, csv_name="labels_points.csv", max_samples=None):
        """
        Args:
            root_dir (str): データセットのルートディレクトリ (例: "clock_kaggle/")
            subset (str): 'train', 'test', 'valid' のいずれか
            transform (callable, optional): 画像の前処理
            csv_name (str): 使うCSVファイル名（デフォルト: labels_points.csv）
            max_samples (int, optional): 最大サンプル数制限（Noneで無制限）
        """
        self.root_dir = Path(root_dir)
        self.subset = subset
        self.transform = transform

        # CSVパス（root_dir 直下にある想定、またはroot_dir の親）
        csv_path = self.root_dir / csv_name
        if not csv_path.exists():
            # root_dir の親を確認
            parent_csv = self.root_dir.parent / csv_name
            if parent_csv.exists():
                csv_path = parent_csv
            else:
                # よくある代替名にも一応対応（必要なければ消してOK）
                alt = self.root_dir / "clocks.csv"
                if alt.exists():
                    csv_path = alt
                else:
                    raise FileNotFoundError(f"CSV not found: {self.root_dir / csv_name} or {parent_csv} (or {alt})")

        df = pd.read_csv(csv_path)

        # status 列があれば ok のみ
        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()

        # subset でフィルタ（file列が "train/..." の形式なのでここで切る）
        if "file" not in df.columns:
            raise ValueError("CSV must have 'file' column like 'train/1-00/0.jpg'.")

        prefix = f"{subset}/"
        df = df[df["file"].astype(str).str.startswith(prefix)].copy()

        # 必須列
        need = ["file", "time_h", "time_m"]
        df = df.dropna(subset=need).reset_index(drop=True)

        # 画像パス解決（root_dir / file）
        # file が既に "train/..." を含むので、root_dir 直下と結合すれば一意に読める
        img_paths = []
        hours = []
        minute_idxs = []

        for _, r in df.iterrows():
            rel = str(r["file"])                 # 例: "train/1-00/0.jpg"
            p = self.root_dir / rel              # 例: root/train/1-00/0.jpg

            if not p.exists():
                # ここで落ちると困るのでスキップ（必要なら raise にしてOK）
                continue

            h = int(r["time_h"])
            m = int(r["time_m"])

            # minute を 1..12 index に変換（00->12, 05->1, ..., 55->11）
            if m == 0:
                m_idx = 12
            else:
                m_idx = m // 5  # 5->1, 10->2, ...

            img_paths.append(p)
            hours.append(h)
            minute_idxs.append(m_idx)
            
            # Limit samples if specified
            if max_samples is not None and len(img_paths) >= max_samples:
                break

        self.images = img_paths
        self.labels = list(zip(hours, minute_idxs))

        if len(self.images) == 0:
            raise RuntimeError(
                f"No images found for subset='{subset}'. "
                f"Check that CSV file paths like '{subset}/...' exist under root_dir."
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        hour, minute_idx = self.labels[idx]

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, hour, minute_idx


def get_clock_data(root_dir, subset='train', csv_name="labels_points.csv"):
    """
    DeepProbLog用のデータリストを作成して返す関数
    戻り値: List[Query]
    クエリ形式: time(X, hour, minute).
    画像テンソルは substitution で Var('X') に渡す（model.py と同じ流儀）
    """

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    dataset = ClockDataset(root_dir, subset=subset, transform=transform, csv_name=csv_name)
    queries = []

    print(f"Loading {subset} data from {root_dir} using {csv_name}...")

    for i in range(len(dataset)):
        img_tensor, h, m = dataset[i]

        q_term = Term('time', Var('X'), Constant(h), Constant(m))
        substitution = {Var('X'): Constant(img_tensor)}
        queries.append(Query(q_term, substitution))

    print(f"Loaded {len(queries)} examples.")
    return queries


if __name__ == "__main__":
    data_path = "/Users/ruka/ResNet_proj/clock_kaggle"

    try:
        train_data = get_clock_data(data_path, subset='train', csv_name="labels_points.csv")
        print(f"Example query: {train_data[0]}")
    except Exception as e:
        print(f"Error: {e}")
        print("データセットのパス / CSVの場所 / CSV内のfileパスを確認してください。")
