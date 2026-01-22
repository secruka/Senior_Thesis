# dataset_csv.py みたいに別ファイルでOK
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class ClockDatasetCSV(Dataset):
    def __init__(self, root_dir, subset, csv_path, transform=None):
        """
        root_dir: /Users/ruka/Senior_Thesis/clock_kaggle
        subset: "train" or "test"
        csv_path: clocks_with_relpath.csv
        """
        self.subset_root = os.path.join(root_dir, subset)  # .../clock_kaggle/train
        self.transform = transform

        df = pd.read_csv(csv_path)
        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()

        # subsetに存在する画像だけ残す（split列が無くてもOK）
        paths = []
        keep = []
        for i, r in df.iterrows():
            rel = str(r["relpath"])
            p = os.path.join(self.subset_root, rel)
            if os.path.exists(p):
                paths.append(rel)
                keep.append(True)
            else:
                keep.append(False)
        df = df[keep].reset_index(drop=True)
        df["relpath"] = paths

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        relpath = r["relpath"]               # 例: "1-00/0.jpg"
        img_path = os.path.join(self.subset_root, relpath)

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # ラベルはフォルダ名から取る（CSVのtime_h/time_mを使ってもOK）
        time_label = relpath.split(os.sep)[0]   # "1-00"
        h_str, m_str = time_label.split("-")
        h = int(h_str)                          # 1..12
        m = int(m_str)                          # 0,5,10,...55（データがそうなら）

        return img, h, m
