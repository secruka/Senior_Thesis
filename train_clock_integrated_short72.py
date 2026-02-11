#!/usr/bin/env python3
"""
train_clock_integrated_short72.py

(雛形) 既存の train_clock_integrated_heatmap.py を“流用”しつつ、
短針を 72分類（5°刻み）にした版。

- minute_img: 12分類（5分刻み） …元のまま
- hour_img:   72分類（5°刻み）  …ここだけ変更
- dial:       4分類 …元のまま

使い方例:
  python train_clock_integrated_short72.py \
    --data_root clock_kaggle \
    --csv rotations_new.csv \
    --prolog models/clock_integrated_72.pl \
    --run pretrain_and_finetune \
    --dial_arch heatmap \
    --epochs 5 --batch_size 8 --lr_hands 1e-4 --lr_dial 1e-4

注意:
  - このスクリプトは train_clock_integrated_heatmap.py と同じフォルダに置く（importするため）
  - Prolog 側も 72対応版を使う（models/clock_integrated_72.pl）
"""

import os
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import pandas as pd

# 元コードを流用
import train_clock_integrated_heatmap as base


def angle_cls_72(hand_x: float, hand_y: float, cx: float, cy: float) -> int:
    """(hand_x,hand_y) から 5°刻み最近傍クラス (0..71)"""
    dx = float(hand_x) - float(cx)
    dy = float(hand_y) - float(cy)
    deg = base.angle_deg_clockwise_from_12(dx, dy)  # 0..360 (12時=0°, 時計回り)
    return int((deg + 2.5) // 5.0) % 72


class RotationCsvTorchDataset72(base.RotationCsvTorchDataset):
    """
    base.RotationCsvTorchDataset を流用しつつ、
    h_img_cls（短針）だけ 72分類に置き換える。
    """

    def _get_labels(self, row) -> base.RowLabels:
        hour_time = int(row["time_h"])
        minute_time = int(row["time_m"])
        rot_cls = base.rotation_deg_to_cls(row["rotation_deg"])

        cx, cy = float(row["cx"]), float(row["cy"])
        mx, my = float(row["minute_x"]), float(row["minute_y"])
        hx, hy = float(row["hour_x"]), float(row["hour_y"])

        # minute は従来通り 12-class
        m_img_cls = base.angle_cls_12(mx, my, cx, cy)
        # hour(=short-hand) を 72-class に変更
        h_img_cls = angle_cls_72(hx, hy, cx, cy)

        twelve_x = None
        twelve_y = None
        if getattr(self, "kp12_map", None) is not None:
            key = str(row["file"]) if isinstance(row, dict) else str(row["file"])
            if key in self.kp12_map:
                twelve_x, twelve_y = self.kp12_map[key]

        return base.RowLabels(
            hour_time=hour_time,
            minute_time=minute_time,
            rot_cls=rot_cls,
            h_img_cls=h_img_cls,
            m_img_cls=m_img_cls,
            twelve_x=twelve_x,
            twelve_y=twelve_y,
        )


def build_deepproblog_model_short72(
    prolog_file: str,
    device: torch.device,
    lr_dial: float,
    lr_hands: float,
    dial_arch: str = "heatmap",
    dial_heatmap_size: int = 56,
    dial_softargmax_temp: float = 1.0,
    dial_angle_softmax_k: float = 8.0,
    weights_dial: Optional[str] = None,
    weights_hour: Optional[str] = None,
    weights_minute: Optional[str] = None,
):
    """base.build_deepproblog_model を流用し、cnn_hour の出力だけ 72 にする"""
    dial_arch = (dial_arch or "heatmap").strip().lower()
    if dial_arch in {"heatmap", "kp", "keypoint", "hm"}:
        cnn_dial = base.DialHeatmapRotationNet(
            heatmap_size=dial_heatmap_size,
            softargmax_temp=dial_softargmax_temp,
            angle_softmax_k=dial_angle_softmax_k,
            use_imagenet=True,
            in_size=224,
            center_xy=(112.0, 112.0),
        ).to(device)
    else:
        cnn_dial = base.ResNetClassifier(num_classes=4).to(device)

    cnn_hour = base.ResNetClassifier(num_classes=72).to(device)   # ★ここだけ違う
    cnn_minute = base.ResNetClassifier(num_classes=12).to(device)

    if weights_dial:
        cnn_dial.load_state_dict(torch.load(weights_dial, map_location=device))
    if weights_hour:
        cnn_hour.load_state_dict(torch.load(weights_hour, map_location=device))
    if weights_minute:
        cnn_minute.load_state_dict(torch.load(weights_minute, map_location=device))

    net_dial = base.Network(cnn_dial, "net_dial", batching=True)
    net_hour = base.Network(cnn_hour, "net_hour", batching=True)
    net_minute = base.Network(cnn_minute, "net_minute", batching=True)

    net_dial.optimizer = torch.optim.Adam(cnn_dial.parameters(), lr=lr_dial)
    net_hour.optimizer = torch.optim.Adam(cnn_hour.parameters(), lr=lr_hands)
    net_minute.optimizer = torch.optim.Adam(cnn_minute.parameters(), lr=lr_hands)

    model = base.Model(prolog_file, [net_dial, net_hour, net_minute])
    model.set_engine(base.ExactEngine(model))
    return model, cnn_dial, cnn_hour, cnn_minute

def save_stage_weights(args, cnn_dial, cnn_hour, cnn_minute, stage: str):
    """Save model weights under args.save_dir with an 'after_<stage>_' prefix.
    This creates stage-wise snapshots so you can resume/inspect after each phase
    (e.g., dial -> hands -> time).
        """
    os.makedirs(args.save_dir, exist_ok=True)
    prefix = f"after_{stage}_" if stage else ""
    torch.save(cnn_dial.state_dict(), os.path.join(args.save_dir, f"{prefix}dial_short72.pth"))
    torch.save(cnn_hour.state_dict(), os.path.join(args.save_dir, f"{prefix}hour_short72_72cls.pth"))
    torch.save(cnn_minute.state_dict(), os.path.join(args.save_dir, f"{prefix}minute_short72_12cls.pth"))
    print(f"[saved] {stage} snapshots written under: {args.save_dir}")

def main():
    args = base.parse_args()
    base.seed_everything(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # resolve CSV path
    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.data_root, csv_path)

    # transforms (base と同じ)
    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # hands/time 用データ（短針72）
    train_torch = RotationCsvTorchDataset72(
        data_root=args.data_root,
        csv_path=csv_path,
        subset="train",
        transform=tfm,
        max_samples=args.max_train,
    )
    test_torch = RotationCsvTorchDataset72(
        data_root=args.data_root,
        csv_path=csv_path,
        subset="test",
        transform=tfm,
        max_samples=args.max_test,
    )

    # dial 用は元の Dataset（短針72は不要）
    dial_root = args.dial_data_root
    if dial_root is None:
        dial_root = os.path.join(args.data_root, "nohands")

    dial_train_torch = base.RotationCsvTorchDataset(
        data_root=dial_root,
        csv_path=csv_path,
        subset="train",
        transform=tfm,
        max_samples=args.max_train,
        kp12_csv_path=args.dial_kp_csv,
    )
    dial_test_torch = base.RotationCsvTorchDataset(
        data_root=dial_root,
        csv_path=csv_path,
        subset="test",
        transform=tfm,
        max_samples=args.max_test,
        kp12_csv_path=args.dial_kp_csv,
    )

    # build model (hour=72)
    model, cnn_dial, cnn_hour, cnn_minute = build_deepproblog_model_short72(
        prolog_file=args.prolog,
        device=device,
        lr_dial=args.lr_dial,
        lr_hands=args.lr_hands,
        dial_arch=args.dial_arch,
        dial_heatmap_size=args.dial_heatmap_size,
        dial_softargmax_temp=args.dial_softargmax_temp,
        dial_angle_softmax_k=args.dial_angle_softmax_k,
        weights_dial=args.load_dial,
        weights_hour=args.load_hour,
        weights_minute=args.load_minute,
    )


    # 以降の学習オーケストレーションは base をそのまま流用
    if args.run == "dial":
        base.train_dial_stage(args, model, cnn_dial, dial_train_torch, dial_test_torch, device)
    elif args.run == "hands":
        args.task = "hands"
        base.run_training(args, model, train_torch, test_torch)
        save_stage_weights(args, cnn_dial, cnn_hour, cnn_minute, "hands")
    elif args.run == "time":
        args.task = "time"
        base.run_training(args, model, train_torch, test_torch)
        save_stage_weights(args, cnn_dial, cnn_hour, cnn_minute, "time")
    else:
        # pretrain_and_finetune: dial -> hands -> time
        base.train_dial_stage(args, model, cnn_dial, dial_train_torch, dial_test_torch, device)

        args.task = "hands"
        base.run_training(args, model, train_torch, test_torch)
        save_stage_weights(args, cnn_dial, cnn_hour, cnn_minute, "hands")
        args.task = "time"
        base.run_training(args, model, train_torch, test_torch)
        save_stage_weights(args, cnn_dial, cnn_hour, cnn_minute, "time")
    # save weights (同名ファイルで上書きされないよう suffix を付ける)
    os.makedirs(args.save_dir, exist_ok=True)
    torch.save(cnn_dial.state_dict(), os.path.join(args.save_dir, "dial_short72_latest.pth"))
    torch.save(cnn_hour.state_dict(), os.path.join(args.save_dir, "hour_short72_72cls_latest.pth"))
    torch.save(cnn_minute.state_dict(), os.path.join(args.save_dir, "minute_short72_12cls_latest.pth"))
    print("[saved] latest weights written under:", args.save_dir)

if __name__ == "__main__":
    main()
