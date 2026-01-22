#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dial_12_keypoints.py

rotations.csv (or rotation.csv) の status=ok の行だけを使い、
rot_deg / rotation_deg から「12時位置（キーポイント）」(twelve_x, twelve_y) を生成して
新しい CSV を書き出すスクリプト。

※ ターミナル引数ではなく、ファイル内のパス定数 IN_CSV / OUT_CSV を編集して使う版です。

前提（画像座標系）:
- x: 右が正
- y: 下が正
- rot_deg = 0   -> 12時は画像の真上
- rot_deg = 90  -> 12時は画像の右
- rot_deg = 180 -> 12時は画像の下
- rot_deg = 270 -> 12時は画像の左

計算式:
theta = rot_deg (度) をラジアンに変換（時計回り）
twelve_x = cx + r * sin(theta)
twelve_y = cy - r * cos(theta)

半径 r の決め方（デフォルト）:
- minute_x/minute_y がある & 欠損でない -> r = 中心〜長針先端の距離
- それが無ければ FIXED_RADIUS を使う
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


# =========================
# ここを編集して使ってください
# =========================
# rotations.csv / rotation.csv のパス
IN_CSV = Path("rotations.csv")

# 出力CSVのパス
OUT_CSV = Path("clock_kaggle/dial_keypoints.csv")

# 長針座標がCSVに無い/欠損のときに使う半径
FIXED_RADIUS = 90.0

# 画像サイズ（clampに使用。224なら座標を[0,223]に丸める）
IMG_SIZE = 224

# 12時キーポイントを画像範囲にclampするか
CLAMP_TO_IMAGE = True
# =========================


def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def compute_radius(
    row: pd.Series,
    cx_col: str,
    cy_col: str,
    mx_col: Optional[str],
    my_col: Optional[str],
    fixed_radius: float,
) -> float:
    """半径 r を決める（長針先端距離があればそれを使う）。"""
    if mx_col is not None and my_col is not None:
        mx, my = row[mx_col], row[my_col]
        if pd.notna(mx) and pd.notna(my):
            cx, cy = float(row[cx_col]), float(row[cy_col])
            return math.hypot(float(mx) - cx, float(my) - cy)
    return float(fixed_radius)


def compute_twelve_point(cx: float, cy: float, r: float, rot_deg: float) -> Tuple[float, float]:
    """rot_deg に基づき 12時位置を計算して返す。"""
    theta = math.radians(rot_deg % 360.0)  # clockwise from top
    x = cx + r * math.sin(theta)
    y = cy - r * math.cos(theta)
    return x, y


def main() -> None:
    in_path = Path(IN_CSV)
    out_path = Path(OUT_CSV)

    if not in_path.exists():
        raise FileNotFoundError(
            f"IN_CSV が見つかりません: {in_path}\n"
            f"ファイル内の IN_CSV を正しいパスに変更してください。"
        )

    df = pd.read_csv(in_path)

    # 列名揺れに対応
    file_col = _find_col(df, ["file", "filepath", "path"])
    status_col = _find_col(df, ["status", "Status"])
    cx_col = _find_col(df, ["cx", "center_x", "c_x"])
    cy_col = _find_col(df, ["cy", "center_y", "c_y"])
    rot_col = _find_col(df, ["rot_deg", "rotation_deg", "rot_degree", "rotation"])

    # 長針先端（あれば半径推定に使用）
    mx_col = _find_col(df, ["minute_x", "min_x", "m_x"])
    my_col = _find_col(df, ["minute_y", "min_y", "m_y"])

    if file_col is None:
        raise ValueError("file 列が見つかりません（候補: file / filepath / path）")
    if status_col is None:
        raise ValueError("status 列が見つかりません（候補: status）")
    if cx_col is None or cy_col is None:
        raise ValueError("中心座標列 cx/cy が見つかりません（候補: cx, cy）")
    if rot_col is None:
        raise ValueError("回転角列 rot_deg/rotation_deg が見つかりません（候補: rot_deg / rotation_deg）")

    # status=ok のみ
    df_ok = df[df[status_col].astype(str).str.lower().eq("ok")].copy()

    # 12時キーポイント生成
    twelve_x_list = []
    twelve_y_list = []
    rot_cls_list = []

    img_max = IMG_SIZE - 1

    for _, row in df_ok.iterrows():
        cx = float(row[cx_col])
        cy = float(row[cy_col])
        rot_deg = float(row[rot_col])

        r = compute_radius(row, cx_col, cy_col, mx_col, my_col, FIXED_RADIUS)
        x, y = compute_twelve_point(cx, cy, r, rot_deg)

        if CLAMP_TO_IMAGE:
            x = float(np.clip(x, 0.0, float(img_max)))
            y = float(np.clip(y, 0.0, float(img_max)))

        twelve_x_list.append(x)
        twelve_y_list.append(y)

        # 0/90/180/270 を 0..3 に落とす（念のため round してから）
        rot_cls_list.append(int((round(rot_deg) % 360) // 90) % 4)

    df_out = pd.DataFrame(
        {
            "file": df_ok[file_col].astype(str),
            "cx": df_ok[cx_col].astype(float),
            "cy": df_ok[cy_col].astype(float),
            "twelve_x": twelve_x_list,
            "twelve_y": twelve_y_list,
            "rot_deg": df_ok[rot_col].astype(float),
            "rot_cls": rot_cls_list,
        }
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_path, index=False)

    print(f"[OK] wrote: {out_path}  (n={len(df_out)})")
    print(df_out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
