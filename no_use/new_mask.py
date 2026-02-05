# remove_hands_from_clockkaggle.py
# ------------------------------------------------------------
# clock_kaggle の rotations.csv（座標付き）を使って、針を消した画像を生成する。
# 出力は OUT_ROOT 以下に、元と同じ相対パスで保存。
#
# 必要: pip install opencv-python pandas numpy
# ------------------------------------------------------------

import os
import math
import numpy as np
import pandas as pd

try:
    import cv2
except ImportError as e:
    raise ImportError(
        "OpenCV が必要です。`pip install opencv-python` を実行してください。"
    ) from e


# =========================
# ここだけ自分の環境に合わせて変更
# =========================
DATA_ROOT = "clock_kaggle"
CSV_PATH  = os.path.join(DATA_ROOT, "rotations_new.csv")  # or "rotations_new.csv"

# 出力先（針を消した画像の保存場所）
OUT_ROOT  = os.path.join(DATA_ROOT, "nohands")

# status==ok のみ使う（ok_nohour など除外）
STATUS_OK_ONLY = True

# 針マスクの太さ（px）
THICK_MINUTE = 10
THICK_HOUR   = 12

# 中心付近（針の根本）を消す半径（px）
CENTER_RADIUS = 10

# inpaint パラメータ
INPAINT_RADIUS = 3
INPAINT_METHOD = "telea"  # "telea" or "ns"
# =========================


def clamp_int(x, lo, hi):
    return int(max(lo, min(hi, round(float(x)))))


def build_hand_mask(h, w, cx, cy, mx, my, hx, hy,
                    thick_m=10, thick_h=12, center_r=10):
    """
    針の線分（中心->端点）2本 + 中心円 を白(255)で塗ったマスクを返す。
    """
    mask = np.zeros((h, w), dtype=np.uint8)

    cx_i = clamp_int(cx, 0, w - 1)
    cy_i = clamp_int(cy, 0, h - 1)

    mx_i = clamp_int(mx, 0, w - 1)
    my_i = clamp_int(my, 0, h - 1)

    hx_i = clamp_int(hx, 0, w - 1)
    hy_i = clamp_int(hy, 0, h - 1)

    # 線（針）
    cv2.line(mask, (cx_i, cy_i), (mx_i, my_i), 255, thickness=thick_m, lineType=cv2.LINE_AA)
    cv2.line(mask, (cx_i, cy_i), (hx_i, hy_i), 255, thickness=thick_h, lineType=cv2.LINE_AA)

    # 中心の丸（根本の残りを消す）
    cv2.circle(mask, (cx_i, cy_i), center_r, 255, thickness=-1, lineType=cv2.LINE_AA)

    return mask


def inpaint_image(img_bgr, mask):
    """
    OpenCV inpaint でマスク領域を埋める。
    """
    if INPAINT_METHOD.lower() == "ns":
        flag = cv2.INPAINT_NS
    else:
        flag = cv2.INPAINT_TELEA
    return cv2.inpaint(img_bgr, mask, INPAINT_RADIUS, flag)


def main():
    df = pd.read_csv(CSV_PATH)

    # フィルタ
    if STATUS_OK_ONLY and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()

    # 必須列チェック
    need_cols = {"file", "cx", "cy", "minute_x", "minute_y", "hour_x", "hour_y"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV に必須列がありません: {missing}")

    os.makedirs(OUT_ROOT, exist_ok=True)

    n_total = len(df)
    n_done = 0
    n_skip = 0

    for _, r in df.iterrows():
        rel_path = str(r["file"]).replace("\\", "/")  # 念のため
        in_path = os.path.join(DATA_ROOT, rel_path)

        if not os.path.exists(in_path):
            n_skip += 1
            continue

        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            n_skip += 1
            continue

        h, w = img.shape[:2]

        cx = r["cx"]; cy = r["cy"]
        mx = r["minute_x"]; my = r["minute_y"]
        hx = r["hour_x"];   hy = r["hour_y"]

        mask = build_hand_mask(
            h, w, cx, cy, mx, my, hx, hy,
            thick_m=THICK_MINUTE, thick_h=THICK_HOUR, center_r=CENTER_RADIUS
        )

        out_img = inpaint_image(img, mask)

        out_path = os.path.join(OUT_ROOT, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        cv2.imwrite(out_path, out_img)

        n_done += 1
        if n_done % 500 == 0:
            print(f"[progress] {n_done}/{n_total} saved...")

    print("==== done ====")
    print(f"input csv : {CSV_PATH}")
    print(f"out root  : {OUT_ROOT}")
    print(f"saved     : {n_done}")
    print(f"skipped   : {n_skip} (missing/corrupt files)")


if __name__ == "__main__":
    main()
