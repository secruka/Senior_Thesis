import os
import pandas as pd
import numpy as np
import cv2

# =========================
# ここだけ編集すればOK
# =========================
CSV_PATH = "labels_points.csv"
IMG_DIR  = "clock_kaggle/"
OUT_DIR  = "images_nohands/"

MODE = "inpaint"          # "inpaint" / "black" / "white"
THICKNESS = 9             # 針の太さ(px)
DILATE_ITER = 2           # マスクを太らせる回数
INPAINT_RADIUS = 3        # inpaintの強さ
ONLY_OK = True            # status == "ok" のみ処理
DEBUG_MASK_DIR = None     # 例: "./masks_debug" にするとマスク画像も保存
# =========================


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_float(x):
    if x is None:
        return None
    if isinstance(x, float) and np.isnan(x):
        return None
    try:
        return float(x)
    except Exception:
        return None

def make_hand_mask(h, w, cx, cy, hx, hy, mx, my, thickness):
    mask = np.zeros((h, w), dtype=np.uint8)
    c = (int(round(cx)), int(round(cy)))

    if hx is not None and hy is not None:
        p = (int(round(hx)), int(round(hy)))
        cv2.line(mask, c, p, 255, thickness=thickness, lineType=cv2.LINE_AA)

    if mx is not None and my is not None:
        p = (int(round(mx)), int(round(my)))
        cv2.line(mask, c, p, 255, thickness=thickness, lineType=cv2.LINE_AA)

    return mask

def dilate_mask(mask, dilate_iter):
    if dilate_iter <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    out = mask.copy()
    for _ in range(dilate_iter):
        out = cv2.dilate(out, k, iterations=1)
    return out

def main():
    print("CWD:", os.getcwd())
    print("OUT_DIR(abs):", os.path.abspath(OUT_DIR))

    # 入力チェック
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not os.path.isdir(IMG_DIR):
        raise FileNotFoundError(f"IMG_DIR not found: {IMG_DIR}")

    df = pd.read_csv(CSV_PATH)

    ensure_dir(OUT_DIR)
    if DEBUG_MASK_DIR:
        ensure_dir(DEBUG_MASK_DIR)

    total = done = skipped = write_failed = 0

    for _, row in df.iterrows():
        if ONLY_OK and ("status" in df.columns) and (str(row.get("status", "")) != "ok"):
            continue

        fname = str(row["file"]).strip()
        in_path = os.path.join(IMG_DIR, fname)
        total += 1

        if not os.path.exists(in_path):
            print(f"[skip] not found: {in_path}")
            skipped += 1
            continue

        img = cv2.imread(in_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"[skip] cannot read: {in_path}")
            skipped += 1
            continue

        h, w = img.shape[:2]

        cx = parse_float(row["cx"]); cy = parse_float(row["cy"])
        hx = parse_float(row["hour_x"]); hy = parse_float(row["hour_y"])
        mx = parse_float(row["minute_x"]); my = parse_float(row["minute_y"])

        if cx is None or cy is None or ((hx is None or hy is None) and (mx is None or my is None)):
            print(f"[skip] missing coords: {fname}")
            skipped += 1
            continue

        mask = make_hand_mask(h, w, cx, cy, hx, hy, mx, my, thickness=THICKNESS)
        mask = dilate_mask(mask, DILATE_ITER)

        if MODE == "inpaint":
            out = cv2.inpaint(img, mask, inpaintRadius=INPAINT_RADIUS, flags=cv2.INPAINT_TELEA)
        elif MODE == "black":
            out = img.copy(); out[mask > 0] = (0, 0, 0)
        elif MODE == "white":
            out = img.copy(); out[mask > 0] = (255, 255, 255)
        else:
            raise ValueError(f"Unknown MODE: {MODE}")

        out_path = os.path.join(OUT_DIR, fname)

        # ★ fname にサブフォルダが含まれる場合に備えて、中間フォルダも作る
        ensure_dir(os.path.dirname(out_path))

        ok = cv2.imwrite(out_path, out)
        if not ok:
            print(f"[FAIL write] {out_path}")
            write_failed += 1
            continue

        done += 1

    print(f"Finished. total_rows={total}, done={done}, skipped={skipped}, write_failed={write_failed}")
    print(f"Saved to: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
