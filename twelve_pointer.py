import os
import numpy as np
import pandas as pd
import cv2

# =========================
# ここだけ編集すればOK
# =========================
CSV_PATH = "labels_points.csv"
IMG_DIR  = "./images_nohands"      # 針を消した画像フォルダでもOK
OUT_CSV  = "clocks_with_noon.csv"

DEBUG_DRAW = True
DEBUG_OUT_DIR = "./debug_noon"     # 12時点を描画した画像を保存（不要ならDEBUG_DRAW=False）

# 12時点を打つ半径：中心から画像端までの最大半径(max_r)に対する割合
R_RATIO = 0.85   # 224x224なら0.80〜0.92あたりが無難
DOT_RADIUS = 4   # 点の大きさ
# =========================


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def norm360(deg):
    return (deg % 360.0 + 360.0) % 360.0

def angle_from_point(x, y, cx, cy):
    # 12時=0°, 時計回りが正
    dx = x - cx
    dy = cy - y  # y反転
    ang = np.degrees(np.arctan2(dx, dy))
    return norm360(ang)

def circ_mean_deg(degs, weights=None):
    degs = np.array(degs, dtype=float)
    if weights is None:
        weights = np.ones_like(degs)
    else:
        weights = np.array(weights, dtype=float)
    rad = np.deg2rad(degs)
    s = np.sum(weights * np.sin(rad))
    c = np.sum(weights * np.cos(rad))
    if np.isclose(s, 0) and np.isclose(c, 0):
        return float(degs[0])
    return norm360(np.degrees(np.arctan2(s, c)))

def parse_float(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    try:
        return float(v)
    except:
        return None

def main():
    df = pd.read_csv(CSV_PATH)

    # 必要列
    needed = ["file", "cx", "cy", "minute_x", "minute_y", "hour_x", "hour_y"]
    for c in needed:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    # 時刻列（time_h,time_m）がある前提。なければ rotation_deg があればそれを使えるようにする
    has_time = ("time_h" in df.columns) and ("time_m" in df.columns)
    has_rot  = ("rotation_deg" in df.columns)

    if not has_time and not has_rot:
        raise ValueError("Need either (time_h,time_m) or rotation_deg in CSV.")

    if DEBUG_DRAW:
        ensure_dir(DEBUG_OUT_DIR)

    x12_list, y12_list, theta_list = [], [], []

    for _, row in df.iterrows():
        fname = str(row["file"]).strip()

        cx = parse_float(row["cx"])
        cy = parse_float(row["cy"])
        mx = parse_float(row["minute_x"])
        my = parse_float(row["minute_y"])
        hx = parse_float(row["hour_x"])
        hy = parse_float(row["hour_y"])

        if cx is None or cy is None:
            x12_list.append(np.nan); y12_list.append(np.nan); theta_list.append(np.nan)
            continue

        # 画像サイズ取得（半径計算のため）
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            # 画像が無い/読めないなら、サイズ不明で半径が決めにくいのでNaNにする
            x12_list.append(np.nan); y12_list.append(np.nan); theta_list.append(np.nan)
            continue
        h, w = img.shape[:2]

        # rotation（12位置の角度）を求める
        theta = None

        # (A) rotation_degがあるならそれを優先（生成時の真値がある場合）
        if has_rot and not (isinstance(row.get("rotation_deg"), float) and np.isnan(row.get("rotation_deg"))):
            theta = norm360(float(row["rotation_deg"]))

        # (B) 時刻＋針から推定
        if theta is None:
            if not has_time:
                x12_list.append(np.nan); y12_list.append(np.nan); theta_list.append(np.nan)
                continue

            th = row["time_h"]
            tm = row["time_m"]
            if (isinstance(th, float) and np.isnan(th)) or (isinstance(tm, float) and np.isnan(tm)):
                x12_list.append(np.nan); y12_list.append(np.nan); theta_list.append(np.nan)
                continue

            hh = int(th) % 12
            mm = int(tm) % 60

            exp_m = 6.0 * mm
            exp_h = 30.0 * hh + 0.5 * mm

            thetas = []
            weights = []

            # minute hand
            if mx is not None and my is not None:
                obs_m = angle_from_point(mx, my, cx, cy)
                thetas.append(norm360(obs_m - exp_m))
                weights.append(2.0)  # minuteをやや重めに

            # hour hand
            if hx is not None and hy is not None:
                obs_h = angle_from_point(hx, hy, cx, cy)
                thetas.append(norm360(obs_h - exp_h))
                weights.append(1.0)

            if len(thetas) == 0:
                x12_list.append(np.nan); y12_list.append(np.nan); theta_list.append(np.nan)
                continue

            theta = circ_mean_deg(thetas, weights)

        # 12時点の半径を決める（中心〜画像端の最大半径に対する割合）
        max_r = min(cx, cy, (w - 1) - cx, (h - 1) - cy)
        r = max_r * R_RATIO

        rad = np.deg2rad(theta)
        x12 = cx + r * np.sin(rad)
        y12 = cy - r * np.cos(rad)

        x12_list.append(x12)
        y12_list.append(y12)
        theta_list.append(theta)

        if DEBUG_DRAW:
            out = img.copy()
            cv2.circle(out, (int(round(x12)), int(round(y12))), DOT_RADIUS, (0, 0, 255), -1)
            cv2.putText(out, "12", (int(round(x12))+6, int(round(y12))-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)
            cv2.imwrite(os.path.join(DEBUG_OUT_DIR, fname), out)

    df["theta12_deg"] = theta_list
    df["noon_x"] = x12_list
    df["noon_y"] = y12_list

    df.to_csv(OUT_CSV, index=False)
    print("Saved:", OUT_CSV)
    if DEBUG_DRAW:
        print("Debug images:", os.path.abspath(DEBUG_OUT_DIR))

if __name__ == "__main__":
    main()
