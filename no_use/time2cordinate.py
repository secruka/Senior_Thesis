import numpy as np
import pandas as pd

# =========================
# ここだけ自分の環境に合わせて変更
# =========================
IN_CSV  = "/Users/ruka/Senior_Thesis/clock_kaggle/rotations.csv"
OUT_CSV = "/Users/ruka/Senior_Thesis/clock_kaggle/rotations_new.csv"

# 中心は常に固定
CX, CY = 112.0, 112.0

# 針の長さ（固定）
R_MIN  = 75.0   # minute hand
R_HOUR = 55.0   # hour hand

# 完璧な status のみ（"ok_nohour" 等は除外）
STATUS_OK_ONLY = True

# rotation_deg はこれだけ残す
ALLOWED_ROT = {0, 90, 180, 270}

# rotation_deg の定義が「時計回り=+」なら False
# もし「反時計回り=+」なら True（90/270 が逆になるので補正）
ROT_DEG_IS_CCW = False
# =========================


def rotate_cw_0_90_180_270(x0, y0, cx, cy, deg):
    """
    中心(cx,cy)周りに時計回りで deg∈{0,90,180,270} 回転（ベクトル化）
    """
    dx = x0 - cx
    dy = y0 - cy

    # それぞれの回転後 (dx', dy') を作る
    # 0:   ( dx,  dy)
    # 90:  (-dy,  dx)
    # 180: (-dx, -dy)
    # 270: ( dy, -dx)
    dx_p = np.select(
        [deg == 0, deg == 90, deg == 180, deg == 270],
        [dx,       -dy,       -dx,        dy],
        default=np.nan
    )
    dy_p = np.select(
        [deg == 0, deg == 90, deg == 180, deg == 270],
        [dy,        dx,       -dy,       -dx],
        default=np.nan
    )
    return cx + dx_p, cy + dy_p


def main():
    df = pd.read_csv(IN_CSV)

    # 必須列チェック
    need = {"time_h", "time_m", "rotation_deg"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")

    # 1) status==ok のみ（ok_nohour 等を確実に除外）
    if STATUS_OK_ONLY and "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().eq("ok")].copy()

    # 2) rotation_deg を 0/90/180/270 だけに制限
    df["rotation_deg_int"] = pd.to_numeric(df["rotation_deg"], errors="coerce").round().astype("Int64")
    df = df[df["rotation_deg_int"].isin(list(ALLOWED_ROT))].copy()

    # CCW定義なら、CWに変換して統一処理
    rot = df["rotation_deg_int"].astype(int).to_numpy()
    if ROT_DEG_IS_CCW:
        rot = (-rot) % 360

    # 3) 時刻 → 角度（ラジアン）
    h = df["time_h"].astype(int).to_numpy() % 12
    m = df["time_m"].astype(int).to_numpy()

    alpha_m = 2 * np.pi * (m / 60.0)
    alpha_h = 2 * np.pi * ((h + m / 60.0) / 12.0)

    # 4) canonical（回転なし）理想座標
    # x = cx + r*sin(alpha), y = cy - r*cos(alpha)
    min_x0 = CX + R_MIN  * np.sin(alpha_m)
    min_y0 = CY - R_MIN  * np.cos(alpha_m)
    hour_x0 = CX + R_HOUR * np.sin(alpha_h)
    hour_y0 = CY - R_HOUR * np.cos(alpha_h)

    # 5) rotation_deg に応じて回転
    min_x, min_y = rotate_cw_0_90_180_270(min_x0, min_y0, CX, CY, rot)
    hour_x, hour_y = rotate_cw_0_90_180_270(hour_x0, hour_y0, CX, CY, rot)

    # 6) 既存列を上書き（小数第2位まで）
    df["cx"] = CX
    df["cy"] = CY
    df["minute_x"] = np.round(min_x, 2)
    df["minute_y"] = np.round(min_y, 2)
    df["hour_x"]   = np.round(hour_x, 2)
    df["hour_y"]   = np.round(hour_y, 2)

    # 余計な補助列を消したければコメントアウト解除
    # df = df.drop(columns=["rotation_deg_int"])

    df.to_csv(OUT_CSV, index=False)
    print(f"Saved: {OUT_CSV}  rows={len(df)}")
    print("Overwritten: cx, cy, minute_x, minute_y, hour_x, hour_y (rounded to 2 decimals)")

if __name__ == "__main__":
    main()
