import os
import math
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

def parse_label_to_time(label: str):
    # labels: "1_00" のような形式
    h, m = label.split("_")
    return int(h), int(m)

def expected_angles(hh, mm):
    # clock-angle: 0deg=12時方向, clockwise positive
    minute = (6 * mm) % 360
    hour = (30 * (hh % 12) + 0.5 * mm) % 360
    return hour, minute

def bg_median(img_bgr, border=6):
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), bool)
    mask[:border, :] = True
    mask[-border:, :] = True
    mask[:, :border] = True
    mask[:, -border:] = True
    pix = img_bgr[mask].reshape(-1, 3)
    return np.median(pix, axis=0)

def tip_on_ray_segment(img_bgr, cx, cy, ang_deg,
                       delta=35, max_r=None, hub_skip=5, gap_tol=3):
    """
    背景色(外枠の中央値)と違うピクセルを「針候補」とみなし、
    中心付近から連続している区間の末端を tip とする。
    """
    h, w = img_bgr.shape[:2]
    if max_r is None:
        max_r = min(h, w) // 2 - 2

    bg = bg_median(img_bgr, border=6)

    theta = math.radians(ang_deg)
    dx = math.sin(theta)
    dy = -math.cos(theta)

    started = False
    last = None
    gap = 0
    score = 0

    for t in range(0, int(max_r) + 1):
        x = int(round(cx + dx * t))
        y = int(round(cy + dy * t))
        if x < 0 or x >= w or y < 0 or y >= h:
            break
        if t < hub_skip:
            continue

        d = np.linalg.norm(img_bgr[y, x].astype(float) - bg)
        is_hand = d > delta

        if not started:
            if is_hand:
                started = True
                last = (x, y, t)
                score += 1
                gap = 0
        else:
            if is_hand:
                last = (x, y, t)
                score += 1
                gap = 0
            else:
                gap += 1
                if gap > gap_tol:
                    break

    return last, score

def detect_tips_with_rotation(img_bgr, hh, mm, cx, cy, delta=35):
    hour_exp, min_exp = expected_angles(hh, mm)

    best = None
    for rot in [0, 90, 180, 270]:
        # 画像が rot 度回っていると仮定 → 針の見える角度も rot だけ回る
        hour_img = (hour_exp + rot) % 360
        min_img  = (min_exp  + rot) % 360

        h_last, h_score = tip_on_ray_segment(img_bgr, cx, cy, hour_img,  delta=delta)
        m_last, m_score = tip_on_ray_segment(img_bgr, cx, cy, min_img,   delta=delta)

        total = h_score + m_score
        if best is None or total > best[0]:
            best = (total, rot, h_last, m_last)

    return best  # (score, rot, hour_tip(x,y,t)|None, minute_tip(x,y,t)|None)

def main(clocks_csv, images_root, out_csv,
         split=None, cx=112, cy=112, delta=35, vis_dir=None):

    df = pd.read_csv(clocks_csv)

    # split 指定がある場合: data set列でフィルタ
    if split is not None:
        df = df[df["data set"] == split].copy()

    if vis_dir is not None:
        os.makedirs(vis_dir, exist_ok=True)

    rows = []
    for _, r in df.iterrows():
        relpath = r["filepaths"]          # 例: train/1-00/0.jpg
        label   = r["labels"]             # 例: 1_00
        hh, mm  = parse_label_to_time(label)

        img_path = os.path.join(images_root, relpath)
        img = cv2.imread(img_path)
        if img is None:
            rows.append({
                "file": relpath, "time_h": hh, "time_m": mm,
                "cx": cx, "cy": cy,
                "minute_x": None, "minute_y": None,
                "hour_x": None, "hour_y": None,
                "rotation_deg": None,
                "status": "file_missing"
            })
            continue

        best = detect_tips_with_rotation(img, hh, mm, cx, cy, delta=delta)
        _, rot, h_tip, m_tip = best

        if h_tip is None or m_tip is None:
            rows.append({
                "file": relpath, "time_h": hh, "time_m": mm,
                "cx": cx, "cy": cy,
                "minute_x": None, "minute_y": None,
                "hour_x": None, "hour_y": None,
                "rotation_deg": rot,
                "status": "fail_ray"
            })
            continue

        hx, hy, _ = h_tip
        mx, my, _ = m_tip

        rows.append({
            "file": relpath, "time_h": hh, "time_m": mm,
            "cx": cx, "cy": cy,
            "minute_x": int(mx), "minute_y": int(my),
            "hour_x": int(hx), "hour_y": int(hy),
            "rotation_deg": int(rot),
            "status": "ok"
        })

        # デバッグ可視化（任意）
        if vis_dir is not None:
            vis = img.copy()
            cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
            cv2.circle(vis, (mx, my), 4, (255, 0, 0), -1)  # minute
            cv2.circle(vis, (hx, hy), 4, (0, 0, 255), -1)  # hour
            outp = os.path.join(vis_dir, relpath.replace("/", "__"))
            cv2.imwrite(outp, vis)

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}  rows={len(out)}")
    if "status" in out.columns:
        print(out["status"].value_counts())

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--clocks_csv", required=True)
    ap.add_argument("--images_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--split", default=None, help="train/valid/test のどれか（任意）")
    ap.add_argument("--cx", type=int, default=112)
    ap.add_argument("--cy", type=int, default=112)
    ap.add_argument("--delta", type=float, default=35)
    ap.add_argument("--vis_dir", default=None)
    args = ap.parse_args()

    main(args.clocks_csv, args.images_root, args.out_csv,
         split=args.split, cx=args.cx, cy=args.cy, delta=args.delta, vis_dir=args.vis_dir)
