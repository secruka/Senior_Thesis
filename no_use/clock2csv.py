#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate rotation_deg (0/90/180/270) for analog-clock images and save to CSV.

Angle convention:
  theta = atan2(x-cx, cy-y) in degrees in [0,360)
  0°=up(12), 90°=right(3), 180°=down(6), 270°=left(9)  (clockwise positive)

rotation from minute hand:
  rotation = (theta_minute - 6*time_m) mod 360
  rotation_deg = nearest multiple of 90 (0/90/180/270)

This script can:
  A) Scan images under --root and create a new CSV
  B) Read an existing --input_csv and (re)compute rotation_deg
"""

from __future__ import annotations
import argparse
import math
import re
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import cv2
import numpy as np
import pandas as pd


# ----------------------------
# Geometry helpers
# ----------------------------
def angle_deg(cx: float, cy: float, x: float, y: float) -> float:
    """0°=up, clockwise positive."""
    dx = x - cx
    dy = cy - y
    ang = (math.degrees(math.atan2(dx, dy)) + 360.0) % 360.0
    return ang


def circ_dist(a: float, b: float) -> float:
    """Circular distance in degrees."""
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


def quantize90(deg: float) -> int:
    """Nearest multiple of 90 in {0,90,180,270}."""
    q = int(round(deg / 90.0) * 90) % 360
    return q


def parse_time_from_path(p: Path) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse time from parent folder name like '1-00', '12-35', '0-05', '09-10'.
    Returns (hour, minute) or (None, None) if not parseable.
    """
    parent = p.parent.name
    m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", parent)
    if not m:
        return None, None
    h = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= h <= 23 and 0 <= mm <= 59):
        return None, None
    return h, mm


def point_segment_distance(px, py, x1, y1, x2, y2) -> float:
    """Distance from point P to segment (x1,y1)-(x2,y2)."""
    vx, vy = x2 - x1, y2 - y1
    wx, wy = px - x1, py - y1
    c1 = vx * wx + vy * wy
    if c1 <= 0:
        return math.hypot(px - x1, py - y1)
    c2 = vx * vx + vy * vy
    if c2 <= c1:
        return math.hypot(px - x2, py - y2)
    t = c1 / c2
    projx = x1 + t * vx
    projy = y1 + t * vy
    return math.hypot(px - projx, py - projy)


# ----------------------------
# Vision: detect hand line candidates
# ----------------------------
def hand_mask_from_image(img_bgr: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """
    Create a binary mask likely containing clock hands.
    Strategy:
      - Otsu threshold for dark objects (binary_inv)
      - Keep only components connected (or very near) to center seed disk
      - Restrict to inner radius to avoid outer frame
    """
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu (dark -> 1)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # If mask is absurdly large/small, try opposite (rare but helps)
    ratio = (th > 0).mean()
    if ratio > 0.65 or ratio < 0.01:
        _, th2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th = th2

    # Remove tiny noise
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, k, iterations=1)

    # Restrict to inner circle (avoid frame)
    R = int(min(h, w) * 0.47)
    yy, xx = np.ogrid[:h, :w]
    inner = ((xx - cx) ** 2 + (yy - cy) ** 2) <= (R ** 2)
    th = (th * inner.astype(np.uint8))

    # Center seed disk
    seed = np.zeros((h, w), np.uint8)
    cv2.circle(seed, (cx, cy), 10, 255, -1)
    seed = cv2.dilate(seed, k, iterations=2)

    # Keep connected components that touch the seed
    th_d = cv2.dilate(th, k, iterations=2)
    num, labels = cv2.connectedComponents((th_d > 0).astype(np.uint8), connectivity=8)

    keep = np.zeros((h, w), np.uint8)
    if num <= 1:
        return keep

    seed_labels = np.unique(labels[seed > 0])
    for lab in seed_labels:
        if lab == 0:
            continue
        keep[labels == lab] = 255

    # Slight thinning for Hough stability (optional)
    keep = cv2.erode(keep, k, iterations=1)
    return keep


def line_candidates(img_bgr: np.ndarray, cx: int, cy: int) -> List[Dict]:
    """
    Return line segment candidates near the center with:
      - angle at far endpoint (0 up, clockwise)
      - length
      - far endpoint (x,y)
      - distance of segment to center
    """
    mask = hand_mask_from_image(img_bgr, cx, cy)
    if mask.sum() == 0:
        return []

    edges = cv2.Canny(mask, 50, 150)

    # Hough line segments
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=25,
        maxLineGap=8,
    )
    if lines is None:
        return []

    cand = []
    for (x1, y1, x2, y2) in lines[:, 0]:
        length = float(math.hypot(x2 - x1, y2 - y1))
        if length < 25:
            continue

        # Must pass near center
        dist = point_segment_distance(cx, cy, x1, y1, x2, y2)
        if dist > 14:
            continue

        # Choose endpoint farther from center as "tip"
        d1 = math.hypot(x1 - cx, y1 - cy)
        d2 = math.hypot(x2 - cx, y2 - cy)
        if d1 >= d2:
            tipx, tipy = x1, y1
        else:
            tipx, tipy = x2, y2

        ang = angle_deg(cx, cy, tipx, tipy)
        cand.append(
            dict(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                tipx=int(tipx), tipy=int(tipy),
                length=length, dist=dist, ang=ang
            )
        )

    # If too few candidates, relax a bit using all lines (fallback)
    if len(cand) == 0 and lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0]:
            length = float(math.hypot(x2 - x1, y2 - y1))
            if length < 30:
                continue
            d1 = math.hypot(x1 - cx, y1 - cy)
            d2 = math.hypot(x2 - cx, y2 - cy)
            tipx, tipy = (x1, y1) if d1 >= d2 else (x2, y2)
            ang = angle_deg(cx, cy, tipx, tipy)
            cand.append(dict(
                x1=int(x1), y1=int(y1), x2=int(x2), y2=int(y2),
                tipx=int(tipx), tipy=int(tipy),
                length=length, dist=0.0, ang=ang
            ))
    return cand


def estimate_rotation_from_candidates(
    candidates: List[Dict],
    time_h: int,
    time_m: int,
) -> Tuple[Optional[int], Optional[Dict], str]:
    """
    Choose best minute-hand candidate by using time prior:
      minute ideal angle = 6*time_m
      rotation = (theta - ideal) mod 360, then quantize to nearest 90
    We pick the candidate whose rotation is closest to a multiple of 90,
    while also preferring longer segments.
    Returns: (rotation_deg, best_candidate, status_msg)
    """
    if not candidates:
        return None, None, "fail:no_candidates"

    ideal_m = (6.0 * time_m) % 360.0

    best = None
    best_score = 1e9
    best_rot_q = None
    best_err = None

    for c in candidates:
        theta = c["ang"]
        rot = (theta - ideal_m) % 360.0
        rot_q = quantize90(rot)
        err = circ_dist(rot, rot_q)

        # score: err dominates, but longer is better (small bonus)
        score = err - 0.02 * c["length"] + 0.1 * c["dist"]
        if score < best_score:
            best_score = score
            best = c
            best_rot_q = rot_q
            best_err = err

    # Status
    if best_rot_q is None:
        return None, None, "fail:rot_none"
    if best_err is not None and best_err <= 25:
        st = "ok"
    else:
        st = f"warn:rot_err={best_err:.1f}"

    return int(best_rot_q), best, st


def pick_hour_candidate(
    candidates: List[Dict],
    minute_cand: Dict,
    time_h: int,
    time_m: int,
    rotation_deg: int
) -> Optional[Dict]:
    """
    Pick an hour-hand candidate closest to expected hour angle:
      ideal_hour = 30*(h mod 12) + 0.5*m
      expected = ideal_hour + rotation_deg
    """
    if not candidates:
        return None

    ideal_h = (30.0 * (time_h % 12) + 0.5 * time_m) % 360.0
    expected = (ideal_h + rotation_deg) % 360.0

    best = None
    best_d = 1e9
    for c in candidates:
        # skip the same segment (very rough check: same tip)
        if c["tipx"] == minute_cand["tipx"] and c["tipy"] == minute_cand["tipy"]:
            continue
        d = circ_dist(c["ang"], expected)
        # hour hand is usually shorter -> mild preference for shorter than minute
        d = d + 0.01 * max(0.0, c["length"] - minute_cand["length"])
        if d < best_d:
            best_d = d
            best = c

    # If it's wildly off, return None
    if best is not None and best_d <= 45:
        return best
    return None


# ----------------------------
# Core per-image processing
# ----------------------------
def process_one_image(
    img_path: Path,
    root: Path,
    fixed_center: Optional[Tuple[int, int]] = None,
    provided_time: Optional[Tuple[int, int]] = None,
) -> Dict:
    # time
    if provided_time is not None:
        time_h, time_m = provided_time
    else:
        time_h, time_m = parse_time_from_path(img_path)
        if time_h is None or time_m is None:
            time_h, time_m = None, None

    # read image
    img = cv2.imread(str(img_path))
    if img is None:
        return dict(
            file=str(img_path.relative_to(root)),
            time_h=-1, time_m=-1,
            cx=-1, cy=-1,
            minute_x=-1, minute_y=-1,
            hour_x=-1, hour_y=-1,
            rotation_deg=-1,
            status="fail:imread",
        )

    h, w = img.shape[:2]
    if fixed_center is not None:
        cx, cy = fixed_center
    else:
        # for 224x224 synthetic, this is usually correct
        cx, cy = w // 2, h // 2

    rel = str(img_path.relative_to(root))

    if time_h is None or time_m is None:
        return dict(
            file=rel, time_h=-1, time_m=-1,
            cx=cx, cy=cy,
            minute_x=-1, minute_y=-1,
            hour_x=-1, hour_y=-1,
            rotation_deg=-1,
            status="fail:no_time",
        )

    cands = line_candidates(img, cx, cy)
    rot_deg, minute_cand, st = estimate_rotation_from_candidates(cands, time_h, time_m)
    if rot_deg is None or minute_cand is None:
        return dict(
            file=rel, time_h=time_h, time_m=time_m,
            cx=cx, cy=cy,
            minute_x=-1, minute_y=-1,
            hour_x=-1, hour_y=-1,
            rotation_deg=-1,
            status=st,
        )

    # minute tip
    minute_x, minute_y = minute_cand["tipx"], minute_cand["tipy"]

    # hour
    hour_cand = pick_hour_candidate(cands, minute_cand, time_h, time_m, rot_deg)
    if hour_cand is None:
        hour_x, hour_y = -1, -1
        st = st + ";no_hour"
    else:
        hour_x, hour_y = hour_cand["tipx"], hour_cand["tipy"]

    return dict(
        file=rel, time_h=time_h, time_m=time_m,
        cx=cx, cy=cy,
        minute_x=minute_x, minute_y=minute_y,
        hour_x=hour_x, hour_y=hour_y,
        rotation_deg=rot_deg,
        status=st,
    )


def compute_rotation_from_existing_keypoints(row: pd.Series) -> Tuple[int, str]:
    """
    If you already have (cx,cy) and (minute_x,minute_y) and time_m,
    compute rotation_deg directly without image processing.
    """
    try:
        cx, cy = float(row["cx"]), float(row["cy"])
        mx, my = float(row["minute_x"]), float(row["minute_y"])
        tm = int(row["time_m"])
        theta = angle_deg(cx, cy, mx, my)
        rot = (theta - 6.0 * tm) % 360.0
        rot_deg = quantize90(rot)
        err = circ_dist(rot, rot_deg)
        st = "ok" if err <= 25 else f"warn:kp_rot_err={err:.1f}"
        return int(rot_deg), st
    except Exception as e:
        return -1, f"fail:kp_exception"


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dataset root (contains train/..., etc.)")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--input_csv", default=None, help="optional existing CSV to recompute/append rotation")
    ap.add_argument("--cx", type=int, default=None, help="fixed center x (optional)")
    ap.add_argument("--cy", type=int, default=None, help="fixed center y (optional)")
    ap.add_argument("--ext", default="jpg,jpeg,png", help="image extensions (comma separated)")
    ap.add_argument("--rel", action="store_true", help="store file paths relative to root (recommended)")
    ap.add_argument("--assume_time", default=None, help="fallback time like '1-00' if folder parsing fails")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out)

    fixed_center = None
    if args.cx is not None and args.cy is not None:
        fixed_center = (args.cx, args.cy)

    assume_time = None
    if args.assume_time:
        m = re.match(r"^\s*(\d{1,2})\s*-\s*(\d{1,2})\s*$", args.assume_time)
        if m:
            assume_time = (int(m.group(1)), int(m.group(2)))

    exts = {e.strip().lower() for e in args.ext.split(",") if e.strip()}
    if args.input_csv:
        df_in = pd.read_csv(args.input_csv)
        rows = []
        for _, r in df_in.iterrows():
            # file path
            f = str(r.get("file", ""))
            img_path = Path(f)
            if not img_path.is_absolute():
                img_path = root / f

            # time
            th = r.get("time_h", None)
            tm = r.get("time_m", None)
            if pd.isna(th) or pd.isna(tm):
                ph, pm = parse_time_from_path(img_path)
                if ph is None or pm is None:
                    if assume_time is None:
                        th, tm = -1, -1
                    else:
                        th, tm = assume_time
                else:
                    th, tm = ph, pm
            else:
                th, tm = int(th), int(tm)

            # center
            cx = r.get("cx", None)
            cy = r.get("cy", None)
            if pd.isna(cx) or pd.isna(cy):
                cx, cy = (fixed_center if fixed_center else (112, 112))
            else:
                cx, cy = int(cx), int(cy)

            # if minute keypoint exists -> compute directly
            mx = r.get("minute_x", np.nan)
            my = r.get("minute_y", np.nan)
            use_kp = (not pd.isna(mx)) and (not pd.isna(my)) and (float(mx) >= 0) and (float(my) >= 0) and (th >= 0) and (tm >= 0)

            out_row = dict(r)
            out_row["time_h"] = th
            out_row["time_m"] = tm
            out_row["cx"] = cx
            out_row["cy"] = cy

            if use_kp:
                rot_deg, st = compute_rotation_from_existing_keypoints(pd.Series(out_row))
                out_row["rotation_deg"] = rot_deg
                out_row["status"] = st
            else:
                # fall back to image processing
                provided_time = (th, tm) if (th >= 0 and tm >= 0) else assume_time
                pr = process_one_image(img_path, root, fixed_center=(cx, cy), provided_time=provided_time)
                out_row.update(pr)

            rows.append(out_row)

        df_out = pd.DataFrame(rows)
        df_out.to_csv(out, index=False)
        print(f"Wrote {len(df_out)} rows -> {out}")
        return

    # no input CSV: scan images
    img_files: List[Path] = []
    for e in exts:
        img_files.extend(root.rglob(f"*.{e}"))
    img_files = sorted(set(img_files))

    rows = []
    for p in img_files:
        pr = process_one_image(p, root, fixed_center=fixed_center, provided_time=None)
        if not args.rel:
            pr["file"] = str(p)
        rows.append(pr)

    df = pd.DataFrame(rows, columns=[
        "file","time_h","time_m","cx","cy",
        "minute_x","minute_y","hour_x","hour_y",
        "rotation_deg","status"
    ])
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
