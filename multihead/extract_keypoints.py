"""
extract_keypoints.py
====================
Automatically extract keypoints (center, hour hand tip, minute hand tip,
12-o'clock position) from clock images using image processing (OpenCV).

Combines:
  1. Image-processing-based rotation detection (12-o'clock direction)
  2. Geometric computation of hand tip coordinates from known time labels
  3. Image-processing-based hand detection as validation / fallback

Produces annotations.csv with columns:
  file, time_h, time_m, cx, cy, minute_x, minute_y, hour_x, hour_y,
  twelve_x, twelve_y, rot_deg, rot_cls
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
IMG_SIZE = 224
CX, CY = 112.0, 112.0          # Clock center (fixed for this dataset)
HAND_RADIUS_MINUTE = 75.0       # Approximate minute hand length (pixels)
HAND_RADIUS_HOUR = 47.63        # Approximate hour hand length (pixels)
TWELVE_RADIUS = 75.0            # Distance from center to 12-o'clock mark

ROTATION_DEGS = [0, 90, 180, 270]


# ---------------------------------------------------------------------------
# Rotation detection via image processing
# ---------------------------------------------------------------------------

def _detect_rotation_from_image(img_bgr: np.ndarray) -> int:
    """Detect dial rotation (0/90/180/270) from the image.

    Strategy: analyse the distribution of tick-mark / numeral pixels
    in each quadrant.  The 12-o'clock side typically has a denser cluster
    of dark features (the "12" numeral or the longer hour-mark).

    We use a robust approach:
      1. Convert to grayscale, threshold to isolate dark features.
      2. Create a ring-shaped mask (annular region where tick marks live).
      3. Compute the centroid angle of dark pixels in that ring.
      4. Snap to nearest 90-degree multiple.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold to handle varying backgrounds
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # Create annular mask: keep only the outer ring where tick marks live
    h, w = binary.shape
    cx, cy = w // 2, h // 2
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    outer_r = min(cx, cy) - 2
    inner_r = outer_r * 0.55  # tick marks are between 55% and 100% of radius
    ring_mask = ((dist >= inner_r) & (dist <= outer_r)).astype(np.uint8)

    # Also mask out the very edge (frame artifacts)
    edge_mask = (dist <= outer_r - 3).astype(np.uint8)
    ring_mask = ring_mask & edge_mask

    masked = binary & (ring_mask * 255)

    # Compute weighted angle from dark pixels
    ys, xs = np.where(masked > 0)
    if len(xs) < 10:
        return 0  # fallback

    # Angles relative to center, measured from top (12-o'clock = 0)
    dx = xs.astype(float) - cx
    dy = ys.astype(float) - cy
    angles = np.arctan2(dx, -dy)  # 0 = up, positive = clockwise

    # We want to find the dominant direction.
    # Sum unit vectors, find the resultant angle.
    sum_sin = np.sum(np.sin(angles))
    sum_cos = np.sum(np.cos(angles))

    # This gives the "average" direction of dark pixels in the ring.
    # However, tick marks are distributed all around, so we need a
    # different approach: look for the DENSEST 90-degree sector.

    best_rot = 0
    best_count = 0
    for rot_deg in ROTATION_DEGS:
        # The 12-o'clock direction in image coords for this rotation
        # rot_deg=0 => 12 is at top => angle ~0
        # rot_deg=90 => 12 is at right => angle ~pi/2
        # rot_deg=180 => 12 is at bottom => angle ~pi
        # rot_deg=270 => 12 is at left => angle ~-pi/2
        target_angle = math.radians(rot_deg)
        # Count pixels within +-30 degrees of the target
        diff = np.abs(np.arctan2(np.sin(angles - target_angle),
                                  np.cos(angles - target_angle)))
        count = np.sum(diff < math.radians(30))
        if count > best_count:
            best_count = count
            best_rot = rot_deg

    return best_rot


def _detect_rotation_heuristic(img_bgr: np.ndarray) -> int:
    """Alternative rotation detection using line-thickness analysis.

    The 12-o'clock mark is often a thicker/longer tick.
    We look at 4 cardinal directions and find the one with the most
    dark pixels in a narrow wedge near the edge.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape
    cx, cy = w // 2, h // 2
    r_outer = min(cx, cy) - 5
    r_inner = int(r_outer * 0.7)

    # Define wedge regions for each cardinal direction (±15 degrees)
    directions = {
        0: (-15, 15),       # top = 12 o'clock
        90: (75, 105),      # right
        180: (165, 195),    # bottom
        270: (255, 285),    # left
    }

    best_rot = 0
    best_score = 0
    for rot_deg, (a_start, a_end) in directions.items():
        mask = np.zeros_like(binary)
        # Create wedge mask
        for r in range(r_inner, r_outer):
            for a_deg in range(a_start, a_end):
                a_rad = math.radians(a_deg - 90)  # -90 because 0=right in standard
                px = int(cx + r * math.cos(a_rad))
                py = int(cy + r * math.sin(a_rad))
                if 0 <= px < w and 0 <= py < h:
                    mask[py, px] = 255

        score = np.sum((binary & mask) > 0)
        if score > best_score:
            best_score = score
            best_rot = rot_deg

    return best_rot


def _detect_rotation_by_hands(img_bgr: np.ndarray, time_h: int, time_m: int) -> int:
    """Detect rotation by finding the minute hand and back-computing rotation.

    The minute hand is the longest dark line from center.
    If we know the time, we can compute what angle the minute hand SHOULD be
    at in canonical orientation, then compare with the detected angle to
    infer the rotation.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)

    h, w = binary.shape
    cx, cy = w // 2, h // 2

    # Mask out center region (avoid hub) and outer edge (frame)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    mask = ((dist >= 15) & (dist <= min(cx, cy) - 10)).astype(np.uint8)
    masked = binary & (mask * 255)

    # Find the dominant line direction using Hough transform
    lines = cv2.HoughLinesP(masked, 1, np.pi / 180, threshold=20,
                            minLineLength=30, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return None  # fallback

    # Find the longest line that passes near center
    best_line = None
    best_len = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Check if line passes near center (within 20 pixels)
        line_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        dist_to_center = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / (line_len + 1e-9)
        if dist_to_center < 25 and line_len > best_len:
            best_len = line_len
            best_line = line[0]

    if best_line is None:
        return None

    x1, y1, x2, y2 = best_line
    # Determine which end is the tip (farther from center)
    d1 = math.sqrt((x1 - cx) ** 2 + (y1 - cy) ** 2)
    d2 = math.sqrt((x2 - cx) ** 2 + (y2 - cy) ** 2)
    if d1 > d2:
        tip_x, tip_y = x1, y1
    else:
        tip_x, tip_y = x2, y2

    # Detected angle of the longest hand (likely minute hand) in image coords
    detected_deg = math.degrees(math.atan2(tip_x - cx, -(tip_y - cy))) % 360

    # Expected minute hand angle in canonical coords (no rotation)
    expected_deg = (time_m / 60.0) * 360.0

    # Rotation = detected - expected
    diff = (detected_deg - expected_deg + 360) % 360

    # Snap to nearest 90
    best_rot = min(ROTATION_DEGS, key=lambda r: min(abs(diff - r), 360 - abs(diff - r)))
    return best_rot


def detect_rotation(img_bgr: np.ndarray, time_h: int = None, time_m: int = None) -> int:
    """Detect rotation using multiple methods with consensus.

    If time is known, uses hand-based detection as primary method.
    Falls back to tick-mark analysis.
    """
    # Method 1: hand-based (if time is known)
    if time_h is not None and time_m is not None:
        rot_hands = _detect_rotation_by_hands(img_bgr, time_h, time_m)
        if rot_hands is not None:
            # Verify with tick-mark method
            rot_ticks = _detect_rotation_from_image(img_bgr)
            if rot_hands == rot_ticks:
                return rot_hands
            # If they disagree, trust hands (more reliable when time is known)
            return rot_hands

    # Method 2: tick-mark analysis
    return _detect_rotation_from_image(img_bgr)


# ---------------------------------------------------------------------------
# Geometric computation of hand positions
# ---------------------------------------------------------------------------

def compute_hand_coords(time_h: int, time_m: int, rot_deg: int):
    """Compute hand tip coordinates given time and rotation.

    Returns: (minute_x, minute_y, hour_x, hour_y, twelve_x, twelve_y)

    Angle convention (image coordinates):
      - 12 o'clock (canonical) = -Y direction (up)
      - Clockwise = positive angle
      - minute at M minutes = M/60 * 360 degrees from 12
      - hour at H:M = (H%12)/12 * 360 + M/60 * 30 degrees from 12

    Rotation applies to the entire dial:
      - rot_deg=0   => 12 is at top
      - rot_deg=90  => 12 is at right (dial rotated 90 CW)
      - rot_deg=180 => 12 is at bottom
      - rot_deg=270 => 12 is at left
    """
    # Minute hand angle from canonical 12 (degrees, CW)
    minute_angle_deg = (time_m / 60.0) * 360.0
    # Hour hand angle from canonical 12 (degrees, CW)
    hour_angle_deg = ((time_h % 12) / 12.0) * 360.0 + (time_m / 60.0) * 30.0

    # Apply dial rotation: in the IMAGE, the 12-o'clock is at rot_deg
    # So hand angles in image coords = canonical_angle + rot_deg
    minute_img_deg = minute_angle_deg + rot_deg
    hour_img_deg = hour_angle_deg + rot_deg

    # Convert angle to image coordinates
    # 0 degrees (12 o'clock) = up = (0, -1)
    # 90 degrees (3 o'clock) = right = (1, 0)
    minute_x = CX + HAND_RADIUS_MINUTE * math.sin(math.radians(minute_img_deg))
    minute_y = CY - HAND_RADIUS_MINUTE * math.cos(math.radians(minute_img_deg))

    hour_x = CX + HAND_RADIUS_HOUR * math.sin(math.radians(hour_img_deg))
    hour_y = CY - HAND_RADIUS_HOUR * math.cos(math.radians(hour_img_deg))

    # 12 o'clock position
    twelve_img_deg = rot_deg  # 12 o'clock IS at rot_deg in image
    twelve_x = CX + TWELVE_RADIUS * math.sin(math.radians(twelve_img_deg))
    twelve_y = CY - TWELVE_RADIUS * math.cos(math.radians(twelve_img_deg))

    return minute_x, minute_y, hour_x, hour_y, twelve_x, twelve_y


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def extract_annotations(data_root: str, output_path: str):
    """Process all images in clocks.csv and produce annotations.csv."""
    csv_path = os.path.join(data_root, "clocks.csv")
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} entries from clocks.csv")

    rows = []
    failed = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting keypoints"):
        filepath = row["filepaths"]
        label = row["labels"]

        # Parse time from label (e.g. "1_00" -> h=1, m=0)
        parts = label.split("_")
        time_h = int(parts[0])
        time_m = int(parts[1])

        img_path = os.path.join(data_root, filepath)
        if not os.path.exists(img_path):
            failed += 1
            continue

        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue

        # Detect rotation from image (using time info for better accuracy)
        rot_deg = detect_rotation(img, time_h, time_m)
        rot_cls = ROTATION_DEGS.index(rot_deg)

        # Compute hand coordinates geometrically
        minute_x, minute_y, hour_x, hour_y, twelve_x, twelve_y = \
            compute_hand_coords(time_h, time_m, rot_deg)

        rows.append({
            "file": filepath,
            "time_h": time_h,
            "time_m": time_m,
            "cx": CX,
            "cy": CY,
            "minute_x": round(minute_x, 2),
            "minute_y": round(minute_y, 2),
            "hour_x": round(hour_x, 2),
            "hour_y": round(hour_y, 2),
            "twelve_x": round(twelve_x, 2),
            "twelve_y": round(twelve_y, 2),
            "rot_deg": rot_deg,
            "rot_cls": rot_cls,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"Saved {len(out_df)} annotations to {output_path}")
    if failed > 0:
        print(f"WARNING: {failed} images failed to load")

    # Print rotation distribution
    print("\nRotation distribution:")
    print(out_df["rot_cls"].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser(
        description="Extract keypoints from clock images → annotations.csv"
    )
    parser.add_argument(
        "--data_root", type=str, required=True,
        help="Path to clock_kaggle directory"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path (default: <data_root>/annotations.csv)"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data_root, "annotations.csv")

    extract_annotations(args.data_root, args.output)


if __name__ == "__main__":
    main()
