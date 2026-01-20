#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

import pandas as pd
import importlib


# ----------------------------
# Label <-> degree mappings (match your clock_integrated.pl)
# ----------------------------
MINUTE_LABELS = [1,2,3,4,5,6,7,8,9,10,11,12]               # 12 means 00 minutes
HOUR_LABELS   = [1,2,3,4,5,6,7,8,9,10,11,12]
ROT_LABELS_DEG = [0,30,60,90,120,150,180,210,240,270,300,330]

# minute: 12->0deg, 1->30deg, ..., 11->330deg
minute_label_to_deg = {12: 0, **{k: k*30 for k in range(1,12)}}
hour_label_to_deg   = {12: 0, **{k: k*30 for k in range(1,12)}}

deg_to_minute_label = {v: k for k, v in minute_label_to_deg.items()}
deg_to_hour_label   = {v: k for k, v in hour_label_to_deg.items()}


def mod360(x: int) -> int:
    return ((x % 360) + 360) % 360


def minute_label_to_minutes(m_label: int) -> int:
    # 12 -> 0, 1 -> 5, ..., 11 -> 55
    return 0 if m_label == 12 else m_label * 5


def hour_label_to_hour(h_label: int) -> int:
    return 12 if h_label == 12 else h_label


# ----------------------------
# Model loading helpers
# ----------------------------
def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    # handles checkpoints saved under DataParallel ("module.xxx")
    if not state_dict:
        return state_dict
    if any(k.startswith("module.") for k in state_dict.keys()):
        return {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    return state_dict


def try_import_class(module_name: str, class_name: str):
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, class_name, None)
    except Exception:
        return None


def fallback_resnet18_softmax(num_classes: int) -> nn.Module:
    m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_dim = m.fc.in_features
    m.fc = nn.Sequential(nn.Linear(in_dim, num_classes), nn.Softmax(dim=1))
    return m


def build_hour_minute_model(device: torch.device) -> nn.Module:
    # Try to reuse your repo's class if available
    ClockNet = try_import_class("model_hands", "ClockNet")
    if ClockNet is not None:
        try:
            return ClockNet(num_classes=12).to(device)
        except Exception:
            pass
    # Fallback
    return fallback_resnet18_softmax(12).to(device)


def build_rot_model(device: torch.device) -> nn.Module:
    DialNet = try_import_class("model_dial", "DialNet")
    if DialNet is not None:
        try:
            return DialNet(num_classes=12).to(device)
        except Exception:
            pass
    # Fallback
    return fallback_resnet18_softmax(12).to(device)


def load_ckpt(model: nn.Module, ckpt_path: str, device: torch.device) -> None:
    sd = torch.load(ckpt_path, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd and isinstance(sd["state_dict"], dict):
        sd = sd["state_dict"]
    if not isinstance(sd, dict):
        raise ValueError(f"Checkpoint at {ckpt_path} is not a state_dict-like dict.")
    sd = _strip_module_prefix(sd)
    model.load_state_dict(sd, strict=False)


def ensure_probs(x: torch.Tensor) -> torch.Tensor:
    """
    Some models already output softmax. If not, apply softmax.
    """
    # x: (1, C)
    if x.dim() != 2:
        raise ValueError(f"Expected 2D logits/probs, got shape {tuple(x.shape)}")
    # heuristic: if sums ~1 and all >=0, treat as probs
    s = x.sum(dim=1)
    if torch.all(x >= -1e-6) and torch.allclose(s, torch.ones_like(s), atol=1e-3):
        return x
    return torch.softmax(x, dim=1)


def topk_with_labels(probs: torch.Tensor, labels: List[int], k: int) -> List[Tuple[int, float]]:
    """
    probs: (1, C)
    labels: length C, label for each class index
    """
    k = min(k, probs.shape[1])
    vals, idx = torch.topk(probs[0], k=k)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        out.append((labels[i], float(v)))
    return out


# ----------------------------
# Compose (same as Prolog)
# ----------------------------
def compose_time_distribution(
    hour_top: List[Tuple[int, float]],
    minute_top: List[Tuple[int, float]],
    rot_top: List[Tuple[int, float]],
    k: int
) -> Tuple[Dict[str, float], Dict[str, List[Dict[str, Any]]], float]:
    """
    Returns:
      - time_probs: aggregated (unnormalized) probability mass over times
      - time_paths: per-time list of contributing paths
      - coverage_mass: total mass covered by top-k combinations
    """
    time_probs: Dict[str, float] = {}
    time_paths: Dict[str, List[Dict[str, Any]]] = {}
    coverage = 0.0

    for (hraw, ph) in hour_top[:k]:
        deg_h_raw = hour_label_to_deg[hraw]
        for (mraw, pm) in minute_top[:k]:
            deg_m_raw = minute_label_to_deg[mraw]
            for (rdeg, pr) in rot_top[:k]:
                p = ph * pm * pr
                coverage += p

                deg_h_corr = mod360(deg_h_raw - rdeg)
                deg_m_corr = mod360(deg_m_raw - rdeg)

                # Map back to discrete labels (0,30,...,330 must exist)
                if deg_h_corr not in deg_to_hour_label or deg_m_corr not in deg_to_minute_label:
                    continue

                h_corr_label = deg_to_hour_label[deg_h_corr]
                m_corr_label = deg_to_minute_label[deg_m_corr]

                hour_val = hour_label_to_hour(h_corr_label)
                minute_val = minute_label_to_minutes(m_corr_label)

                time_str = f"{hour_val}:{minute_val:02d}"

                time_probs[time_str] = time_probs.get(time_str, 0.0) + p
                time_paths.setdefault(time_str, []).append({
                    "p": p,
                    "Hraw": hraw,
                    "Mraw": mraw,
                    "Rdeg": rdeg,
                    "Hraw_deg": deg_h_raw,
                    "Mraw_deg": deg_m_raw,
                    "Hcorr_deg": deg_h_corr,
                    "Mcorr_deg": deg_m_corr,
                    "H": h_corr_label,
                    "M": m_corr_label,
                    "time": time_str
                })

    return time_probs, time_paths, coverage


def entropy_of_distribution(d: Dict[str, float]) -> float:
    total = sum(d.values())
    if total <= 0:
        return float("nan")
    ent = 0.0
    for _, v in d.items():
        p = v / total
        if p > 0:
            ent -= p * math.log(p + 1e-12)
    return ent


# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default="clock_kaggle",
                        help="Dataset root that contains train/... and CSV.")
    parser.add_argument("--csv", type=str, default="labels_points.csv",
                        help="CSV filename under root_dir (must have 'file', 'time_h', 'time_m').")
    parser.add_argument("--subset", type=str, default=None,
                        help="Optional: filter rows by file prefix 'train/' or 'test/' etc.")
    parser.add_argument("--index", type=int, default=0,
                        help="Row index after filtering (used if --file not provided).")
    parser.add_argument("--file", type=str, default=None,
                        help="Image path. If relative, interpreted as root_dir/<file>.")
    parser.add_argument("--hour_ckpt", type=str, required=True,
                        help="Checkpoint for hour model (state_dict).")
    parser.add_argument("--minute_ckpt", type=str, required=True,
                        help="Checkpoint for minute model (state_dict).")
    parser.add_argument("--rot_ckpt", type=str, required=True,
                        help="Checkpoint for rotation model (state_dict).")
    parser.add_argument("--k", type=int, default=3,
                        help="Top-k for explanations (default 3).")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional: write JSON explanation to this path.")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu / auto (default auto).")

    args = parser.parse_args()

    if args.device is None or args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    root_dir = Path(args.root_dir)

    # Decide image path + (optional) ground truth from CSV
    gt_time = None
    chosen_file = None

    if args.file is not None:
        p = Path(args.file)
        img_path = p if p.is_absolute() else (root_dir / args.file)
        chosen_file = str(args.file)
    else:
        csv_path = root_dir / args.csv
        df = pd.read_csv(csv_path)
        if "status" in df.columns:
            df = df[df["status"].astype(str) == "ok"].copy()
        if args.subset is not None:
            prefix = f"{args.subset}/"
            df = df[df["file"].astype(str).str.startswith(prefix)].copy()

        df = df.reset_index(drop=True)
        if len(df) == 0:
            raise RuntimeError("No rows left after filtering. Check --subset or CSV contents.")

        if args.index < 0 or args.index >= len(df):
            raise IndexError(f"--index {args.index} out of range (0..{len(df)-1}).")

        row = df.iloc[args.index]
        chosen_file = str(row["file"])
        img_path = root_dir / chosen_file

        # ground truth time if present
        if "time_h" in row and "time_m" in row:
            try:
                gt_time = f"{int(row['time_h'])}:{int(row['time_m']):02d}"
            except Exception:
                gt_time = None

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    # Transform (match training)
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(img_path).convert("RGB")
    x = tfm(img).unsqueeze(0).to(device)  # (1,3,224,224)

    # Build + load models
    hour_model = build_hour_minute_model(device)
    minute_model = build_hour_minute_model(device)
    rot_model = build_rot_model(device)

    load_ckpt(hour_model, args.hour_ckpt, device)
    load_ckpt(minute_model, args.minute_ckpt, device)
    load_ckpt(rot_model, args.rot_ckpt, device)

    hour_model.eval()
    minute_model.eval()
    rot_model.eval()

    with torch.no_grad():
        p_h = ensure_probs(hour_model(x))
        p_m = ensure_probs(minute_model(x))
        p_r = ensure_probs(rot_model(x))

    # Top-k
    k = max(1, args.k)
    hour_top = topk_with_labels(p_h, HOUR_LABELS, k)
    minute_top = topk_with_labels(p_m, MINUTE_LABELS, k)
    rot_top = topk_with_labels(p_r, ROT_LABELS_DEG, k)

    # Compose time distribution from top-k combinations
    time_probs, time_paths, coverage = compose_time_distribution(hour_top, minute_top, rot_top, k)

    # Sort times
    time_sorted = sorted(time_probs.items(), key=lambda kv: kv[1], reverse=True)
    time_topk = time_sorted[:k]  # show k times

    conf_max = time_topk[0][1] if time_topk else 0.0
    ent = entropy_of_distribution(time_probs)

    # Main explanations for best time: take top 3 contributing paths
    main_expl = []
    if time_topk:
        best_time = time_topk[0][0]
        paths = time_paths.get(best_time, [])
        paths_sorted = sorted(paths, key=lambda d: d["p"], reverse=True)[:3]
        main_expl = paths_sorted

    # Print
    print("========================================")
    print("Explain (k=3)")
    print(f"device: {device}")
    print(f"image: {img_path}")
    if gt_time is not None:
        print(f"ground_truth: {gt_time}")
    print("----------------------------------------")
    print("Top-k (hour):   ", hour_top)
    print("Top-k (minute): ", minute_top)
    print("Top-k (rot):    ", rot_top)
    print("----------------------------------------")
    print("Top time candidates (aggregated from top-k combos):")
    for t, p in time_topk:
        print(f"  {t:>5}  p={p:.6f}")
    print("----------------------------------------")
    print(f"coverage_mass(top-k combos): {coverage:.6f}")
    print(f"confidence_max(top time mass): {conf_max:.6f}")
    print(f"entropy(normalized over produced times): {ent:.6f} (nats)")
    if main_expl:
        print("----------------------------------------")
        print("Main paths for best time:")
        for i, d in enumerate(main_expl, 1):
            print(f"  #{i} p={d['p']:.6f}  (Hraw={d['Hraw']}, Mraw={d['Mraw']}, Rdeg={d['Rdeg']})"
                  f" -> (H={d['H']}, M={d['M']}) time={d['time']}"
                  f" | raw_deg(H,M)=({d['Hraw_deg']},{d['Mraw_deg']})"
                  f" corr_deg(H,M)=({d['Hcorr_deg']},{d['Mcorr_deg']})")
    print("========================================")

    # JSON output
    explanation = {
        "image": str(img_path),
        "file": chosen_file,
        "ground_truth": gt_time,
        "k": k,
        "hour_topk": hour_top,
        "minute_topk": minute_top,
        "rot_topk": rot_top,
        "time_topk": time_topk,
        "coverage_mass": coverage,
        "confidence_max": conf_max,
        "entropy_nats": ent,
        "main_explanations": main_expl,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(explanation, f, ensure_ascii=False, indent=2)
        print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
