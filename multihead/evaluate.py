"""
evaluate.py
===========
Evaluation script for the multi-head clock recognition model.

Performs:
  - Naive decode: argmax each head -> rotation correction -> time conversion
  - Per-class accuracy analysis
  - Overall metrics: exact match, hour accuracy, minute accuracy
"""

import argparse
import os
import sys

# Ensure parent directory is in path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

from multihead.networks import MultiHeadClockNet
from multihead.dataset import ClockDataset, get_transform


# ---------------------------------------------------------------------------
# Naive decoding (mirrors the ProbLog logic in Python)
# ---------------------------------------------------------------------------

ROT_STEPS = {0: 0, 1: 3, 2: 6, 3: 9}


def correct_idx(image_idx: int, steps: int) -> int:
    """Image coords -> canonical coords (mod 12)."""
    return (image_idx - steps + 120) % 12


def idx_to_hour(canon_idx: int) -> int:
    return 12 if canon_idx == 0 else canon_idx


def idx_to_minute(canon_idx: int) -> int:
    return canon_idx * 5


def naive_decode(rot_cls: int, hour_img_cls: int, minute_img_cls: int):
    """Decode time from raw head predictions.

    Applies rotation correction and hour-minute coupling constraint.
    Returns (hour, minute).
    """
    steps = ROT_STEPS[rot_cls]
    h_pos = correct_idx(hour_img_cls, steps)
    m_pos = correct_idx(minute_img_cls, steps)

    minute = idx_to_minute(m_pos)

    # Hour depends on minute: if minute >= 30, hour hand is past the tick
    if m_pos < 6:
        hour = idx_to_hour(h_pos)
    else:
        h_prev = (h_pos - 1 + 12) % 12
        hour = idx_to_hour(h_prev)

    return hour, minute


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(args, device):
    """Run evaluation on the test set."""
    print("=" * 60)
    print("Evaluating Multi-Head Clock Model")
    print("=" * 60)

    # Load model
    shared_net = MultiHeadClockNet().to(device)
    if args.weights:
        shared_net.load_state_dict(
            torch.load(args.weights, map_location=device)
        )
        print(f"Loaded weights from {args.weights}")
    else:
        print("WARNING: No weights loaded, using random initialisation")

    shared_net.eval()

    # Load dataset
    test_ds = ClockDataset(
        args.data_root, args.annotations, subset=args.subset,
        transform=get_transform(train=False), max_samples=args.max_samples,
    )

    def collate_fn(batch):
        imgs, labels = zip(*batch)
        return torch.stack(imgs), list(labels)

    test_loader = TorchDataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"Evaluating on {len(test_ds)} samples ({args.subset} set)")

    # Metrics
    exact_correct = 0
    hour_correct = 0
    minute_correct = 0
    rot_correct = 0
    total = 0

    # Per-time-class tracking
    time_results = {}  # (h, m) -> {"correct": int, "total": int}

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Evaluating"):
            imgs = imgs.to(device)

            rot_probs = shared_net.forward_rot(imgs)
            hour_probs = shared_net.forward_hour(imgs)
            min_probs = shared_net.forward_minute(imgs)

            shared_net.clear_cache()

            for i in range(len(labels)):
                lab = labels[i]
                gt_h, gt_m = lab.time_h, lab.time_m
                gt_rot = lab.rot_cls

                pred_rot = rot_probs[i].argmax().item()
                pred_h_img = hour_probs[i].argmax().item()
                pred_m_img = min_probs[i].argmax().item()

                pred_h, pred_m = naive_decode(pred_rot, pred_h_img, pred_m_img)

                # Rotation accuracy
                if pred_rot == gt_rot:
                    rot_correct += 1

                # Hour accuracy
                if pred_h == gt_h:
                    hour_correct += 1

                # Minute accuracy
                if pred_m == gt_m:
                    minute_correct += 1

                # Exact match
                is_correct = (pred_h == gt_h) and (pred_m == gt_m)
                if is_correct:
                    exact_correct += 1

                # Per-class tracking
                key = (gt_h, gt_m)
                if key not in time_results:
                    time_results[key] = {"correct": 0, "total": 0}
                time_results[key]["total"] += 1
                if is_correct:
                    time_results[key]["correct"] += 1

                total += 1

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total samples:     {total}")
    print(f"Exact match acc:   {exact_correct/total:.4f} ({exact_correct}/{total})")
    print(f"Hour accuracy:     {hour_correct/total:.4f} ({hour_correct}/{total})")
    print(f"Minute accuracy:   {minute_correct/total:.4f} ({minute_correct}/{total})")
    print(f"Rotation accuracy: {rot_correct/total:.4f} ({rot_correct}/{total})")

    # Per-time-class analysis
    if args.verbose:
        print("\n" + "-" * 60)
        print("Per-time-class accuracy:")
        print("-" * 60)
        for (h, m) in sorted(time_results.keys()):
            info = time_results[(h, m)]
            acc = info["correct"] / info["total"] if info["total"] > 0 else 0
            print(f"  {h:2d}:{m:02d}  {acc:.4f}  ({info['correct']}/{info['total']})")

    # Find worst-performing classes
    print("\nTop-10 worst classes:")
    sorted_classes = sorted(
        time_results.items(),
        key=lambda x: x[1]["correct"] / max(x[1]["total"], 1),
    )
    for (h, m), info in sorted_classes[:10]:
        acc = info["correct"] / max(info["total"], 1)
        print(f"  {h:2d}:{m:02d}  {acc:.4f}  ({info['correct']}/{info['total']})")

    return exact_correct / total if total > 0 else 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Multi-Head Clock Model"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to clock_kaggle directory")
    parser.add_argument("--annotations", type=str, default=None,
                        help="Path to annotations.csv")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--subset", type=str, default="test",
                        choices=["train", "valid", "test"])
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-time-class accuracy")

    args = parser.parse_args()

    if args.annotations is None:
        args.annotations = os.path.join(args.data_root, "annotations.csv")

    if not os.path.exists(args.annotations):
        print(f"ERROR: {args.annotations} not found")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    evaluate(args, device)


if __name__ == "__main__":
    main()
