# eval_constraints.py
import os
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def build_resnet18(n_classes: int) -> nn.Module:
    """Match your saved keys: backbone.* and backbone.fc.0.*"""
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Sequential(nn.Linear(512, n_classes))
    model = nn.Module()
    model.backbone = backbone
    return model

@torch.no_grad()
def forward_probs(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    logits = model.backbone(x)
    return F.softmax(logits, dim=1).squeeze(0)  # (C,)

def infer_time_distribution(pR, pH0, pM0, logic: str, rot_steps_per_90: int = 3):
    """
    pR: (4,) dial rotation probs for R in {0,1,2,3} (0,90,180,270 clockwise)
    pH0: (12,) hour hand class probs in image coords (0..11)
    pM0: (12,) minute hand class probs in image coords (0..11)

    Returns:
      best_hour (1..12), best_minute (0..55 step 5), best_prob
    """
    # Distribution over canonical (hour_pos 0..11 where 0 means 12), (minute_pos 0..11)
    dist = torch.zeros((12, 12), dtype=torch.float64)

    for r in range(4):
        pr = float(pR[r].item())
        if pr == 0.0:
            continue
        shift = (rot_steps_per_90 * r) % 12  # 90deg -> 3 steps

        # Correct to canonical dial coordinates
        for h0 in range(12):
            ph = pr * float(pH0[h0].item())
            if ph == 0.0:
                continue
            hpos = (h0 + shift) % 12

            for m0 in range(12):
                p = ph * float(pM0[m0].item())
                if p == 0.0:
                    continue
                mpos = (m0 + shift) % 12

                if logic == "constrained":
                    # minute>=30  <=> mpos>=6 (since each step is 5 minutes)
                    if mpos >= 6:
                        hpos_eff = (hpos - 1) % 12
                    else:
                        hpos_eff = hpos
                elif logic == "unconstrained":
                    # no coupling between hour and minute
                    hpos_eff = hpos
                else:
                    raise ValueError("logic must be constrained or unconstrained")

                dist[hpos_eff, mpos] += p

    # MAP
    flat_idx = int(torch.argmax(dist).item())
    hpos = flat_idx // 12
    mpos = flat_idx % 12
    best_prob = float(dist[hpos, mpos].item())

    best_hour = 12 if hpos == 0 else hpos
    best_minute = mpos * 5
    return best_hour, best_minute, best_prob

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--status", default="ok")
    ap.add_argument("--dial_pth", required=True)
    ap.add_argument("--hour_pth", required=True)
    ap.add_argument("--minute_pth", required=True)
    ap.add_argument("--logic", choices=["constrained", "unconstrained"], required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if "split" in df.columns:
        df = df[df["split"] == args.split]
    if "status" in df.columns and args.status is not None:
        df = df[df["status"] == args.status]

    # Required label columns
    for col in ["file", "time_h", "time_m"]:
        if col not in df.columns:
            raise ValueError(f"CSV must contain column '{col}'")

    device = torch.device(args.device)

    # Build models with correct output dims from state_dict
    sd_d = torch.load(args.dial_pth, map_location="cpu")
    sd_h = torch.load(args.hour_pth, map_location="cpu")
    sd_m = torch.load(args.minute_pth, map_location="cpu")

    dial_c = sd_d["backbone.fc.0.weight"].shape[0]
    hour_c = sd_h["backbone.fc.0.weight"].shape[0]
    min_c  = sd_m["backbone.fc.0.weight"].shape[0]

    dial_model = build_resnet18(dial_c).to(device).eval()
    hour_model = build_resnet18(hour_c).to(device).eval()
    min_model  = build_resnet18(min_c).to(device).eval()

    dial_model.load_state_dict(sd_d, strict=True)
    hour_model.load_state_dict(sd_h, strict=True)
    min_model.load_state_dict(sd_m, strict=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    n = 0
    correct_time = 0
    correct_hour = 0
    correct_min  = 0

    for _, row in df.iterrows():
        rel = row["file"]
        img_path = rel if os.path.isabs(rel) else os.path.join(args.data_root, rel)
        if not os.path.exists(img_path):
            # skip silently, but you can print if needed
            continue

        img = Image.open(img_path).convert("RGB")
        x = tfm(img).unsqueeze(0).to(device)

        pR  = forward_probs(dial_model, x)
        pH0 = forward_probs(hour_model, x)
        pM0 = forward_probs(min_model, x)

        # sanity: this script expects dial=4, hands=12. It can still run otherwise, but time mapping assumes these.
        if pR.numel() != 4 or pH0.numel() != 12 or pM0.numel() != 12:
            raise ValueError(f"Expected dial=4 and hands=12, got dial={pR.numel()} hour={pH0.numel()} minute={pM0.numel()}")

        pred_h, pred_m, _ = infer_time_distribution(pR, pH0, pM0, logic=args.logic)

        true_h = int(row["time_h"])
        true_m = int(row["time_m"])

        n += 1
        correct_time += int((pred_h == true_h) and (pred_m == true_m))
        correct_hour += int(pred_h == true_h)
        correct_min  += int(pred_m == true_m)

    if n == 0:
        print("No samples evaluated. Check paths / split / status.")
        return

    print(f"[{args.logic}] N={n}")
    print(f"  time_acc  = {correct_time/n:.4f}")
    print(f"  hour_acc  = {correct_hour/n:.4f}")
    print(f"  minute_acc= {correct_min/n:.4f}")

if __name__ == "__main__":
    main()
