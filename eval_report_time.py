import os
import argparse
import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from problog.logic import Term, Var, Constant
from deepproblog.query import Query

# あなたの現行実装（softlabel版）から流用
from train_clock_integrated_heatmap import (
    RotationCsvTorchDataset,
    build_deepproblog_model,
)

# -------------------------
# helpers: decoding (same as Prolog logic)
# -------------------------
ROT_STEPS = {0: 0, 1: 3, 2: 6, 3: 9}

def correct_idx(img_idx: int, steps: int) -> int:
    return (int(img_idx) - int(steps) + 120) % 12

def prev_idx(i: int) -> int:
    return (int(i) - 1 + 12) % 12

def idx_to_hour(hour_idx: int) -> int:
    return 12 if int(hour_idx) == 0 else int(hour_idx)

def idx_to_minute(min_idx: int) -> int:
    return int(min_idx) * 5

def naive_decode_time(dial_pred: int, h_img_pred: int, m_img_pred: int):
    """
    非DPB（統合しない）:
      argmax dial/hour/minute を使って
      回転補正 -> minute=5刻み -> minute<30 なら hourそのまま else hour-1
    """
    steps = ROT_STEPS[int(dial_pred)]
    hpos = correct_idx(h_img_pred, steps)
    midx = correct_idx(m_img_pred, steps)

    minute = idx_to_minute(midx)
    if midx < 6:
        hour = idx_to_hour(hpos)
    else:
        hour = idx_to_hour(prev_idx(hpos))
    return hour, minute

def parse_out_key_to_hour_minute(key):
    """
    DeepProbLog solveの戻りのキー形式に幅を持たせて対応。
    """
    # (H,M) の tuple/list
    if isinstance(key, (tuple, list)) and len(key) == 2:
        return int(str(key[0])), int(str(key[1]))

    # Term っぽい
    if hasattr(key, "functor") and hasattr(key, "args"):
        fun = str(key.functor)
        args = list(getattr(key, "args", []))
        # time(X,H,M)
        if fun == "time" and len(args) >= 3:
            return int(str(args[1])), int(str(args[2]))
        # 2引数Term (H,M)
        if len(args) == 2:
            return int(str(args[0])), int(str(args[1]))

    # それ以外は失敗
    raise ValueError(f"Cannot parse key: {key} ({type(key)})")

def collate_labels_list(batch):
    imgs = torch.stack([b[0] for b in batch])
    labels = [b[1] for b in batch]
    return imgs, labels

# -------------------------
# main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test", "valid"])
    ap.add_argument("--prolog", default="models/clock_integrated.pl")

    ap.add_argument("--dial_pth", default=None)
    ap.add_argument("--hour_pth", required=True)
    ap.add_argument("--minute_pth", required=True)
    ap.add_argument("--dial_arch", default="heatmap", choices=["heatmap", "cls"])

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--max_samples", type=int, default=None)

    ap.add_argument("--run_dpb", action="store_true", help="DPB統合推論（time/3）も回す（遅いので必要時だけ）")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--print_examples", type=int, default=15)
    ap.add_argument("--out_dir", default="eval_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    csv_path = args.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(args.data_root, csv_path)

    # same normalization as training
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    torch_ds = RotationCsvTorchDataset(
        data_root=args.data_root,
        csv_path=csv_path,
        subset=args.split,
        transform=tfm,
        max_samples=args.max_samples,
    )

    loader = DataLoader(
        torch_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_labels_list,
    )

    # build DPB model (we can still use cnn outputs even if we don't run DPB solve)
    model, cnn_dial, cnn_hour, cnn_minute = build_deepproblog_model(
        prolog_file=args.prolog,
        device=device,
        lr_dial=1e-4,
        lr_hands=1e-4,
        dial_arch=args.dial_arch,
        weights_dial=args.dial_pth,
        weights_hour=args.hour_pth,
        weights_minute=args.minute_pth,
    )
    model.eval()
    cnn_dial.eval(); cnn_hour.eval(); cnn_minute.eval()

    # aggregate counters
    n = 0
    dial_ok = 0
    hour_ok = 0
    minute_ok = 0
    hands_joint_ok = 0
    all3_joint_ok = 0  # dial & hour & minute all correct

    naive_time_ok = 0
    dpb_time_ok = 0

    # per-time stats: key=(H,M)
    per_time = defaultdict(lambda: {"n": 0, "naive_ok": 0, "dpb_ok": 0})

    # per-sample logging
    rows = []

    # For DPB solve (time/3 with variables)
    q_time = Term("time", Var("X"), Var("H"), Var("M"))

    for imgs, labels_list in loader:
        imgs = imgs.to(device)

        with torch.no_grad():
            dial_probs = cnn_dial(imgs)
            hour_probs = cnn_hour(imgs)
            minute_probs = cnn_minute(imgs)

        dial_pred = torch.argmax(dial_probs, dim=1).cpu().numpy()
        hour_pred = torch.argmax(hour_probs, dim=1).cpu().numpy()
        minute_pred = torch.argmax(minute_probs, dim=1).cpu().numpy()

        dial_conf = dial_probs.max(dim=1).values.detach().cpu().numpy()
        hour_conf = hour_probs.max(dim=1).values.detach().cpu().numpy()
        minute_conf = minute_probs.max(dim=1).values.detach().cpu().numpy()

        B = len(labels_list)
        for b in range(B):
            lab = labels_list[b]
            gt_rot = int(lab.rot_cls)
            gt_himg = int(lab.h_img_cls)
            gt_mimg = int(lab.m_img_cls)
            gt_hour = int(lab.hour_time)
            gt_min = int(lab.minute_time)

            d_pred = int(dial_pred[b])
            h_pred = int(hour_pred[b])
            m_pred = int(minute_pred[b])

            d_ok = (d_pred == gt_rot)
            h_ok = (h_pred == gt_himg)
            m_ok = (m_pred == gt_mimg)

            hj_ok = (h_ok and m_ok)
            a3_ok = (d_ok and h_ok and m_ok)

            dial_ok += int(d_ok)
            hour_ok += int(h_ok)
            minute_ok += int(m_ok)
            hands_joint_ok += int(hj_ok)
            all3_joint_ok += int(a3_ok)

            # naive time
            naive_h, naive_m = naive_decode_time(d_pred, h_pred, m_pred)
            naive_ok = (naive_h == gt_hour and naive_m == gt_min)
            naive_time_ok += int(naive_ok)

            dpb_pred_h = None
            dpb_pred_m = None
            dpb_best_p = None
            topk_list = None
            dpb_ok = None

            if args.run_dpb:
                img_const = Constant(imgs[b].detach().cpu())  # DeepProbLog expects CPU tensor in Constant
                query = Query(q_time, {Var("X"): img_const}, output_ind=(1, 2))
                ans = model.solve([query])[0]

                if len(ans.result) > 0:
                    parsed = []
                    for k, p in ans.result.items():
                        try:
                            hh, mm = parse_out_key_to_hour_minute(k)
                            parsed.append((hh, mm, float(p)))
                        except Exception:
                            continue

                    if len(parsed) > 0:
                        parsed.sort(key=lambda x: x[2], reverse=True)
                        dpb_pred_h, dpb_pred_m, dpb_best_p = parsed[0]
                        topk_list = parsed[: args.topk]
                        dpb_ok = (dpb_pred_h == gt_hour and dpb_pred_m == gt_min)
                        dpb_time_ok += int(dpb_ok)
                    else:
                        dpb_ok = False
                else:
                    dpb_ok = False

            per_time[(gt_hour, gt_min)]["n"] += 1
            per_time[(gt_hour, gt_min)]["naive_ok"] += int(naive_ok)
            if args.run_dpb:
                per_time[(gt_hour, gt_min)]["dpb_ok"] += int(dpb_ok)

            rows.append({
                "gt_hour": gt_hour, "gt_min": gt_min,
                "gt_rot": gt_rot, "gt_himg": gt_himg, "gt_mimg": gt_mimg,
                "dial_pred": d_pred, "dial_ok": int(d_ok), "dial_conf": float(dial_conf[b]),
                "hour_pred": h_pred, "hour_ok": int(h_ok), "hour_conf": float(hour_conf[b]),
                "min_pred": m_pred, "min_ok": int(m_ok), "min_conf": float(minute_conf[b]),
                "hands_joint_ok": int(hj_ok),
                "all3_joint_ok": int(a3_ok),
                "naive_hour": naive_h, "naive_min": naive_m, "naive_time_ok": int(naive_ok),
                "dpb_hour": dpb_pred_h, "dpb_min": dpb_pred_m, "dpb_best_p": dpb_best_p,
                "dpb_time_ok": int(dpb_ok) if args.run_dpb else None,
                "dpb_topk": str(topk_list) if topk_list is not None else None,
            })

            n += 1

    # summary
    dial_acc = dial_ok / max(n, 1)
    hour_acc = hour_ok / max(n, 1)
    minute_acc = minute_ok / max(n, 1)
    hands_joint_acc = hands_joint_ok / max(n, 1)
    empirical_all3_joint = all3_joint_ok / max(n, 1)

    product_est = dial_acc * hands_joint_acc
    naive_time_acc = naive_time_ok / max(n, 1)

    print("\n=== Single / combined ===")
    print(f"N={n}")
    print(f"dial_acc             = {dial_acc:.4f}")
    print(f"hands hour_acc        = {hour_acc:.4f}")
    print(f"hands minute_acc      = {minute_acc:.4f}")
    print(f"hands joint_acc       = {hands_joint_acc:.4f}")
    print(f"dial_acc * hands_joint= {product_est:.4f}  (independence estimate)")
    print(f"empirical_all3_joint  = {empirical_all3_joint:.4f}  (dial&hour&min same-sample)")

    print("\n=== Time (non-DPB) ===")
    print(f"naive_time_exact_acc  = {naive_time_acc:.4f}")

    if args.run_dpb:
        dpb_time_acc = dpb_time_ok / max(n, 1)
        print("\n=== Time (DPB integrated) ===")
        print(f"dpb_time_exact_acc    = {dpb_time_acc:.4f}")
        print(f"delta(dpb-naive)      = {dpb_time_acc - naive_time_acc:+.4f}")

    # save CSVs
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "per_sample.csv"), index=False)

    # per-time table (144 classes)
    per_rows = []
    for (h, m), d in per_time.items():
        acc_naive = d["naive_ok"] / max(d["n"], 1)
        acc_dpb = (d["dpb_ok"] / max(d["n"], 1)) if args.run_dpb else None
        per_rows.append({
            "hour": h, "minute": m,
            "n": d["n"],
            "naive_acc": acc_naive,
            "dpb_acc": acc_dpb,
            "delta": (acc_dpb - acc_naive) if args.run_dpb else None,
        })
    per_df = pd.DataFrame(per_rows).sort_values(["hour", "minute"])
    per_df.to_csv(os.path.join(args.out_dir, "per_time.csv"), index=False)

    # show strongest/weakest times
    print("\n=== Per-time (naive) top/bottom ===")
    tmp = per_df.copy()
    tmp["naive_acc"] = pd.to_numeric(tmp["naive_acc"], errors="coerce")
    tmp_sorted = tmp.sort_values("naive_acc", ascending=False)
    print("Top 10 easiest (naive):")
    print(tmp_sorted.head(10)[["hour","minute","n","naive_acc"]].to_string(index=False))
    print("\nBottom 10 hardest (naive):")
    print(tmp_sorted.tail(10)[["hour","minute","n","naive_acc"]].to_string(index=False))

    if args.run_dpb:
        tmp["dpb_acc"] = pd.to_numeric(tmp["dpb_acc"], errors="coerce")
        tmp["delta"] = pd.to_numeric(tmp["delta"], errors="coerce")
        inc = tmp.sort_values("delta", ascending=False)
        dec = tmp.sort_values("delta", ascending=True)
        print("\n=== Per-time improvement (dpb - naive) ===")
        print("Top 10 improved:")
        print(inc.head(10)[["hour","minute","n","naive_acc","dpb_acc","delta"]].to_string(index=False))
        print("\nTop 10 worsened:")
        print(dec.head(10)[["hour","minute","n","naive_acc","dpb_acc","delta"]].to_string(index=False))

        # rescued / harmed examples
        rescued = df[(df["naive_time_ok"] == 0) & (df["dpb_time_ok"] == 1)].copy()
        harmed  = df[(df["naive_time_ok"] == 1) & (df["dpb_time_ok"] == 0)].copy()

        rescued.to_csv(os.path.join(args.out_dir, "rescued.csv"), index=False)
        harmed.to_csv(os.path.join(args.out_dir, "harmed.csv"), index=False)

        print("\n=== DPB interpretability signals ===")
        print(f"rescued (naive wrong -> dpb correct): {len(rescued)}")
        print(f"harmed  (naive correct -> dpb wrong): {len(harmed)}")

        k = args.print_examples
        if k > 0 and len(rescued) > 0:
            print("\n--- rescued examples (showing dpb_topk) ---")
            print(rescued.head(k)[["gt_hour","gt_min","naive_hour","naive_min","dpb_hour","dpb_min","dpb_best_p","dpb_topk"]].to_string(index=False))
        if k > 0 and len(harmed) > 0:
            print("\n--- harmed examples (showing dpb_topk) ---")
            print(harmed.head(k)[["gt_hour","gt_min","naive_hour","naive_min","dpb_hour","dpb_min","dpb_best_p","dpb_topk"]].to_string(index=False))

    print(f"\nSaved: {args.out_dir}/per_sample.csv, per_time.csv (and rescued/harmed if run_dpb)")

if __name__ == "__main__":
    main()
