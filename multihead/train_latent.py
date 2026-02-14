"""
train_latent.py
===============
2-stage training for the latent-variable clock recognition model.

Stages:
  1. Backbone Warmup  - 144-class time classification with CrossEntropyLoss
  2. Latent DPB       - time(X, Hour, Minute) only; R, H, M are latent variables

No component labels (rotation, hour angle, minute angle) are used.
All supervision comes from time labels extracted from folder names.

Usage:
  python multihead/train_latent.py --data_root ../../clock_kaggle --run all
  python multihead/train_latent.py --data_root ../../clock_kaggle --run warmup
  python multihead/train_latent.py --data_root ../../clock_kaggle --run latent
"""

import argparse
import os
import random
import sys

# Ensure parent directory is in path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm

# DeepProbLog imports
from deepproblog.train import train_model
from deepproblog.model import Model
from deepproblog.network import Network
from deepproblog.dataset import DataLoader as DPBDataLoader
from deepproblog.engines import ExactEngine

# Local imports
from multihead.networks import (
    MultiHeadClockNet,
    WarmupNet,
    RotAdapter,
    HourAdapter,
    MinuteAdapter,
    transfer_backbone_weights,
)
from multihead.dataset import (
    LatentClockDataset,
    DPBLatentTimeDataset,
    get_transform,
)


def seed_everything(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def collate_fn(batch):
    """Custom collate for LatentClockDataset (img, LatentClockLabels)."""
    imgs, labels = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    return imgs, list(labels)


# ---------------------------------------------------------------------------
# Stage 1: Backbone Warmup (144-class, time labels only)
# ---------------------------------------------------------------------------

def train_warmup(args, device):
    """Train backbone with 144-class time classification."""
    print("=" * 60)
    print("STAGE 1: Backbone Warmup (144-class classification)")
    print("=" * 60)

    warmup_net = WarmupNet(num_classes=144).to(device)
    optimizer = torch.optim.Adam(warmup_net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_ds = LatentClockDataset(
        args.data_root, subset="train",
        transform=get_transform(train=True), max_samples=args.max_samples,
    )
    valid_ds = LatentClockDataset(
        args.data_root, subset="valid",
        transform=get_transform(train=False), max_samples=args.max_samples,
    )

    train_loader = TorchDataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )
    valid_loader = TorchDataLoader(
        valid_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    print(f"Train: {len(train_ds)} samples, Valid: {len(valid_ds)} samples")

    for epoch in range(1, args.epochs_warmup + 1):
        warmup_net.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc=f"Warmup Epoch {epoch}/{args.epochs_warmup}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            targets = torch.tensor(
                [lab.class_index for lab in labels], dtype=torch.long, device=device
            )

            logits = warmup_net(imgs)
            loss = criterion(logits, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += len(imgs)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        train_acc = correct / total
        train_loss = total_loss / total

        # Validation
        warmup_net.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, labels in valid_loader:
                imgs = imgs.to(device)
                targets = torch.tensor(
                    [lab.class_index for lab in labels], dtype=torch.long, device=device
                )
                logits = warmup_net(imgs)
                preds = logits.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += len(imgs)

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"  Epoch {epoch}: train_loss={train_loss:.4f} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")

    # Save warmup weights
    os.makedirs(args.save_dir, exist_ok=True)
    warmup_path = os.path.join(args.save_dir, "warmup_backbone.pth")
    torch.save(warmup_net.state_dict(), warmup_path)
    print(f"Saved warmup weights to {warmup_path}")

    return warmup_net


# ---------------------------------------------------------------------------
# Stage 2: Latent DPB Training (time labels only)
# ---------------------------------------------------------------------------

def train_latent(args, device, shared_net: MultiHeadClockNet):
    """Train with DeepProbLog time(X, Hour, Minute) — all heads latent."""
    print("=" * 60)
    print("STAGE 2: Latent DPB Training (time labels only)")
    print("  R, H, M are latent variables marginalised by DeepProbLog")
    print("=" * 60)

    shared_net.to(device)

    # Create adapters (all share the same backbone)
    rot_adapter = RotAdapter(shared_net)
    hour_adapter = HourAdapter(shared_net)
    min_adapter = MinuteAdapter(shared_net)

    # DeepProbLog Networks
    net_rot = Network(rot_adapter, "net_rot", batching=True)
    net_hour = Network(hour_adapter, "net_hour", batching=True)
    net_minute = Network(min_adapter, "net_minute", batching=True)

    # Differential learning rates: backbone lower, heads higher
    backbone_params = list(shared_net.features.parameters()) + \
                      list(shared_net.avgpool.parameters())
    head_params = list(shared_net.head_rot.parameters()) + \
                  list(shared_net.head_hour.parameters()) + \
                  list(shared_net.head_minute.parameters())

    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": args.lr / 10},
        {"params": head_params, "lr": args.lr},
    ])

    net_rot.optimizer = optimizer
    net_hour.optimizer = optimizer
    net_minute.optimizer = optimizer

    # Build DeepProbLog model
    model = Model(args.prolog, [net_rot, net_hour, net_minute])
    model.set_engine(ExactEngine(model))

    # Datasets — loaded from folder structure, no annotations.csv
    train_ds = LatentClockDataset(
        args.data_root, subset="train",
        transform=get_transform(train=True), max_samples=args.max_samples,
    )
    valid_ds = LatentClockDataset(
        args.data_root, subset="valid",
        transform=get_transform(train=False), max_samples=args.max_samples,
    )

    train_dpb = DPBLatentTimeDataset(train_ds)
    valid_dpb = DPBLatentTimeDataset(valid_ds)

    train_loader = DPBDataLoader(train_dpb, batch_size=args.batch_size_dpb)
    valid_loader = DPBDataLoader(valid_dpb, batch_size=args.batch_size_dpb)

    print(f"Train: {len(train_dpb)} queries, Valid: {len(valid_dpb)} queries")

    # Train with DeepProbLog
    train_model(
        model,
        train_loader,
        args.epochs_latent,
        test_iter=valid_loader,
        log_iter=args.log_iter,
        profile=0,
    )

    # Save weights
    os.makedirs(args.save_dir, exist_ok=True)
    path = os.path.join(args.save_dir, "final_latent.pth")
    torch.save(shared_net.state_dict(), path)
    print(f"Saved latent-trained weights to {path}")

    return shared_net, model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Latent-Variable Clock Recognition - 2-Stage Training"
    )
    parser.add_argument("--data_root", type=str, required=True,
                        help="Path to clock_kaggle directory")
    parser.add_argument("--prolog", type=str, default=None,
                        help="Path to ProbLog file (default: models/clock_latent.pl)")
    parser.add_argument("--run", type=str, default="all",
                        choices=["warmup", "latent", "all"],
                        help="Which stage(s) to run")
    parser.add_argument("--epochs_warmup", type=int, default=10)
    parser.add_argument("--epochs_latent", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for Stage 1 warmup")
    parser.add_argument("--batch_size_dpb", type=int, default=4,
                        help="Batch size for DeepProbLog Stage 2")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="weights_latent",
                        help="Directory to save model weights")
    parser.add_argument("--log_iter", type=int, default=100,
                        help="DeepProbLog logging frequency")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset size for debugging")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_warmup", type=str, default=None,
                        help="Path to pre-trained warmup weights")

    args = parser.parse_args()

    # Resolve default Prolog path
    if args.prolog is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        args.prolog = os.path.join(script_dir, "..", "models", "clock_latent.pl")

    # Validate
    if not os.path.isdir(args.data_root):
        print(f"ERROR: data_root not found at {args.data_root}")
        sys.exit(1)
    if not os.path.exists(args.prolog):
        print(f"ERROR: ProbLog file not found at {args.prolog}")
        sys.exit(1)

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"ProbLog: {args.prolog}")

    # ---------------------------------------------------------------
    # Stage 1: Backbone Warmup
    # ---------------------------------------------------------------
    warmup_net = None
    if args.run in ("warmup", "all"):
        warmup_net = train_warmup(args, device)

    # ---------------------------------------------------------------
    # Stage 2: Latent DPB Training
    # ---------------------------------------------------------------
    shared_net = MultiHeadClockNet().to(device)

    # Transfer backbone from warmup
    if warmup_net is not None:
        transfer_backbone_weights(warmup_net, shared_net)
    elif args.load_warmup is not None:
        wnet = WarmupNet(num_classes=144)
        wnet.load_state_dict(torch.load(args.load_warmup, map_location=device))
        transfer_backbone_weights(wnet, shared_net)
        print(f"Loaded warmup weights from {args.load_warmup}")

    if args.run in ("latent", "all"):
        shared_net, _ = train_latent(args, device, shared_net)

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
