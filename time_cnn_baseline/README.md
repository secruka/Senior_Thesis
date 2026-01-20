# Baseline: Clock Time CNN (144-class)

This is a simple **baseline CNN** that predicts the clock time directly from the image as a **144-class classification** problem (12 hours × 12 five-minute bins).

It uses the provided `clocks.csv` (columns: `class index`, `filepaths`, `labels`, `data set`) and expects your images to be under a root directory like:

```
/Users/ruka/Senior_Thesis/clock_kaggle/train/1-00/0.jpg
```

`filepaths` in the CSV are relative (e.g., `train/1-00/0.jpg`), so the loader joins them with `--data_root`.

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m time_cnn.train \
  --data_root /Users/ruka/Senior_Thesis/clock_kaggle \
  --csv_path  /Users/ruka/Senior_Thesis/clock_kaggle/clocks.csv \
  --model resnet18 \
  --epochs 20 \
  --batch_size 64 \
  --lr 3e-4 \
  --num_workers 4
```

Outputs are saved under `runs/<run_name>/` (checkpoint + class map).

## Inference (single image)

```bash
python -m time_cnn.infer \
  --ckpt runs/<run_name>/best.pt \
  --img  /Users/ruka/Senior_Thesis/clock_kaggle/train/1-00/0.jpg
```

## Notes on augmentation

Avoid flips/rotations: they *change the time label* for analog clocks. This baseline only uses light color jitter + random erasing.
