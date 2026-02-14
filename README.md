# Senior_Thesis
Interpretable Analog Clock Reading with Neuro-Symbolic Constraints (DeepProbLog)

# DeepProbLog-based Analog Clock Reading

## Overview
This project aims to recognize time from analog clock images using a neuro-symbolic approach.
We separately estimate **clock hands** and **dial orientation**, then integrate them via **"probabilistic logic reasoning (DeepProbLog)"**.

## Objective
- Recognize the time from analog clock images
- Improve reliability by enforcing **geometric consistency constraints**

## Method
### 1) Hand module (supervised)
- Label keypoints: **center**, **minute-hand tip**, **hour-hand tip**
- Train a supervised model to output a **probabilistic distribution** over angles (or discretized time bins)

### 2) Dial module (supervised)
- Detect the **12 o'clock direction** by keypoint detector.
- Estimate clock **dial orientation** to adapt to normalization.

### 3) Logic integration (DeepProbLog)
- Use probabilistic outputs from the hand(angle) & dial(rotation) modules
- Apply geometric rules to convert them into a time prediction
- Train by maximizing the probability (likelihood) of the correct time label via probabilistic inference

## Baselines (Comparison)
We compare the proposed model with:
1. **End-to-end single CNN** (image → time)
2. **Two-stage classical pipeline** (hand/dial → geometry → time, without logic learning)
3. **DeepProbLog neuro-symbolic integration** (proposed)

## Datasets
Two datasets are prepared:
1. **Synthetic analog clock images(clock kaggle)**
2. **Real-world clock images**

For each dataset, we evaluate all three methods above.

---

# Multi-Head Clock Recognition Model (New)

Neuro-symbolic analog clock time recognition using a **shared EfficientNet-B0 backbone** with DeepProbLog integration.

## Architecture

```
EfficientNet-B0 (shared backbone, ImageNet pretrained)
        |
    avgpool -> 1280-dim features
        |
   +---------+---------+
   |         |         |
 Head_R    Head_H    Head_M
 (4-cls)  (12-cls)  (12-cls)
 rotation   hour     minute
```

Three DeepProbLog neural predicates share a single backbone. ProbLog rules enforce rotation correction and hour-minute geometric constraints.

### Key Differences from Existing Model

| Aspect | Existing | Multi-Head Model |
|--------|----------|------------------|
| Network | 3 separate ResNet18 (~33M params) | 1 shared EfficientNet-B0 (~5M params) |
| Data | rotations_new.csv (hand keypoints) | clocks.csv + auto-extracted keypoints |
| Feature sharing | None | Full backbone sharing |
| Dial training | Heatmap on nohands images | Learned from time labels |

### Pipeline

#### 1. Extract Keypoints
```bash
python multihead/extract_keypoints.py --data_root /path/to/clock_kaggle
```

#### 2. Train (3 stages)
```bash
python multihead/train.py --data_root /path/to/clock_kaggle --run all
```

- **Stage 1 - Backbone Warmup**: 144-class time classification (CE loss)
- **Stage 2 - Component Training**: DeepProbLog `component(X, R, H, M)`
- **Stage 3 - Time Integration**: DeepProbLog `time(X, Hour, Minute)` with constraints

#### 3. Evaluate
```bash
python multihead/evaluate.py --data_root /path/to/clock_kaggle --weights weights_multihead/final_shared.pth
```

### File Structure
```
multihead/
    extract_keypoints.py     # Auto keypoint extraction
    networks.py              # MultiHeadClockNet + adapters
    dataset.py               # PyTorch + DeepProbLog datasets
    train.py                 # 3-stage training script
    evaluate.py              # Evaluation script
models/
    clock_multihead.pl       # ProbLog program
```

### Requirements
- Python 3.8+, PyTorch >= 2.0, torchvision >= 0.15
- deepproblog >= 0.1, problog >= 2.2
- OpenCV, pandas, tqdm
