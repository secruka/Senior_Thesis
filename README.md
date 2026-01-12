# Senior_Thesis
Interpretable Analog Clock Reading with Neuro-Symbolic Constraints (DeepProbLog) 

# DeepProbLog-based Analog Clock Reading

## Overview
This project aims to recognize time from analog clock images using a neuro-symbolic approach.
We separately estimate **clock hands** and **dial orientation**, then integrate them via "probabilistic logic reasoning (DeepProbLog)".

## Objective
- Recognize the time from analog clock images
- Improve reliability by enforcing **geometric consistency constraints**

## Method
### 1) Hand module (supervised)
- Label keypoints: **center**, **minute-hand tip**, **hour-hand tip**
- Train a supervised model to output a **probabilistic distribution** over angles (or discretized time bins)

### 2) Dial module (supervised)
- Detect the **12 o’clock direction**
- Estimate clock **in-plane rotation** (dial orientation)

### 3) Logic integration (DeepProbLog)
- Use probabilistic outputs from the hand & dial modules
- Apply geometric rules to convert them into a time prediction
- Train by maximizing the probability (likelihood) of the correct time label via probabilistic inference

## Baselines (Comparison)
We compare the proposed model with:
1. **End-to-end single CNN** (image → time)
2. **Two-stage classical pipeline** (hand/dial → geometry → time, without logic learning)
3. **DeepProbLog neuro-symbolic integration** (proposed)

## Datasets
Two datasets are prepared:
1. **Synthetic analog clock images**
2. **Real-world clock images** (with plot/annotations)

For each dataset, we evaluate all three methods above.
