# HSRest dataset paper inpainting model implementation of HLRTF as a baseline
# HLRTF (Pytorch)


## Kindly refer to the original git repo
https://github.com/YisiLuo/HLRTF

# This readme is a modified version of the original code to make it work with our dataset.




# Hyperspectral Image Restoration Benchmark Pipeline

A unified benchmarking and evaluation pipeline for hyperspectral image restoration tasks including denoising, inpainting, and super-resolution.

This repository provides reproducible evaluation scripts, visualization utilities, spectral analysis tools, and baseline implementations for comparative hyperspectral image restoration experiments.

---

## Overview

This project focuses on evaluating hyperspectral image restoration methods under multiple degradation settings using consistent datasets, corruption models, and evaluation metrics.

Supported tasks include:

- Hyperspectral Image Denoising
- Hyperspectral Image Inpainting
- Hyperspectral Image Super-Resolution

The repository includes:

- Spectral reconstruction analysis
- Error visualization
- RGB/FCC rendering
- Quantitative benchmarking
- Reconstruction mosaics
- Comparative qualitative analysis

---

## Repository Structure

```text
project_root/
│
├── data/                  # Input hyperspectral datasets
├── results/               # Reconstruction outputs
├── scripts/               # Evaluation and visualization scripts
├── plots/                 # Spectral plots and mosaics
├── checkpoints/           # Saved model outputs
└── README.md
```

---

## Dataset Format

Each `.mat` file should contain:

```python
GT      # Ground truth hyperspectral image
Nhsi    # Corrupted hyperspectral image
mask    # Binary observation mask
```


The mat file metadata is
=== KEYS ===
dict_keys(['__header__', '__version__', '__globals__', 'Nhsi', 'GT', 'mask'])

=== NHSI ===
Shape: (128, 128, 175)
Dtype: float32
Min: 0.0
Max: 1.0

=== GT ===
Shape: (128, 128, 175)
Dtype: float32
Min: 0.0
Max: 1.0

=== MASK ===
Shape: (128, 128, 175)
Dtype: float32
Min: 0.0
Max: 1.0
Unique values: [0. 1.]

Our test dataset is here 
data\EO1H0420342007273110PF_stacked_3.mat

### Shape Requirements

```python
(Height, Width, Bands)
```

Example:

```python
(128, 128, 175)
```

All hyperspectral cubes must be normalized to:

```python
[0, 1]
```

---

## Environment Setup

### Create Environment

```bash
conda create -n hsi_env python=3.8 -y
conda activate hsi_env
```

### Install Dependencies

```bash
pip install numpy scipy matplotlib scikit-image torch torchvision
```

---

## Running Evaluation

### Example Command

```bash
python HLRTF_tensor_completion_demo.py
```

Outputs include:

- Reconstructed hyperspectral images
- Spectral signature plots
- Error maps
- PSNR / SSIM metrics
- Visualization mosaics

---

## Visualization Utilities

The repository includes utilities for:

- Spectral signature comparison
- Error visualization
- Reconstruction mosaics
- False-color composite (FCC) rendering
- Zoomed qualitative comparisons

Typical spectral plots compare:

- Ground Truth
- Corrupted Input
- Reconstructed Output

across multiple semantic regions such as:

- Water
- Cropland
- Forest
- Wetland
- Built-up regions

---

## Baseline Models

This repository evaluates multiple restoration approaches under a unified experimental setting.

### HLRTF Baseline

> **Important Note**
>
> HLRTF is **not** a model proposed by the author of this repository.
>
> The HLRTF implementation was executed strictly as a **baseline comparison method** for benchmarking purposes against other restoration approaches.
>
> Full credit for the original HLRTF method belongs to the respective original authors.

---

## Evaluation Metrics

The following metrics are used for quantitative evaluation:

- PSNR
- SSIM
- Spectral reconstruction consistency
- Visual qualitative assessment

---

## Reproducibility

For fair comparison across methods:

- Use identical corruption masks
- Use identical train/test splits
- Normalize all data consistently
- Fix random seeds when applicable

---

## Exact code implementation
python HLRTF_tensor_completion_demo.py

## Acknowledgements

This repository includes comparative evaluations involving previously published restoration methods from prior literature.

Special acknowledgement to the original authors of HLRTF for their foundational contribution to hyperspectral image restoration research.

---

## Citation

If you use this repository in your research, please cite the corresponding paper/project associated with this implementation.

---

## Contact

For questions, suggestions, or issues, please open an issue in this repository.
