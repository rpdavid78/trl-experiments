# Tubular Riemannian Laplace Approximations for Bayesian Neural Networks (TRL)

This repository contains the official experimental code for the paper **"Tubular Riemannian Laplace Approximations for Bayesian Neural Networks"** (under review at ICML 2025). It provides implementations for toy experiments, full-scale ResNet-18 evaluation on CIFAR-10, out-of-distribution (OOD) detection, and comparisons with state-of-the-art uncertainty quantification methods.

## Abstract

Laplace approximations are a practical method for approximate Bayesian inference in neural networks. However, their standard Euclidean formulation struggles with the highly anisotropic, curved loss surfaces and large symmetry groups that characterize modern deep models. This work introduces the **Tubular Riemannian Laplace (TRL)** approximation, a geometric posterior approximation designed for Bayesian neural networks that explicitly models the posterior as a *probabilistic tube* following a low-loss valley. TRL uses a Fisher/Gauss-Newton metric to separate prior-dominated tangential uncertainty from data-dominated transverse uncertainty. We show that TRL is a scalable and effective method that yields well-calibrated uncertainty estimates and preserves high accuracy, outperforming standard baselines.

## The Problem with Standard Laplace Approximations

Modern deep learning models exhibit complex loss landscapes. Key challenges for traditional methods like the Euclidean Laplace Approximation (ELA) include:

- **Anisotropic Curvature**: The loss surface is highly curved, with different directions having vastly different scales of curvature. A single Gaussian ellipsoid (as used in ELA) is a poor fit for these long, narrow valleys of low loss.
- **Functional Symmetries**: Operations like neuron permutation or positive scaling of activations in ReLU networks create large manifolds of parameters that correspond to the same function. ELA fails to account for these symmetries, leading to a mischaracterization of the posterior.

## The TRL Solution

TRL addresses these limitations by constructing a geometry-aware approximation. The core idea is to build a "tube" around a central "spine" that traces a low-loss valley in the parameter space.

1.  **Spine Construction**: A curve γ(t) is defined to follow a path of low loss, representing the central axis of the posterior tube.
2.  **Tangent & Transverse Decomposition**: At each point on the spine, the parameter space is decomposed into directions *tangent* to the valley (where uncertainty is high and dominated by the prior) and directions *transverse* to it (where uncertainty is low and constrained by the data).
3.  **Riemannian Metric**: A Fisher/Gauss-Newton metric is used to define this decomposition, providing a data-aware local coordinate system.

The resulting approximation, `q_TRL`, is a reparametrized Gaussian that captures the tubular geometry, leading to more accurate posterior estimates compared to ELA and Linearised Laplace (LLA).

## Repository Contents

| File | Description |
|------|-------------|
| `toy_experiments.ipynb` | Jupyter notebook with toy experiments on 1D regression and 2D classification. Demonstrates the fundamental advantages of TRL in low-dimensional settings. Ready to run on Google Colab. |
| `Resnet_experiments.py` | Main script for full-scale experiments. Trains a ResNet-18 on CIFAR-10 and compares TRL against MAP, last-layer ELA, and last-layer LLA baselines. |
| `ResNet_Laplace_TRL_OOD.py` | Out-of-distribution (OOD) detection experiments. Evaluates ELA, LLA, and TRL on CIFAR-10 (in-distribution) vs. SVHN (out-of-distribution) using AUROC metric. |
| `SOTA_experiments.py` | Comparison with state-of-the-art uncertainty quantification methods: Deep Ensembles (M=5), SWAG (Diagonal), and MC Dropout. Evaluates all metrics including OOD detection. |

## Experimental Results

### In-Distribution Performance and OOD Detection

TRL was evaluated on a ResNet-18 model with 11 million parameters. The table below shows comprehensive results on CIFAR-10 (in-distribution) and SVHN (out-of-distribution).

| Method | Accuracy ↑ | NLL ↓ | ECE ↓ | Brier ↓ | AUROC ↑ | Cost |
|--------|------------|-------|-------|---------|---------|------|
| **Standard Baselines** | | | | | | |
| MAP | 94.32% | 0.2110 | 0.0296 | 0.0910 | 0.9093 | 1× |
| MC Dropout | 91.97% | 0.2969 | 0.0406 | 0.1254 | 0.8737 | 1× |
| **Strong Baselines** | | | | | | |
| SWAG (Diagonal) | 93.69% | 0.2040 | 0.0226 | 0.0955 | 0.9477 | 1× |
| Deep Ensembles | **95.17%** | **0.1534** | 0.0073 | **0.0730** | **0.9499** | 5× |
| **Laplace Baselines** | | | | | | |
| ELA (Last-Layer) | 94.06% | 0.3450 | 0.1676 | 0.1307 | 0.9042 | 1× |
| LLA (Last-Layer) | 94.26% | 0.1942 | 0.0215 | 0.0866 | 0.9123 | 1× |
| **TRL (Ours)** | 94.19% | 0.1837 | **0.0063** | 0.0875 | 0.9355 | 1× |

### Key Findings

TRL demonstrates a remarkable efficiency-reliability trade-off:

- **Best Calibration**: TRL achieves the lowest ECE of **0.0063**, outperforming even the 5-model Deep Ensemble (0.0073). This indicates that the tubular posterior captures the true probability of correctness more faithfully than heuristic ensembling.

- **Efficiency**: While Deep Ensembles achieve the highest accuracy (95.17%), they incur a 5× computational cost. TRL provides robust OOD detection (0.9355 AUROC) competitive with SWAG (0.9477) and superior calibration within a single model's memory footprint (1× params).

- **Geometric Fidelity**: TRL reduces the calibration error (ECE) by over 3× compared to the strongest Laplace baseline (LLA: 0.0215 → 0.0063), confirming that navigating the full weight space geometrically is superior to local linearization.

## Usage

### Toy Experiments (Google Colab)

1.  Open `toy_experiments.ipynb` in Google Colab.
2.  Run all cells sequentially to reproduce the installation, model training, and visualizations.

### ResNet-18 Experiments

To run the full-scale experiment with in-distribution metrics:

```bash
python Resnet_experiments.py
```

### OOD Detection Experiments

To evaluate out-of-distribution detection with Laplace methods (ELA, LLA, TRL):

```bash
python ResNet_Laplace_TRL_OOD.py
```

This script evaluates on CIFAR-10 (ID) vs. SVHN (OOD) and reports AUROC scores.

### SOTA Baselines Comparison

To run experiments comparing with Deep Ensembles, SWAG, and MC Dropout:

```bash
python SOTA_experiments.py
```

**Note**: Deep Ensembles require training 5 separate models, which increases training time significantly.

## Requirements

The code relies on the `laplace-torch` library and its dependencies. Install all required packages using pip:

```bash
pip install torch torchvision laplace-torch "curvlinops-for-pytorch>=2.0,<3.0" matplotlib scikit-learn
```

## TRL Configuration

The following hyperparameters are used for the TRL experiments:

| Parameter | Value | Description |
|-----------|-------|-------------|
| k_⊥ | 20 | Transverse rank (via Lanczos iteration) |
| T | 20 | Number of spine steps |
| Δs | 0.03 | Step size along the valley |
| β_⊥ | 1.0 | Transverse scale factor |

The method handles Batch Normalization statistics by recalibrating them on a subset of training data for each posterior sample ("FixBN"), a crucial step for valid inference in the weight space of deep networks.

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@inproceedings{trl2025,
  title={Tubular Riemannian Laplace Approximations for Bayesian Neural Networks},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```
