# Tubular Riemannian Laplace Approximations for Bayesian Neural Networks

This repository contains the official experimental code for the paper **"Tubular Riemannian Laplace Approximations for Bayesian Neural Networks"** ([arXiv:2412.22087](https://arxiv.org/abs/2412.22087)).

## Abstract

Laplace approximations are a practical method for approximate Bayesian inference in neural networks. However, their standard Euclidean formulation struggles with the highly anisotropic, curved loss surfaces and large symmetry groups that characterize modern deep models. This work introduces the **Tubular Riemannian Laplace (TRL)** approximation, a geometric posterior approximation designed for Bayesian neural networks that explicitly models the posterior as a *probabilistic tube* following a low-loss valley. TRL uses a Fisher/Gauss-Newton metric to separate prior-dominated tangential uncertainty from data-dominated transverse uncertainty.

## The Problem with Standard Laplace Approximations

Modern deep learning models exhibit complex loss landscapes with two key challenges for traditional methods like the Euclidean Laplace Approximation (ELA):

**Anisotropic Curvature.** The loss surface is highly curved, with different directions having vastly different scales of curvature. A single Gaussian ellipsoid (as used in ELA) is a poor fit for these long, narrow valleys of low loss.

**Functional Symmetries.** Operations like neuron permutation or positive scaling of activations in ReLU networks create large manifolds of parameters that correspond to the same function. ELA fails to account for these symmetries, leading to a mischaracterization of the posterior.

## The TRL Solution

TRL addresses these limitations by constructing a geometry-aware approximation. The core idea is to build a "tube" around a central "spine" that traces a low-loss valley in the parameter space.

1. **Spine Construction**: A curve γ(t) is defined to follow a path of low loss, representing the central axis of the posterior tube.
2. **Tangent & Transverse Decomposition**: At each point on the spine, the parameter space is decomposed into directions *tangent* to the valley (where uncertainty is high and dominated by the prior) and directions *transverse* to it (where uncertainty is low and constrained by the data).
3. **Riemannian Metric**: A Fisher/Gauss-Newton metric is used to define this decomposition, providing a data-aware local coordinate system.

## Repository Structure

| File | Description |
|------|-------------|
| `toy_experiments.ipynb` | Jupyter notebook with toy experiments on 1D regression and 2D classification. Demonstrates the fundamental advantages of TRL in low-dimensional settings. Ready to run on Google Colab. |
| `cifar10_trl_laplace.py` | CIFAR-10 experiments comparing TRL against MAP, ELA, and LLA baselines on ResNet-18. |
| `cifar10_ood_detection.py` | Out-of-distribution detection experiments on CIFAR-10 (ID) vs SVHN (OOD) for Laplace methods. |
| `cifar10_sota_baselines.py` | CIFAR-10 comparison with SOTA methods: Deep Ensembles (M=5), SWAG, and MC Dropout. |
| `cifar100_all_methods.py` | Comprehensive CIFAR-100 benchmark including all methods: MAP, ELA, LLA, TRL, Deep Ensembles, SWAG, and MC Dropout. |

## Experimental Results

### CIFAR-100 (ResNet-18)

TRL achieves ensemble-level calibration (ECE) using a single model (1× params), significantly outperforming SWAG and MC Dropout in NLL and calibration metrics. The method bridges the gap to the Deep Ensemble gold standard without the cost of training multiple models.

| Method | Acc ↑ | NLL ↓ | ECE ↓ | Brier ↓ | AUROC ↑ | Cost |
|--------|-------|-------|-------|---------|---------|------|
| **Gold Standard (High Cost)** | | | | | | |
| Deep Ensembles | **78.36%** | **0.7919** | 0.0161 | **0.3032** | **0.8843** | 5× |
| **Training-based Baselines** | | | | | | |
| SWAG | 75.29% | 1.0055 | 0.0917 | 0.3572 | 0.8033 | 1× |
| MC Dropout | 74.83% | 1.0124 | 0.0894 | 0.3630 | 0.8226 | 1× |
| **Post-hoc Baselines** | | | | | | |
| MAP | 74.22% | 1.0358 | 0.0946 | 0.3689 | 0.8028 | 1× |
| ELA (Last-Layer) | 72.83% | 1.3433 | 0.2626 | 0.4654 | 0.8451 | 1× |
| LLA (Last-Layer) | 73.91% | 1.4787 | 0.3636 | 0.5164 | 0.8740 | 1× |
| **TRL (Ours)** | 74.51% | 0.9525 | **0.0171** | 0.3533 | 0.8255 | 1× |

### CIFAR-10 (ResNet-18)

| Method | Acc ↑ | NLL ↓ | ECE ↓ | Brier ↓ | AUROC ↑ | Cost |
|--------|-------|-------|-------|---------|---------|------|
| MAP | 94.32% | 0.2110 | 0.0296 | 0.0910 | 0.9093 | 1× |
| MC Dropout | 91.97% | 0.2969 | 0.0406 | 0.1254 | 0.8737 | 1× |
| SWAG | 93.69% | 0.2040 | 0.0226 | 0.0955 | 0.9477 | 1× |
| Deep Ensembles | **95.17%** | **0.1534** | 0.0073 | **0.0730** | **0.9499** | 5× |
| ELA (Last-Layer) | 94.06% | 0.3450 | 0.1676 | 0.1307 | 0.9042 | 1× |
| LLA (Last-Layer) | 94.26% | 0.1942 | 0.0215 | 0.0866 | 0.9123 | 1× |
| **TRL (Ours)** | 94.19% | 0.1837 | **0.0063** | 0.0875 | 0.9355 | 1× |

### Key Findings

**Ensemble-Grade Calibration at Single-Model Cost.** TRL achieves an ECE of **0.0171** on CIFAR-100, virtually indistinguishable from the 5× costlier Deep Ensemble (0.0161) and 5× better than SWAG (0.0917). This demonstrates that exploring a single connected basin with high geometric precision yields equivalent calibration benefits to training multiple independent models.

**Geometric Failure of Static Approximations.** In the high-dimensional regime of CIFAR-100 (|θ_last| ≈ 51,200), both ELA and LLA severely degrade the NLL compared to MAP (1.03 → 1.47 for LLA), yielding unacceptable calibration errors (ECE > 0.26). TRL solves this by physically moving the distribution mean along the invariance manifold.

**Superior Reliability.** TRL corrects the underfitting pathologies of static Laplace approximations and surpasses training-based methods like SWAG by explicitly modeling the non-linear manifold structure.

## Usage

### Toy Experiments (Google Colab)

Open `toy_experiments.ipynb` in Google Colab and run all cells sequentially.

### CIFAR-10 Experiments

```bash
# Main TRL vs Laplace comparison
python cifar10_trl_laplace.py

# OOD detection (CIFAR-10 vs SVHN)
python cifar10_ood_detection.py

# SOTA baselines (Deep Ensembles, SWAG, MC Dropout)
python cifar10_sota_baselines.py
```

### CIFAR-100 Experiments

```bash
# All methods benchmark
python cifar100_all_methods.py
```

**Note:** Deep Ensembles require training 5 separate models, which increases training time significantly.

## Requirements

```bash
pip install torch torchvision laplace-torch "curvlinops-for-pytorch>=2.0,<3.0" matplotlib scikit-learn scipy
```

## TRL Configuration

The following hyperparameters are used for the experiments:

| Parameter | CIFAR-10 | CIFAR-100 | Description |
|-----------|----------|-----------|-------------|
| k_⊥ | 20 | 30 | Transverse rank (via Lanczos iteration) |
| T | 20 | 40 | Number of spine steps |
| Δs | 0.03 | 0.01 | Step size along the valley |
| β_⊥ | 1.0 | 4.0 | Transverse scale factor |
| S | 25 | 25 | Monte Carlo samples |

Batch Normalization statistics are handled by recalibrating them on a clean subset of training data for each posterior sample ("FixBN"), a crucial step for valid inference in the weight space of deep networks.



## License

## Citation

If you find this code useful for your research, please consider citing our paper:

```bibtex
@article{david2024tubular,
  title={Tubular Riemannian Laplace Approximations for Bayesian Neural Networks},
  author={David, Rodrigo Pereira},
  journal={arXiv preprint arXiv:2412.22087},
  year={2024}
}

This project is released for academic and research purposes.
