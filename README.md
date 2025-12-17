# Tubular Riemannian Laplace Approximations for Bayesian Neural Networks (TRL)

This repository contains the official experimental code for the paper **"Tubular Riemannian Laplace Approximations for Bayesian Neural Networks"** (under review at ICML 2025). It provides implementations for the toy experiments and the full-scale ResNet-18 evaluation on CIFAR-10.

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
| `toy_experiments.ipynb` | A Jupyter notebook containing toy experiments on 1D regression and 2D classification. It demonstrates the fundamental advantages of TRL in low-dimensional settings and is ready to run on Google Colab. |
| `Resnet_experiments.py` | The main script for full-scale experiments. It trains a ResNet-18 on CIFAR-10 and compares the performance and calibration of TRL against MAP, last-layer ELA, and last-layer LLA baselines. |

## Experimental Results

TRL was evaluated on a ResNet-18 model with 11 million parameters on the CIFAR-10 dataset. The results show that TRL achieves the lowest Negative Log-Likelihood (NLL) and Expected Calibration Error (ECE), indicating superior predictive uncertainty, without sacrificing accuracy.

| Method | Accuracy ↑ | NLL ↓ | ECE ↓ | Brier ↓ |
|---|---|---|---|---|
| MAP | 94.32% | 0.2110 | 0.0296 | 0.0910 |
| ELA (Last-Layer) | 94.06% | 0.3450 | 0.1676 | 0.1307 |
| LLA (Last-Layer) | 94.26% | 0.1942 | 0.0215 | 0.0866 |
| **TRL (Full-Net)** | **94.19%** | **0.1837** | **0.0063** | **0.0875** |

## Usage

### Toy Experiments (Google Colab)

1.  Open `toy_experiments.ipynb` in Google Colab.
2.  Run all cells sequentially to reproduce the installation, model training, and visualizations.

### ResNet-18 Experiments

To run the full-scale experiment, execute the following command:

```bash
python Resnet_experiments.py
```

The script will automatically handle data preparation, MAP training (or load from the `resnet18_cifar10_map_relu_v2.pth` checkpoint if it exists), and evaluation of all Laplace baselines.

## Requirements

The code relies on the `laplace-torch` library and its dependencies. You can install all required packages using pip:

```bash
pip install torch torchvision laplace-torch "curvlinops-for-pytorch>=2.0,<3.0" matplotlib scikit-learn
```

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
