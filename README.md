# TRL Experiments

This repository contains the experimental code for the paper **"Tubular Riemannian Laplace Approximations for Bayesian Neural Networks"** (under review at ICML 2025).

## Overview

Laplace approximations are among the simplest and most practical methods for approximate Bayesian inference in neural networks. However, their Euclidean formulation struggles with the highly anisotropic, curved loss surfaces and large symmetry groups that characterize modern deep models.

**Tubular Riemannian Laplace (TRL)** is a geometric posterior approximation designed for Bayesian neural networks. TRL explicitly models the posterior as a *probabilistic tube* that follows a low-loss valley induced by functional symmetries, using a Fisher/Gauss-Newton metric to separate prior-dominated tangential uncertainty from data-dominated transverse uncertainty.

## Repository Contents

| File | Description |
|------|-------------|
| `toy_experiments.ipynb` | Jupyter notebook with toy experiments demonstrating TRL on simple models. Ready to run on Google Colab. |
| `Resnet_experiments.py` | Full-scale experiments with ResNet-18 on CIFAR-10, comparing TRL against Euclidean and Linearised Laplace baselines. |

## Key Features

- **Geometric analysis** of parameter space for Bayesian neural networks based on Fisher/Gauss-Newton metric
- **TRL approximation** as a reparametrised Gaussian approximation adapted to loss valleys
- **Practical implementation** with computational cost comparable to classical Laplace and LLA
- **Scalable** to deep architectures such as ResNet-18

## Requirements

```bash
pip install laplace-torch "curvlinops-for-pytorch>=2.0,<3.0" matplotlib scikit-learn torch torchvision
```

## Usage

### Toy Experiments (Google Colab)

Open `toy_experiments.ipynb` in Google Colab and run all cells sequentially.

### ResNet-18 Experiments

```bash
python Resnet_experiments.py
```

The script will:
1. Train a ResNet-18 on CIFAR-10 (or load from checkpoint if available)
2. Fit Laplace approximations (ELA, LLA, TRL)
3. Evaluate predictive performance and uncertainty calibration

## Citation

If you use this code, please cite our paper:

```bibtex
@inproceedings{trl2025,
  title={Tubular Riemannian Laplace Approximations for Bayesian Neural Networks},
  author={Anonymous},
  booktitle={International Conference on Machine Learning (ICML)},
  year={2025}
}
```

## License

This project is released for academic and research purposes.
