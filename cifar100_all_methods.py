# cifar100_all_baselines_trl_laplace_de_swag_mcdropout.py
# A single script for CIFAR-100:
# - MAP
# - Laplace (Last-layer, KRON) + marglik => ELA/LLA
# - Deep Ensemble (DE)
# - SWAG
# - MC Dropout
# - TRL Stage-2 (HVP ablation) with tube_scale sweep
#
# Maintains the essential logic of your scripts:
# - Fixed split 45k/5k with SEED
# - Train with AUG; CLEAN loaders available
# - TRL: HVP with BN in TRAIN; FixBN with AUG loader
# - Laplace: last-layer + kron + optimize_prior_precision(method="marglik")
#
# Requisitos:
# pip install laplace-torch scipy scikit-learn torchvision

import os
import copy
import gc
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import scipy.sparse.linalg as sla
from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from laplace import Laplace


# ==============================================================================
# CONFIG
# ==============================================================================
@dataclass
class CFG:
    # system
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # data
    batch_size: int = 128
    num_workers: int = 2
    num_classes: int = 100
    ood_svhn_n: int = 2000

    # MAP training (base recipe)
    epochs_map: int = 50
    lr_map: float = 0.1
    wd_map: float = 5e-4
    momentum: float = 0.9

    # checkpoints
    ckpt_dir: str = "./checkpoints_c100"
    map_ckpt: str = "resnet18_cifar100_map.pth"
    mcdo_ckpt: str = "resnet18_cifar100_mcdo.pth"
    ens_prefix: str = "c100_ens"
    swag_stats: str = "c100_swag_stats.pth"
    trl_spine: str = "c100_trl_stage2_spine.pth"

    # Laplace
    laplace_subset: int = 5000
    laplace_fit_bs: int = 32
    laplace_n_samples_ela: int = 25  # ELA
    # LLA does not use n_samples in its code

    # Deep Ensemble
    ens_M: int = 5

    # SWAG
    swag_epochs: int = 10
    swag_lr: float = 1e-3
    swag_samples: int = 20
    swag_sample_scale: float = 1.0
    swag_collect_momentum: float = 0.9
    swag_collect_alpha: float = 0.1
    swag_fixbn_batches: int = 20

    # MC Dropout
    mcdo_p: float = 0.2
    mcdo_samples: int = 25
    mcdo_fixbn_batches: int = 20

    # TRL Stage-2 fixed parameters
    trl_k_perp: int = 30
    trl_steps: int = 40
    trl_step_size: float = 0.01
    trl_eta: float = 1e-3
    trl_val_samples: int = 25
    trl_fixbn_batches: int = 25
    trl_tube_scales: Tuple[float, ...] = (2.0, 3.0, 4.0, 6.0, 10.0, 20.0)

    # TRL numerics / safety
    trl_max_delta_norm: float = 0.02
    trl_hvp_batches: int = 5

    # TRL spine storage (memory)
    trl_store_every: int = 1  # 1 = saves every step (like your script). e.g.: 2 = saves every 2 steps.


CFG_ = CFG()
DEVICE = torch.device(CFG_.device)


# ==============================================================================
# UTILS
# ==============================================================================
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def flatten_grads(grads, params):
    vec = []
    for g, p in zip(grads, params):
        if g is not None:
            vec.append(g.contiguous().view(-1))
        else:
            vec.append(torch.zeros_like(p).view(-1))
    return torch.cat(vec)


def get_targets(loader) -> torch.Tensor:
    ys = []
    for _, y in loader:
        ys.append(y)
    return torch.cat(ys)


# ==============================================================================
# DATA
# ==============================================================================
def get_data(cfg: CFG):
    print(">>> Preparando Dados CIFAR-100...")

    mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    t_train_aug = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])
    t_clean = transforms.Compose([
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])

    trainset_aug = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=t_train_aug)
    trainset_clean = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=t_clean)

    # fixed split with seed
    indices = torch.randperm(len(trainset_aug), generator=torch.Generator().manual_seed(cfg.seed)).tolist()
    train_idx = indices[:45000]
    val_idx = indices[45000:]

    tr_sub_aug = torch.utils.data.Subset(trainset_aug, train_idx)
    tr_sub_clean = torch.utils.data.Subset(trainset_clean, train_idx)
    val_sub_clean = torch.utils.data.Subset(trainset_clean, val_idx)

    tr_loader_aug = torch.utils.data.DataLoader(
        tr_sub_aug, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    bn_loader_clean = torch.utils.data.DataLoader(
        tr_sub_clean, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        val_sub_clean, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    testset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=t_clean)
    ts_loader = torch.utils.data.DataLoader(
        testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
    )

    # OOD: SVHN
    svhn = torchvision.datasets.SVHN(root="./data", split="test", download=True, transform=t_clean)
    ood_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(svhn, range(cfg.ood_svhn_n)),
        batch_size=cfg.batch_size,
        shuffle=False
    )

    return tr_loader_aug, bn_loader_clean, val_loader, ts_loader, ood_loader


# ==============================================================================
# MODEL (ResNet-18 for CIFAR - "CIFAR-style" version as its TRL/Laplace)
# ==============================================================================
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_dropout=False, p_drop=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.use_dropout = use_dropout
        self.drop = nn.Dropout(p_drop) if use_dropout and p_drop > 0 else nn.Identity()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=100, use_dropout=False, p_drop=0.0):
        super().__init__()
        self.in_planes = 64
        self.use_dropout = use_dropout
        self.p_drop = p_drop

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop_head = nn.Dropout(p_drop) if use_dropout and p_drop > 0 else nn.Identity()
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(BasicBlock(self.in_planes, planes, st, use_dropout=self.use_dropout, p_drop=self.p_drop))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = self.avgpool(out).flatten(1)
        out = self.drop_head(out)
        return self.linear(out)


# ==============================================================================
# TRAIN / LOAD (MAP e afins)
# ==============================================================================
def train_model(model: nn.Module, train_loader, cfg: CFG, epochs: int, lr: float, wd: float, ckpt_path: Optional[str] = None):
    model = model.to(DEVICE)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg.momentum, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Ep {epoch+1}/{epochs} done.")
        cleanup()

    if ckpt_path is not None:
        torch.save(model.state_dict(), ckpt_path)

    return model


def load_or_train_map(tr_loader_aug, cfg: CFG) -> nn.Module:
    ensure_dir(cfg.ckpt_dir)
    path = os.path.join(cfg.ckpt_dir, cfg.map_ckpt)
    model = ResNetCIFAR(cfg.num_classes, use_dropout=False).to(DEVICE)

    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(">>> MAP Carregado.")
        return model

    print(f">>> Treinando MAP ({cfg.epochs_map} epochs)...")
    model = train_model(model, tr_loader_aug, cfg, epochs=cfg.epochs_map, lr=cfg.lr_map, wd=cfg.wd_map, ckpt_path=path)
    return model


# ==============================================================================
# PRED / METRICS
# ==============================================================================
@torch.no_grad()
def predict_probs(model: nn.Module, loader) -> torch.Tensor:
    model.eval()
    probs = []
    for x, _ in loader:
        p = torch.softmax(model(x.to(DEVICE)), dim=1).cpu()
        probs.append(p)
    return torch.cat(probs, dim=0)


def calc_metrics(probs: torch.Tensor, targets: torch.Tensor, num_classes: int):
    p = probs.clamp(1e-7, 1 - 1e-7)
    nll = nn.NLLLoss()(torch.log(p), targets.long()).item()
    acc = p.argmax(1).eq(targets).float().mean().item()

    confs, preds = p.max(1)
    ece = 0.0
    bins = torch.linspace(0, 1, 16)
    for i in range(15):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() > 0:
            ece += torch.abs(
                confs[mask].mean() - preds[mask].eq(targets[mask]).float().mean()
            ) * (mask.sum() / len(p))

    oh = F.one_hot(targets.long(), num_classes).float()
    brier = ((p - oh) ** 2).sum(1).mean().item()
    return acc, nll, float(ece), brier


def auroc_entropy(p_id: torch.Tensor, p_ood: torch.Tensor) -> float:
    p_id = p_id.clamp(1e-9, 1.0)
    p_ood = p_ood.clamp(1e-9, 1.0)
    ent_id = -(p_id * torch.log(p_id)).sum(1).numpy()
    ent_ood = -(p_ood * torch.log(p_ood)).sum(1).numpy()
    y = np.concatenate([np.zeros(len(ent_id)), np.ones(len(ent_ood))])
    s = np.concatenate([ent_id, ent_ood])
    return float(roc_auc_score(y, s))


# ==============================================================================
# Laplace baseline (ELA/LLA)
# ==============================================================================
def laplace_fit_and_predict(model_map: nn.Module, bn_loader_clean, ts_loader, ood_loader, cfg: CFG):
    print("\n>>> [Laplace] Ajustando Laplace (Last-Layer, KRON) + marglik...")
    # subset for fit
    subset_idx = torch.randperm(len(bn_loader_clean.dataset), generator=torch.Generator().manual_seed(cfg.seed))[:cfg.laplace_subset]
    sub_tr = torch.utils.data.Subset(bn_loader_clean.dataset, subset_idx)

    la = Laplace(model_map, likelihood="classification", subset_of_weights="last_layer", hessian_structure="kron")
    la.fit(torch.utils.data.DataLoader(sub_tr, batch_size=cfg.laplace_fit_bs, shuffle=True))
    la.optimize_prior_precision(method="marglik")

    # prior base (LL)
    if la.prior_precision.ndim > 1:
        base_val = la.prior_precision.diag().mean().item()
    else:
        base_val = la.prior_precision.mean().item()
    print(f"    Prior Base (LL): {base_val:.4f}")

    # ELA / LLA prediction (maintains its behavior)
    def laplace_pred(la_obj, loader, pred_type):
        out = []
        link = "mc" if pred_type == "nn" else "probit"
        for x, _ in loader:
            if pred_type == "nn":
                out.append(
                    la_obj(x.to(DEVICE), pred_type=pred_type, link_approx=link, n_samples=cfg.laplace_n_samples_ela)
                    .detach().cpu()
                )
            else:
                out.append(
                    la_obj(x.to(DEVICE), pred_type=pred_type, link_approx=link)
                    .detach().cpu()
                )
        return torch.cat(out)

    print(">>> [Laplace] Pred ELA/LLA...")
    p_ela = laplace_pred(la, ts_loader, "nn")
    p_lla = laplace_pred(la, ts_loader, "glm")
    p_ela_ood = laplace_pred(la, ood_loader, "nn")
    p_lla_ood = laplace_pred(la, ood_loader, "glm")

    return la, base_val, p_ela, p_lla, p_ela_ood, p_lla_ood


# ==============================================================================
# Deep Ensemble
# ==============================================================================
def deep_ensemble(tr_loader_aug, ts_loader, ood_loader, cfg: CFG):
    print(f"\n>>> [Deep Ensemble] Training/Loading M={cfg.ens_M}...")
    ensure_dir(cfg.ckpt_dir)

    preds_id = []
    preds_ood = []
    last_model = None

    targets_ts = get_targets(ts_loader)

    for i in range(cfg.ens_M):
        m = ResNetCIFAR(cfg.num_classes, use_dropout=False).to(DEVICE)
        path = os.path.join(cfg.ckpt_dir, f"{cfg.ens_prefix}_{i}.pth")

        if os.path.exists(path):
            print(f"  Loading '{os.path.basename(path)}'...")
            m.load_state_dict(torch.load(path, map_location=DEVICE))
        else:
            print(f"  Training '{os.path.basename(path)}'...")
            m = train_model(m, tr_loader_aug, cfg, epochs=cfg.epochs_map, lr=cfg.lr_map, wd=cfg.wd_map, ckpt_path=path)

        p_id = predict_probs(m, ts_loader)
        p_ood = predict_probs(m, ood_loader)

        preds_id.append(p_id)
        preds_ood.append(p_ood)
        last_model = copy.deepcopy(m)

        cleanup()

    p_ens = torch.stack(preds_id).mean(0)
    p_ens_ood = torch.stack(preds_ood).mean(0)

    # also returns the last member (as in your script) to start SWAG
    return p_ens, p_ens_ood, last_model, targets_ts


# ==============================================================================
# SWAG (simplified, maintaining its behavior)
# ==============================================================================
class SWAG:
    def __init__(self, base_model: nn.Module, cfg: CFG):
        self.base_model = copy.deepcopy(base_model).to(DEVICE)
        self.n = 0
        self.mean = [torch.zeros_like(p, device=DEVICE) for p in self.base_model.parameters()]
        self.sq_mean = [torch.zeros_like(p, device=DEVICE) for p in self.base_model.parameters()]
        self.cfg = cfg

    @torch.no_grad()
    def collect(self, model: nn.Module):
        self.n += 1
        mom = self.cfg.swag_collect_momentum
        a = self.cfg.swag_collect_alpha
        for i, p in enumerate(model.parameters()):
            p_data = p.data.to(DEVICE)
            self.mean[i] = self.mean[i] * mom + p_data * a
            self.sq_mean[i] = self.sq_mean[i] * mom + (p_data ** 2) * a

    @torch.no_grad()
    def sample(self, scale: float):
        for i, p in enumerate(self.base_model.parameters()):
            mu = self.mean[i]
            var = torch.clamp(self.sq_mean[i] - mu ** 2, min=1e-30)
            z = torch.randn_like(p, device=DEVICE)
            p.data = mu + math.sqrt(scale) * torch.sqrt(var) * z


def fix_bn(model: nn.Module, loader, device: torch.device, num_batches: int):
    model.train()
    with torch.no_grad():
        it = iter(loader)
        for _ in range(num_batches):
            try:
                x, _ = next(it)
            except StopIteration:
                break
            model(x.to(device))
    model.eval()


def run_swag(tr_loader_aug, ts_loader, ood_loader, last_model: nn.Module, cfg: CFG):
    print("\n>>> [SWAG] Training SWAG (fine-tune + collect)...")
    ensure_dir(cfg.ckpt_dir)
    stats_path = os.path.join(cfg.ckpt_dir, cfg.swag_stats)

    # se stats existem, carrega
    if os.path.exists(stats_path):
        print("  Loading SWAG stats...")
        payload = torch.load(stats_path, map_location=DEVICE)
        swag = SWAG(last_model, cfg)
        swag.n = payload["n"]
        swag.mean = [t.to(DEVICE) for t in payload["mean"]]
        swag.sq_mean = [t.to(DEVICE) for t in payload["sq_mean"]]
    else:
        swag = SWAG(last_model, cfg)
        m = copy.deepcopy(last_model).to(DEVICE)
        opt = torch.optim.SGD(m.parameters(), lr=cfg.swag_lr, momentum=cfg.momentum)

        for ep in range(cfg.swag_epochs):
            m.train()
            for x, y in tr_loader_aug:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad(set_to_none=True)
                nn.CrossEntropyLoss()(m(x), y).backward()
                opt.step()
            swag.collect(m)
            cleanup()

        torch.save({
            "n": swag.n,
            "mean": [t.detach().cpu() for t in swag.mean],
            "sq_mean": [t.detach().cpu() for t in swag.sq_mean],
        }, stats_path)

    print(">>> [SWAG] Sampling...")
    preds_id = []
    preds_ood = []
    for _ in range(cfg.swag_samples):
        swag.sample(scale=cfg.swag_sample_scale)
        fix_bn(swag.base_model, tr_loader_aug, DEVICE, num_batches=cfg.swag_fixbn_batches)
        preds_id.append(predict_probs(swag.base_model, ts_loader))
        preds_ood.append(predict_probs(swag.base_model, ood_loader))
        cleanup()

    p_swag = torch.stack(preds_id).mean(0)
    p_swag_ood = torch.stack(preds_ood).mean(0)
    return p_swag, p_swag_ood


# ==============================================================================
# MC Dropout
# ==============================================================================
def enable_dropout_only(model: nn.Module):
    # Keeps BN in eval, but activates Dropout
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model: nn.Module, loader, tr_loader_for_bn, cfg: CFG):
    # calibra BN once (optional; here follows the "fix_bn" style of your baselines)
    fix_bn(model, tr_loader_for_bn, DEVICE, num_batches=cfg.mcdo_fixbn_batches)

    preds = []
    for _ in range(cfg.mcdo_samples):
        enable_dropout_only(model)
        probs = []
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(DEVICE)), dim=1).cpu())
        preds.append(torch.cat(probs))
        cleanup()
    return torch.stack(preds).mean(0)


def load_or_train_mcdo(tr_loader_aug, cfg: CFG):
    ensure_dir(cfg.ckpt_dir)
    path = os.path.join(cfg.ckpt_dir, cfg.mcdo_ckpt)

    model = ResNetCIFAR(cfg.num_classes, use_dropout=True, p_drop=cfg.mcdo_p).to(DEVICE)
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=DEVICE))
        print(">>> MC Dropout model carregado.")
        return model

    print(f">>> Treinando MC Dropout model (dropout={cfg.mcdo_p}, {cfg.epochs_map} epochs)...")
    model = train_model(model, tr_loader_aug, cfg, epochs=cfg.epochs_map, lr=cfg.lr_map, wd=cfg.wd_map, ckpt_path=path)
    return model


# ==============================================================================
# TRL Stage-2 (HVP ablation)
# ==============================================================================
def get_hvp_function_ablation(model: nn.Module, loader, device: torch.device, num_batches: int):
    # ABLATION: BN in training during HVP
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    num_params = sum(p.numel() for p in params)

    data_cache = []
    it = iter(loader)
    for _ in range(num_batches):
        try:
            data_cache.append(next(it))
        except StopIteration:
            break
    if len(data_cache) == 0:
        raise RuntimeError("HVP: data_cache vazio.")

    def hvp(v_numpy):
        v = torch.from_numpy(v_numpy).float().to(device)
        model.zero_grad(set_to_none=True)

        loss_avg = torch.tensor(0.0, device=device)
        for x, y in data_cache:
            x, y = x.to(device), y.to(device)
            loss_avg = loss_avg + nn.CrossEntropyLoss()(model(x), y)
        loss_avg = loss_avg / len(data_cache)

        grads = torch.autograd.grad(loss_avg, params, create_graph=True, allow_unused=True)
        g_vec = flatten_grads(grads, params)

        prod = torch.dot(g_vec, v)
        hv_grads = torch.autograd.grad(prod, params, retain_graph=False, allow_unused=True)
        hv_vec = flatten_grads(hv_grads, params)

        del loss_avg, grads, g_vec, prod, hv_grads
        return hv_vec.detach().cpu().numpy()

    _ = num_params
    return hvp


class PracticalTRLStage2:
    def __init__(
        self,
        map_model: nn.Module,
        prior_vec: torch.Tensor,
        clean_loader,
        steps: int,
        k_perp: int,
        step_size: float,
        eta: float,
        tube_scale: float,
        max_delta_norm: float,
        hvp_batches: int,
        store_every: int = 1,
    ):
        self.map_state = copy.deepcopy(map_model.state_dict())
        self.model = copy.deepcopy(map_model).to(DEVICE)
        self.prior = prior_vec.detach().to(DEVICE)
        self.loader = clean_loader

        self.T = steps
        self.k = k_perp
        self.ds = step_size
        self.eta = eta
        self.beta = tube_scale
        self.max_delta_norm = max_delta_norm
        self.hvp_batches = hvp_batches
        self.store_every = max(1, int(store_every))

        self.spine: List[Dict[str, torch.Tensor]] = []

    def _reset(self):
        self.model.load_state_dict(self.map_state)
        self.model.eval()

    def build(self):
        self._reset()

        params = [p for p in self.model.parameters() if p.requires_grad]
        curr_theta = parameters_to_vector(params).detach()
        num_params = curr_theta.numel()

        print(f"    [TRL] Build Start: P={num_params}, T={self.T}, K={self.k}")
        print("    [TRL] Mode: EFFICIENT (Geometric Transport) | HVP(train-BN)")

        hvp_fn = get_hvp_function_ablation(self.model, self.loader, DEVICE, num_batches=self.hvp_batches)
        op = sla.LinearOperator((num_params, num_params), matvec=hvp_fn)
        vals, vecs = sla.eigsh(op, k=self.k + 1, which="LA")

        idx = np.argsort(vals)[::-1][:self.k]
        N = torch.from_numpy(vecs[:, idx].copy()).float().to(DEVICE)
        evals = torch.maximum(
            torch.from_numpy(vals[idx].copy()).float().to(DEVICE),
            torch.tensor(0.0, device=DEVICE),
        )

        # tangente inicial
        vr = torch.randn(num_params, device=DEVICE)
        vr = vr - N @ (N.T @ vr)
        v = vr / (vr.norm() + 1e-9)

        data_iterator = iter(self.loader)

        for t in range(self.T):
            # geometria transversal + prior
            prior_proj = torch.sum((N ** 2) * self.prior.unsqueeze(1), dim=0)
            prec = torch.clamp(evals + prior_proj, min=1e-6)
            inv_sqrt_prec = torch.rsqrt(prec)

            # spine guard (with stride)
            if (t % self.store_every) == 0:
                self.spine.append({
                    "theta": curr_theta.detach().cpu(),
                    "N": N.detach().cpu(),
                    "inv_sqrt_prec": inv_sqrt_prec.detach().cpu(),
                })

            # step in the trajectory
            self.model.eval()
            self.model.zero_grad(set_to_none=True)
            vector_to_parameters(curr_theta, self.model.parameters())

            try:
                xb, yb = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.loader)
                xb, yb = next(data_iterator)

            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            logits = self.model(xb)
            if not torch.isfinite(logits).all():
                raise RuntimeError("TRL build: logits NaN/Inf")

            loss = nn.CrossEntropyLoss()(logits, yb)
            grads = torch.autograd.grad(loss, params, allow_unused=True)
            g = flatten_grads(grads, params)

            g_perp = g - torch.dot(g, v) * v
            delta = self.ds * v - self.eta * g_perp

            dnorm = delta.norm()
            if dnorm > self.max_delta_norm:
                delta = delta * (self.max_delta_norm / (dnorm + 1e-12))

            theta_next = curr_theta + delta

            # transporte
            d = theta_next - curr_theta
            d_norm = d.norm()
            if d_norm > 1e-9:
                v_new = d / d_norm
                proj = v_new @ N
                N_ortho = N - torch.outer(v_new, proj)
                N, _ = torch.linalg.qr(N_ortho, mode="reduced")
                v = v_new

            curr_theta = theta_next

            del loss, grads, g, g_perp, delta
            if (t % 5) == 0:
                cleanup()

    def predict(self, loader, bn_loader_aug, n_samples: int, fix_bn_batches: int):
        if not self.spine:
            raise RuntimeError("Spine vazia. Rode build() antes.")

        ens_probs = []
        targets_all = []

        for i in range(n_samples):
            pt = self.spine[np.random.randint(len(self.spine))]
            th_loc = pt["theta"].to(DEVICE)
            N_loc = pt["N"].to(DEVICE)
            isp_loc = pt["inv_sqrt_prec"].to(DEVICE)

            z = torch.randn(self.k, device=DEVICE)
            theta_sample = th_loc + N_loc @ (self.beta * (isp_loc * z))

            vector_to_parameters(theta_sample, self.model.parameters())

            # ABLATION: FixBN with AUG loader
            fix_bn(self.model, bn_loader_aug, DEVICE, num_batches=fix_bn_batches)

            preds_batch = []
            with torch.no_grad():
                for x, y in loader:
                    preds_batch.append(torch.softmax(self.model(x.to(DEVICE)), 1).cpu())
                    if i == 0:
                        targets_all.append(y)

            ens_probs.append(torch.cat(preds_batch))
            print(".", end="", flush=True)

            del theta_sample, th_loc, N_loc, isp_loc
            cleanup()

        print()
        return torch.stack(ens_probs).mean(0), (torch.cat(targets_all) if targets_all else None)


def build_trl_prior_from_laplace(base_val: float, model: nn.Module) -> torch.Tensor:
    # keeps its logic: conv prior = max(base*50, 5), head prior = base
    boost_val = max(base_val * 50.0, 5.0)

    prior_list = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # our head is "linear."
        val_to_use = base_val if ("linear." in name or "fc." in name) else boost_val
        prior_list.append(torch.full((param.numel(),), float(val_to_use), device=DEVICE))
    return torch.cat(prior_list)


def trl_stage2_run(model_map, base_val, bn_loader_clean, tr_loader_aug, val_loader, ts_loader, ood_loader, cfg: CFG):
    print("\n>>> [TRL Stage-2] Construindo prior e rodando build + sweep tube_scale...")

    prior_vec = build_trl_prior_from_laplace(base_val, model_map)

    trl = PracticalTRLStage2(
        map_model=model_map,
        prior_vec=prior_vec,
        clean_loader=bn_loader_clean,  # build with clean (stable)
        steps=cfg.trl_steps,
        k_perp=cfg.trl_k_perp,
        step_size=cfg.trl_step_size,
        eta=cfg.trl_eta,
        tube_scale=0.1,  # placeholder
        max_delta_norm=cfg.trl_max_delta_norm,
        hvp_batches=cfg.trl_hvp_batches,
        store_every=cfg.trl_store_every,
    )

    print("\n>>> [TRL Stage-2 BUILD] (HVP)...")
    trl.build()

    # sweep in VAL with fixed seed by tube_scale (same as yours)
    print(f"\n>>> [TRL Stage-2 SWEEP] n_samples={cfg.trl_val_samples}, fix_bn_batches={cfg.trl_fixbn_batches}")
    best_nll = float("inf")
    best_ts = None
    sweep = []

    t_val = get_targets(val_loader)

    for i, ts in enumerate(cfg.trl_tube_scales, 1):
        print(f"[{i}/{len(cfg.trl_tube_scales)}] tube_scale={ts} ... ", end="")
        cleanup()

        try:
            set_seed(1000)  # fixo -> mesmas amostras
            trl.beta = float(ts)

            p_val, _ = trl.predict(
                val_loader,
                bn_loader_aug=tr_loader_aug,
                n_samples=cfg.trl_val_samples,
                fix_bn_batches=cfg.trl_fixbn_batches,
            )
            p_val = p_val.clamp(1e-7, 1.0)
            nll = nn.NLLLoss()(torch.log(p_val), t_val.long()).item()

            print(f"Val NLL={nll:.4f}")
            sweep.append((float(ts), float(nll)))
            if np.isfinite(nll) and nll < best_nll:
                best_nll = nll
                best_ts = float(ts)
        except Exception as e:
            print(f"Fail: {e}")

        cleanup()

    print(f">>> Vencedor TRL Stage-2 (Val NLL): tube_scale={best_ts} (Val NLL={best_nll:.4f})")

    # salva spine + config
    ensure_dir(cfg.ckpt_dir)
    spine_path = os.path.join(cfg.ckpt_dir, cfg.trl_spine)
    torch.save({
        "cfg": cfg.__dict__,
        "best_tube_scale": best_ts,
        "best_val_nll": best_nll,
        "sweep": sweep,
        "spine": trl.spine,
    }, spine_path)
    print(f">>> Spine salva em: {spine_path}")

    # final no TEST + OOD
    print(f"\n>>> [TRL Stage-2 FINAL] Test/OOD com tube_scale={best_ts}, n_samples={cfg.trl_val_samples}")
    trl.beta = float(best_ts)

    p_trl, t_ts = trl.predict(ts_loader, bn_loader_aug=tr_loader_aug, n_samples=cfg.trl_val_samples, fix_bn_batches=cfg.trl_fixbn_batches)
    p_trl_ood, _ = trl.predict(ood_loader, bn_loader_aug=tr_loader_aug, n_samples=cfg.trl_val_samples, fix_bn_batches=cfg.trl_fixbn_batches)

    return p_trl, p_trl_ood, sweep, best_ts, best_nll


# ==============================================================================
# MAIN
# ==============================================================================
def main(cfg: CFG):
    set_seed(cfg.seed)
    ensure_dir(cfg.ckpt_dir)
    cleanup()

    # data
    tr_aug, bn_clean, val, ts, ood = get_data(cfg)

    # MAP (base) for Laplace and TRL
    print("\n>>> [MAP]...")
    model_map = load_or_train_map(tr_aug, cfg)

    targets_ts = get_targets(ts)

    # Laplace => ELA/LLA + base_val
    la, base_val, p_ela, p_lla, p_ela_ood, p_lla_ood = laplace_fit_and_predict(
        model_map=model_map,
        bn_loader_clean=bn_clean,
        ts_loader=ts,
        ood_loader=ood,
        cfg=cfg
    )

    # MAP metrics
    print("\n>>> [MAP] Pred...")
    p_map = predict_probs(model_map, ts)
    p_map_ood = predict_probs(model_map, ood)

    # Deep Ensemble
    p_ens, p_ens_ood, last_model, _ = deep_ensemble(tr_aug, ts, ood, cfg)

    # SWAG (starting from the last_model of the ensemble, as in your script)
    p_swag, p_swag_ood = run_swag(tr_aug, ts, ood, last_model, cfg)

    # MC Dropout (dropout model)
    print("\n>>> [MC Dropout]...")
    mc_model = load_or_train_mcdo(tr_aug, cfg)
    p_mc = mc_dropout_predict(mc_model, ts, tr_aug, cfg)
    p_mc_ood = mc_dropout_predict(mc_model, ood, tr_aug, cfg)

    # TRL Stage-2 (HVP ablation)
    p_trl, p_trl_ood, trl_sweep, best_ts, best_val_nll = trl_stage2_run(
        model_map=model_map,
        base_val=base_val,
        bn_loader_clean=bn_clean,
        tr_loader_aug=tr_aug,
        val_loader=val,
        ts_loader=ts,
        ood_loader=ood,
        cfg=cfg
    )

    # report
    def show(name: str, pid: torch.Tensor, pood: torch.Tensor):
        acc, nll, ece, bri = calc_metrics(pid, targets_ts, cfg.num_classes)
        auc = auroc_entropy(pid, pood)
        print(f"{name:12s} | {acc*100:6.2f} | {nll:.4f} | {ece:.4f} | {bri:.4f} | {auc:.4f}")

    print("\n" + "=" * 100)
    print(" RESULTS: CIFAR-100 (ResNet-18 CIFAR-style) [MAP | Laplace(ELA/LLA) | DE | SWAG | MC-Dropout | TRL]")
    print("=" * 100)
    print("Method       | Acc %  | NLL    | ECE    | Brier  | AUROC")
    print("-" * 100)
    show("MAP", p_map, p_map_ood)
    show("ELA", p_ela, p_ela_ood)
    show("LLA", p_lla, p_lla_ood)
    show("DeepEns", p_ens, p_ens_ood)
    show("SWAG", p_swag, p_swag_ood)
    show("MC-Dropout", p_mc, p_mc_ood)
    show("TRL", p_trl, p_trl_ood)
    print("-" * 100)

    print("\n>>> TRL sweep (tube_scale, Val NLL):")
    for ts_, nll_ in trl_sweep:
        print(f"  tube_scale={ts_:<6}  Val NLL={nll_:.4f}")
    print(f">>> Best TRL tube_scale: {best_ts} | Val NLL: {best_val_nll:.4f}")


if __name__ == "__main__":
    main(CFG_)