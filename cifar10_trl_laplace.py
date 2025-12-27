import os
import copy
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from torch.nn.utils import parameters_to_vector, vector_to_parameters
from laplace import Laplace

# ==============================================================================
# GLOBAL SETTINGS
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Checkpoint name for saving the trained weights
CHECKPOINT_PATH = "resnet18_cifar10_map_relu_v2.pth"

# Training and evaluation batch size
BATCH_SIZE = 128

# Number of epochs for MAP training (50 is a good balance between time and accuracy)
EPOCHS = 50 
LEARNING_RATE = 0.1

print(f"Running on: {DEVICE}")


# ==============================================================================
# 1. DATA PREPARATION (CIFAR-10)
# ==============================================================================

def get_data():
    print(">>> Preparando Dados CIFAR-10...")

    # Standard Data Augmentation for SOTA on CIFAR-10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download of the datasets
    trainset_full = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform_train
    )
    
    # Division: 45k for Training (MAP) / 5k for Validation (TRL Grid Search)
    # We set the seed for reproducibility of the splits
    train_subset, val_subset = torch.utils.data.random_split(
        trainset_full, [45000, 5000], generator=torch.Generator().manual_seed(42)
    )

    # Dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform_test
    )
    
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    return train_loader, val_loader, test_loader


def flatten_tensors(tensors):
    """Concatenate a list of tensors into a continuous 1D vector."""
    return torch.cat([t.contiguous().view(-1) for t in tensors])


# ==============================================================================
# 2. MODEL: CUSTOMIZED RESNET-18 (ReLU)
# ==============================================================================
# We use an explicit definition to avoid compatibility issues
# with the 'curvlinops' library from Laplace. We use explicit nn.ReLU.

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True) # Explicit
        
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True) # Explicit

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 64
        
        # Geometric adaptation for CIFAR-10 (32x32 images)
        # We replaced the 7x7 stride 2 kernel with a 3x3 stride 1 to avoid reducing the dimension too much
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True) # Explicit
        
        # Standard ResNet Layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        
        # Camada final
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        out = self.linear(out)
        return out


def get_resnet_cifar_relu():
    """Instantiate the ResNet configured for CIFAR on the correct GPU."""
    return ResNetCIFAR(num_classes=10).to(DEVICE)


# ==============================================================================
# 3. TRAINING FUNCTION (MAP ESTIMATION)
# ==============================================================================

def train_or_load_map(train_loader, val_loader):
    model = get_resnet_cifar_relu()

    # If a trained checkpoint already exists, load it to save time
    if os.path.exists(CHECKPOINT_PATH):
        print(f">>> Loading MAP from checkpoint: {CHECKPOINT_PATH}")
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            return model
        except Exception as e:
            print(f"[Notice] Error loading checkpoint ({e}). Training from scratch.")

    print(f">>> Starting MAP Training from Scratch ({EPOCHS} epochs)...")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4
    )
    # Cosine Scheduler is standard for achieving SOTA in ResNets
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    start_time = time.time()
    best_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        loss_epoch = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            loss_epoch += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        scheduler.step()

        # Periodic validation
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                    outputs = model(inputs)
                    _, predicted = outputs.max(1)
                    val_total += targets.size(0)
                    val_correct += predicted.eq(targets).sum().item()
            
            val_acc = 100.0 * val_correct / val_total
            print(
                f"Ep {epoch+1:02d}/{EPOCHS} | "
                f"Loss {loss_epoch/len(train_loader):.3f} | "
                f"TrainAcc {100*correct/total:.1f}% | "
                f"ValAcc {val_acc:.1f}% | "
                f"Time {(time.time()-start_time)/60:.1f}min"
            )
            
            # Save the best model found
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), CHECKPOINT_PATH)

    print(f">>> Training Completed. Best Val Acc: {best_acc:.2f}%")
    # Load the best saved state
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    return model


# ==============================================================================
# 4. COMPLETE IMPLEMENTATION OF TRL (PracticalTRL)
# ==============================================================================

def get_hvp_function(model, loader, device, num_batches=10):
    """Returns a function that computes the Hessian-Vector product (H*v)
using a subset of data for efficiency and stability (Lanczos)."""
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Pre-loads a buffer of batches to ensure consistency in Lanczos
    data_cache = []
    it = iter(loader)
    for _ in range(num_batches):
        try:
            data_cache.append(next(it))
        except StopIteration:
            break

    def hvp(v_numpy):
        # Convert a numpy vector (from Scipy) to Torch
        v = torch.from_numpy(v_numpy).float().to(device)

        model.zero_grad()
        total_loss = 0.0
        
        # Calculate average loss over the buffer
        for x, y in data_cache:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            loss = nn.CrossEntropyLoss()(out, y)
            total_loss += loss

        loss_avg = total_loss / len(data_cache)

        # 1º Backward: Gradiente da Loss
        grads = torch.autograd.grad(loss_avg, params, create_graph=True)
        g_vec = flatten_tensors(grads)

        # 2º Backward: Gradiente do Produto Interno (Pearlmutter Trick)
        prod = torch.dot(g_vec, v)
        Hv = torch.autograd.grad(prod, params, retain_graph=False)
        Hv_vec = flatten_tensors(Hv)

        return Hv_vec.detach().cpu().numpy()

    return hvp


def fix_bn(model, loader, device, num_batches=15):
    """Recalibrates the Batch Norm statistics (Running Mean/Var)
by passing training data through the model with the current weights.
Crucial for the performance of Bayesian methods in ResNets."""
    model.train()  # Enables BN buffer updates
    with torch.no_grad():
        it = iter(loader)
        for _ in range(num_batches):
            try:
                x, _ = next(it)
                model(x.to(device))
            except StopIteration:
                break
    model.eval()   # Return to evaluation mode


class PracticalTRL:
    def __init__(
        self,
        map_model,
        prior_vec,      # Prior precision vector (diagonal)
        train_loader,   # Training DataLoader (with shuffle=True)
        steps=10,       # Spine length (T)
        k_perp=10,      # Transverse dimension (Lanczos)
        step_size=0.05, # Geometric SGD Step
        eta=0.1,        # Cross-correction rate
        tube_scale=1.0, # Beta (largura do tubo)
    ):
        self.model = copy.deepcopy(map_model)
        self.prior = prior_vec.detach().to(DEVICE)
        self.loader = train_loader

        self.T = steps
        self.k = k_perp
        self.ds = step_size
        self.eta = eta
        self.beta = tube_scale

        self.spine = []  # Stores the points of the trajectory: {theta, N, L}

    def build(self):
        """Builds the spine of the tube (Trajectory + Transverse Geometry)."""
        curr_theta = flatten_tensors(list(self.model.parameters())).detach()
        num_params = curr_theta.numel()
        
        print(f"    [TRL] Build Init: P={num_params}, T={self.T}, K={self.k}")

        # 1. Initial Base (Lanczos): Finds directions of greatest curvature (Stiff)
        # We use 10 batches to estimate H
        hvp_fn = get_hvp_function(self.model, self.loader, DEVICE, num_batches=10)
        
        op = sla.LinearOperator((num_params, num_params), matvec=hvp_fn)
        
        # 'LA' = Largest Algebraic (Maiores autovalores positivos)
        vals, vecs = sla.eigsh(op, k=self.k + 1, which="LA")

        # Ordena descendente
        idx = np.argsort(vals)[::-1][:self.k]

        N = torch.from_numpy(vecs[:, idx].copy()).float().to(DEVICE)     # [P, k]
        evals = torch.from_numpy(vals[idx].copy()).float().to(DEVICE)    # [k]

        # Protection against negative eigenvalues (non-convex noise)
        evals = torch.maximum(evals, torch.tensor(0.0, device=DEVICE))

        # 2. Tangente Inicial (v): 
        # Heuristic: Random direction projected onto the null space of the stiff directions
        vr = torch.randn(num_params, device=DEVICE)
        # Remove projection in N
        vr = vr - N @ (N.T @ vr)
        v = vr / (vr.norm() + 1e-9)

        # Iterator for the Geometric SGD
        data_iterator = iter(self.loader)

        print(f"    [TRL] Starting build loop ({self.T} steps)...")
        for _ in range(self.T):
            
            # A. Armazenar Geometria Transversal Local
            # Covariancia Transversal: Sigma = (H + Prior)^-1
            # Prior projected = diagonal of the prior in the N directions
            prior_proj = torch.sum((N ** 2) * self.prior.unsqueeze(1), dim=0) # [k]

            # Total Accuracy = Eigenvalue (Data) + Prior (Regularization)
            prec = evals + prior_proj
            
            # Safety clamp to prevent NaN if the precision is too low
            # (Here, the Boost in Priority helps a lot to keep > 1.0)
            prec = torch.clamp(prec, min=1e-6)
            
            # Scale Factor (L) = 1 / sqrt(precision)
            scale_factors = torch.rsqrt(prec)
            L = torch.diag(scale_factors)

            # Guarda o ponto
            self.spine.append({
                "theta": curr_theta.clone().cpu(),
                "N": N.clone().cpu(),
                "L": L.clone().cpu(),
            })

            # B. Dynamic Step in the Spine
            vector_to_parameters(curr_theta, self.model.parameters())
            self.model.zero_grad()

            # Pega um batch novo
            try:
                xb, yb = next(data_iterator)
            except StopIteration:
                data_iterator = iter(self.loader)
                xb, yb = next(data_iterator)

            xb = xb.to(DEVICE); yb = yb.to(DEVICE)

            loss = nn.CrossEntropyLoss()(self.model(xb), yb)
            grads = torch.autograd.grad(loss, self.model.parameters())
            g = flatten_tensors(grads)

            # Transverse Gradient (Corrects only perpendicular to the movement)
            g_perp = g - torch.dot(g, v) * v
            
            # Update: Advances v, Corrects g_perp
            theta_next = curr_theta + self.ds * v - self.eta * g_perp

            # C. Discrete Parallel Transport
            d = theta_next - curr_theta
            d_norm = d.norm()
            if d_norm > 1e-9:
                v_new = d / d_norm
            else:
                v_new = v

            # Projection and Re-orthonormalization (QR) to update N
            # N_new = N - v_new * (v_new^T * N)
            p_n = v_new @ N
            N_trans = N - torch.outer(v_new, p_n)
            
            N_new, _ = torch.linalg.qr(N_trans, mode='reduced')

            # Update statuses
            curr_theta = theta_next
            v = v_new
            N = N_new
            # We assume that 'evals' (curvature) changes slowly,
            # so we don't run Lanczos at every step to save time.

    def predict(self, loader, n_samples=10, fix_bn_batches=15):
        """Sampling Tube Ensemble Inference."""
        ens_probs = []
        targets_all = []

        # Sample S models
        for i in range(n_samples):
            # 1. Draws a point on the spine and transverse noise
            pt = self.spine[np.random.randint(len(self.spine))]
            z = torch.randn(self.k, device=DEVICE)
            
            N_dev = pt['N'].to(DEVICE)
            L_dev = pt['L'].to(DEVICE)
            th_dev = pt['theta'].to(DEVICE)

            # Mapeamento TRL: theta = theta0 + N * beta * L * z
            delta = N_dev @ (self.beta * (L_dev @ z))
            theta_sample = th_dev + delta

            # 2. Carrega pesos e Recalibra BN
            vector_to_parameters(theta_sample, self.model.parameters())
            fix_bn(self.model, self.loader, DEVICE, num_batches=fix_bn_batches)

            # 3. Prediction on the entire set
            preds_batch = []
            get_targets = (i == 0) # Pega targets only the first time
            
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(DEVICE)
                    out = self.model(x)
                    probs = torch.softmax(out, dim=1)
                    preds_batch.append(probs.cpu())
                    
                    if get_targets:
                        targets_all.append(y)

            ens_probs.append(torch.cat(preds_batch))
            print(".", end="", flush=True)

        print() # Newline
        
        if len(targets_all) > 0:
            final_targets = torch.cat(targets_all)
        else:
            final_targets = None
            
        # Average of the probabilities (Bayesian Model Averaging)
        mean_probs = torch.stack(ens_probs).mean(dim=0)
        return mean_probs, final_targets


# ==============================================================================
# 5. METRICS AND PLOT UTILITIES
# ==============================================================================

def calc_all_metrics(probs, targets):
    # Robust numerical protection to prevent NaN in NLL
    p = probs.clamp(1e-7, 1.0 - 1e-7)

    # NLL
    nll = nn.NLLLoss()(torch.log(p), targets.long()).item()

    # Accuracy
    acc = p.argmax(1).eq(targets).float().mean().item()

    # ECE (Expected Calibration Error)
    confs, preds = p.max(1)
    accs = preds.eq(targets)
    ece = 0.0
    bins = torch.linspace(0, 1, 16) # 15 bins
    for i in range(15):
        mask = (confs > bins[i]) & (confs <= bins[i + 1])
        if mask.sum() > 0:
            bin_acc = accs[mask].float().mean()
            bin_conf = confs[mask].mean()
            ece += torch.abs(bin_conf - bin_acc) * (mask.float().sum() / len(p))

    # Brier Score
    oh = F.one_hot(targets.long(), 10).float()
    brier = ((p - oh) ** 2).sum(dim=1).mean().item()

    return acc, nll, ece.item(), brier


def laplace_predict_loader(la, loader, pred_type, n_samples=30):
    """    Wrapper for Laplace process prediction processing batch-by-batch
    to avoid memory overflow with the Jacobian."""
    all_preds = []
    # Determine the linking method
    # 'nn' (ELA) exige amostragem de pesos (mc)
    # 'glm' (LLA) can use probit or mc. We use mc for consistency.
    link = "mc" if pred_type == "nn" else "probit"
    
    # Se for GLM MC, precisa passar n_samples
    # If it is Probit, n_samples is ignored
    
    for x, _ in loader:
        x = x.to(DEVICE)
        
        # Safe call compatible with Laplace-torch > 0.1
        if pred_type == "nn":
             out = la(x, pred_type="nn", link_approx="mc", n_samples=n_samples)
        else:
             # LLA (GLM) with Probit is analytical and fast
             out = la(x, pred_type="glm", link_approx="probit")
             
        all_preds.append(out.detach().cpu())
        
    return torch.cat(all_preds)


def plot_final(map_p, ela_p, lla_p, trl_p, targets):
    plt.figure(figsize=(12, 5))

    # 1. Reliability Diagram
    plt.subplot(1, 2, 1)

    def get_curve(p, t):
        c, pr = p.max(1); ac = pr.eq(t)
        X, Y = [], []
        bins = np.linspace(0, 1, 11)
        for k in range(10):
            m = (c > bins[k]) & (c <= bins[k + 1])
            if m.any():
                X.append(c[m].mean().item())
                Y.append(ac[m].float().mean().item())
        return X, Y

    methods = [
        ("MAP", map_p, "gray", 1, "o"),
        ("ELA", ela_p, "orange", 1, "o"),
        ("LLA", lla_p, "blue", 1, "o"),
        ("TRL (Ours)", trl_p, "green", 2, "*")
    ]

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
    for label, prob, col, lw, mark in methods:
        x, y = get_curve(prob, targets)
        plt.plot(x, y, label=label, color=col, linewidth=lw, marker=mark)

    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram (CIFAR-10)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 2. Entropy Histogram
    plt.subplot(1, 2, 2)
    def get_ent(p):
        return -torch.sum(p * torch.log(p + 1e-9), dim=1).numpy()

    for label, prob, col, lw, _ in methods:
        ent = get_ent(prob)
        # Different style for TRL
        htype = 'step' if 'TRL' in label else 'bar'
        alpha = 1.0 if 'TRL' in label else 0.3
        
        plt.hist(ent, bins=30, density=True, alpha=alpha, 
                 label=label, color=col, histtype=htype, linewidth=lw)

    plt.xlabel("Predictive Entropy")
    plt.title("Entropy Distribution")
    plt.legend()

    plt.tight_layout()
    plt.savefig("cifar10_results_final.png")
    print(">>> Gráficos salvos: cifar10_results_final.png")


def run_real_grid_search(model, prior, tr_dl, val_dl):
    """    Run TRL with some configurations on the validation set and return the best one."""
    print("\n>>> [GRID SEARCH] Otimizando TRL no Validation Set...")

    grid_params = {
        'steps': [10, 20],        
        'step_size': [0.03, 0.05],
        'tube_scale': [0.5, 1.0] # Test minor scales
    }
    
    keys, vals = zip(*grid_params.items())
    combos = [dict(zip(keys, x)) for x in itertools.product(*vals)]
    
    best_nll = float('inf')
    best_cfg = None
    
    for i, cfg in enumerate(combos):
        print(f"[{i+1}/{len(combos)}] Testando {cfg} ... ", end="")
        try:
            # Light version for validation (k=5)
            trl = PracticalTRL(model, prior, tr_dl, k_perp=5, **cfg)
            trl.build()
            
            # Quick inference on Val (S=3 samples)
            p_val, t_val = trl.predict(val_dl, n_samples=3, fix_bn_batches=5)
            
            # Calc NLL
            p_safe = p_val.clamp(1e-7, 1-1e-7)
            nll = nn.NLLLoss()(torch.log(p_safe), t_val.long()).item()
            print(f"NLL={nll:.4f}")
            
            if nll < best_nll:
                best_nll = nll
                best_cfg = cfg
                
        except Exception as e:
            print(f"Falha: {e}")
    
    # Fallback caso tudo falhe (evita crash do main)
    if best_cfg is None:
        print("[CRITICAL] Grid Search falhou. Usando config padrão segura.")
        best_cfg = {'steps': 10, 'step_size': 0.03, 'tube_scale': 0.5}
        
    print(f">>> Vencedor Grid: {best_cfg} (NLL Valid: {best_nll:.4f})")
    return best_cfg


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    
    # 1. Load Data
    tr, val, ts = get_data()
    
    # 2. Carrega MAP
    model = train_or_load_map(tr, val)
    torch.cuda.empty_cache() # Clear memory

    # 3. BASELINE: LAPLACE LAST-LAYER (FULL)
    # The "Last-Layer" strategy is robust, fast, and fits in memory.
    # It is a SOTA baseline for efficient calibration.
    print("\n>>> [Baseline] Ajustando Laplace (Last-Layer)...")
    
    # Subsample safe for the Laplace Fit
    subset_len = 5000 
    inds = torch.randperm(len(tr.dataset))[:subset_len]
    tr_sub = torch.utils.data.Subset(tr.dataset, inds)
    la_loader = torch.utils.data.DataLoader(tr_sub, batch_size=32, shuffle=True)
    
    la = Laplace(
        model, 
        likelihood="classification", 
        subset_of_weights="last_layer", 
        hessian_structure="full"
    )
    
    la.fit(la_loader)
    print("    Otimizando Prior (MargLik)...")
    la.optimize_prior_precision(method="marglik")
    
    # 4. PREPARE PRIOR FOR TRL (FULL-NETWORK)
    # Here we expand the prior of the last layer to the entire network.
    # + WE APPLY BOOSTING TO PREVENT EXPLOSION IN CONVOLUTIONS.
    
    prior_raw = la.prior_precision
    # If the last layer is full, prior_raw is a matrix. If diag, it is a vector. If scalar...
    # Marglik usually gives scalar or vector for layer.
    
    if torch.is_tensor(prior_raw) and prior_raw.ndim > 1:
        # If it is a full precision matrix (KxK), take the average of the diagonal.
        base_val = prior_raw.diag().mean().item()
    elif torch.is_tensor(prior_raw):
        base_val = prior_raw.mean().item()
    else:
        base_val = float(prior_raw)
        
    print(f"\n[TRL Setup] Prior Base (LL): {base_val:.5f}")
    
    # >>> BOOSTING <<<
    # We increased the regularization for the TRL Full-Network
    boost_factor = 50.0 
    safe_prior_val = max(base_val * boost_factor, 5.0) 
    print(f"[TRL Setup] Prior Boosted (Full): {safe_prior_val:.5f}")
    
    # Create full isotropic vector
    num_total_params = sum(p.numel() for p in model.parameters())
    prior_vec_trl = torch.full((num_total_params,), safe_prior_val, device=DEVICE)
    
    
    # 5. FINAL EVALUATION
    targets = torch.cat([y for _,y in ts])
    
    # Avalia ELA/LLA
    print("\n>>> Avaliando Baselines...")
    p_ela = laplace_predict_loader(la, ts, 'nn', 25)
    p_lla = laplace_predict_loader(la, ts, 'glm', 25) # Probit inside
    
    r_ela = calc_all_metrics(p_ela, targets)
    r_lla = calc_all_metrics(p_lla, targets)

    # Avalia TRL (Grid + Final)
    best_cfg = run_real_grid_search(model, prior_vec_trl, tr, val)
    
    print(f"\n>>> [FINAL RUN] TRL High-Fi (k=20, S=25)...")
    trl = PracticalTRL(model, prior_vec_trl, tr, 
                       k_perp=20, # Highest rank for final test
                       steps=best_cfg['steps'],
                       step_size=best_cfg['step_size'],
                       tube_scale=best_cfg['tube_scale'])
    trl.build()
    p_trl, _ = trl.predict(ts, n_samples=25, fix_bn_batches=25)
    
    r_trl = calc_all_metrics(p_trl, targets)
    
    # Avalia MAP
    p_m=[]
    model.eval()
    with torch.no_grad():
        for x,_ in ts: p_m.append(torch.softmax(model(x.to(DEVICE)),1).cpu())
    p_map = torch.cat(p_m)
    r_map = calc_all_metrics(p_map, targets)

    # TABELA FINAL
    print("\n" + "="*50)
    print(" RESULTS (ResNet-18 CIFAR-10)")
    print("="*50)
    print(f"M   | Acc    | NLL    | ECE    | Brier")
    print(f"MAP | {r_map[0]:.4f} | {r_map[1]:.4f} | {r_map[2]:.4f} | {r_map[3]:.4f}")
    print(f"ELA | {r_ela[0]:.4f} | {r_ela[1]:.4f} | {r_ela[2]:.4f} | {r_ela[3]:.4f}")
    print(f"LLA | {r_lla[0]:.4f} | {r_lla[1]:.4f} | {r_lla[2]:.4f} | {r_lla[3]:.4f}")
    print(f"TRL | {r_trl[0]:.4f} | {r_trl[1]:.4f} | {r_trl[2]:.4f} | {r_trl[3]:.4f}")
    
    plot_final(p_map, p_ela, p_lla, p_trl, targets)