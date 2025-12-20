import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn.utils import vector_to_parameters, parameters_to_vector
from sklearn.metrics import roc_auc_score

# ==============================================================================
# CONFIGURAÇÕES
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128
EPOCHS = 50 
LR = 0.1
MOMENTUM = 0.9
WD = 5e-4

# Configs Específicas dos Baselines
ENSEMBLE_SIZE = 5         # Padrão de ouro (Lakshminarayanan et al.)
MCD_SAMPLES = 25          # Amostras para MC Dropout
MCD_P = 0.1               # Taxa de Dropout
SWAG_SAMPLES = 25         # Amostras para SWAG
SWAG_START_EPOCH = 30     # Começa a coletar estatísticas após 60% do treino
SWAG_LR = 0.01            # LR constante para a fase de coleta

print(f"Executando Baselines SOTA em: {DEVICE}")

# ==============================================================================
# 1. DADOS (ID: CIFAR-10, OOD: SVHN)
# ==============================================================================
def get_loaders():
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    t_train = transforms.Compose([
        transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(*stats)
    ])
    t_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    # CIFAR-10
    trainset_full = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=t_train)

     # Usamos a mesma seed (42) para garantir que as 45k imagens sejam as mesmas do TRL
    tr_sub, _ = torch.utils.data.random_split(trainset_full, [45000, 5000], generator=torch.Generator().manual_seed(42))

    tr_loader = torch.utils.data.DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=t_test)
    ts_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # SVHN (OOD) - Usando subset para rapidez na validação
    svhn = torchvision.datasets.SVHN(root='./data', split='test', download=True, transform=t_test)
    ood_sub = torch.utils.data.Subset(svhn, range(2000))
    ood_loader = torch.utils.data.DataLoader(ood_sub, batch_size=BATCH_SIZE, shuffle=False)

    return tr_loader, ts_loader, ood_loader

# ==============================================================================
# 2. MODELOS (ResNet Padrão e ResNet com Dropout)
# ==============================================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)  # <-- aqui

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out
    
class ResNetCIFAR(nn.Module):
    def __init__(self, num_classes=10, use_dropout=False, drop_p=0.1):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 64
        self.use_dropout = use_dropout
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Dropout antes da linear (Gal & Ghahramani para CNNs)
        if self.use_dropout:
            self.dropout = nn.Dropout(p=drop_p)
            
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes * BasicBlock.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.flatten(1)
        
        if self.use_dropout:
            out = self.dropout(out)
            
        out = self.linear(out)
        return out

# ==============================================================================
# 3. UTILS: METRICS, FIX_BN, TRAINING
# ==============================================================================
def get_metrics(p, t):
    p = p.clamp(1e-7, 1-1e-7)
    nll = nn.NLLLoss()(torch.log(p), t.long()).item()
    acc = p.argmax(1).eq(t).float().mean().item()
    
    # ECE
    c, _ = p.max(1); ac = p.argmax(1).eq(t)
    ece = 0.0; bins = torch.linspace(0, 1, 16)
    for i in range(15):
        m = (c > bins[i]) & (c <= bins[i+1])
        if m.sum() > 0:
            ece += torch.abs(c[m].mean() - ac[m].float().mean()) * (m.float().sum() / len(p))
            
    # Brier
    oh = F.one_hot(t.long(), 10).float()
    bri = ((p - oh)**2).sum(1).mean().item()
    return acc, nll, ece.item(), bri

def entropy(p): return -torch.sum(p * torch.log(p.clamp(1e-9, 1.0)), dim=1).cpu().numpy()

def fix_bn(model, loader):
    """Recalibra BN para SWAG"""
    model.train()
    with torch.no_grad():
        for x, _ in loader:
            model(x.to(DEVICE))
    model.eval()

def train_epoch(model, loader, opt, crit):
    model.train()
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        opt.zero_grad()
        loss = crit(model(x), y)
        loss.backward()
        opt.step()

# ==============================================================================
# 4. IMPLEMENTAÇÃO DOS BASELINES
# ==============================================================================

# --- A. DEEP ENSEMBLES ---
def run_ensembles(tr_loader, ts_loader, ood_loader):
    print("\n>>> [1/3] Deep Ensembles (M=5)...")
    models = []
    
    for i in range(ENSEMBLE_SIZE):
        path = f"resnet_ens_{i}.pth"
        m = ResNetCIFAR().to(DEVICE)
        
        if os.path.exists(path):
            print(f"    Carregando Ensemble {i}...")
            m.load_state_dict(torch.load(path, map_location=DEVICE))
        else:
            print(f"    Treinando Ensemble {i}...")
            opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
            crit = nn.CrossEntropyLoss()
            
            for ep in range(EPOCHS):
                train_epoch(m, tr_loader, opt, crit)
                sched.step()
            torch.save(m.state_dict(), path)
        
        m.eval()
        models.append(m)

    # Inferência Ensemble
    def predict_ens(loader):
        probs_all = []
        targets = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                outs = torch.stack([torch.softmax(m(x), 1) for m in models]) # [M, B, 10]
                mean_p = outs.mean(0)
                probs_all.append(mean_p.cpu())
                targets.append(y)
        return torch.cat(probs_all), torch.cat(targets)

    p_id, t_id = predict_ens(ts_loader)
    p_ood, _   = predict_ens(ood_loader)
    return p_id, t_id, p_ood

# --- B. MC DROPOUT ---
def run_mcdropout(tr_loader, ts_loader, ood_loader):
    print("\n>>> [2/3] MC Dropout (p=0.1)...")
    path = "resnet_mcd.pth"
    m = ResNetCIFAR(use_dropout=True, drop_p=MCD_P).to(DEVICE)
    
    if os.path.exists(path):
        print("    Carregando modelo MCD...")
        m.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        print("    Treinando modelo MCD...")
        opt = torch.optim.SGD(m.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        crit = nn.CrossEntropyLoss()
        for ep in range(EPOCHS):
            train_epoch(m, tr_loader, opt, crit)
            sched.step()
        torch.save(m.state_dict(), path)

    # Inferência MCD: Mantém dropout ativo (train mode)
    def predict_mcd(loader):
        probs_all = []
        targets = []
        m.train() # Habilita Dropout
        with torch.no_grad():
            for x, y in loader:
                x = x.to(DEVICE)
                # S forward passes
                outs = []
                for _ in range(MCD_SAMPLES):
                    outs.append(torch.softmax(m(x), 1))
                mean_p = torch.stack(outs).mean(0)
                probs_all.append(mean_p.cpu())
                targets.append(y)
        m.eval()
        return torch.cat(probs_all), torch.cat(targets)

    p_id, t_id = predict_mcd(ts_loader)
    p_ood, _   = predict_mcd(ood_loader)
    return p_id, t_id, p_ood

# --- C. SWAG (DIAGONAL) ---
class SWAG(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.register_buffer('n_models', torch.zeros(1))
        # Armazena média e quadrado da média para cada parâmetro
        self.params_mean = nn.ParameterDict()
        self.params_sq_mean = nn.ParameterDict()
        
        for name, param in base_model.named_parameters():
            self.params_mean[name.replace('.', '_')] = nn.Parameter(torch.zeros_like(param), requires_grad=False)
            self.params_sq_mean[name.replace('.', '_')] = nn.Parameter(torch.zeros_like(param), requires_grad=False)

    def collect_model(self, model):
        self.n_models += 1
        factor = 1.0 / self.n_models.item()
        for name, param in model.named_parameters():
            key = name.replace('.', '_')
            curr_mean = self.params_mean[key]
            curr_sq = self.params_sq_mean[key]
            # Média Online
            new_mean = curr_mean * (1.0 - factor) + param.data * factor
            new_sq = curr_sq * (1.0 - factor) + (param.data ** 2) * factor
            self.params_mean[key].data.copy_(new_mean)
            self.params_sq_mean[key].data.copy_(new_sq)

    def sample_weights(self, scale=0.5):
        # Amostra N(mean, scale * diag_cov)
        for name, param in self.base_model.named_parameters():
            key = name.replace('.', '_')
            mean = self.params_mean[key]
            sq_mean = self.params_sq_mean[key]
            
            # Var = E[x^2] - (E[x])^2. Clamp para evitar erros numéricos negativos.
            var = torch.clamp(sq_mean - mean ** 2, min=1e-30)
            std = torch.sqrt(var)
            
            # Amostragem
            z = torch.randn_like(param)
            param.data.copy_(mean + (scale ** 0.5) * std * z)

def run_swag(tr_loader, ts_loader, ood_loader):
    print("\n>>> [3/3] SWAG-Diagonal...")
    path = "resnet_swag.pth"
    model = ResNetCIFAR().to(DEVICE)
    swag_model = SWAG(model).to(DEVICE)
    
    if os.path.exists(path):
        print("    Carregando estado SWAG...")
        # Carregamento simplificado: assumimos que salvamos o state_dict do wrapper
        swag_model.load_state_dict(torch.load(path, map_location=DEVICE))
    else:
        print("    Treinando SWAG...")
        opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WD)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        crit = nn.CrossEntropyLoss()
        
        # Fase 1: Treino Padrão
        for ep in range(SWAG_START_EPOCH):
            train_epoch(model, tr_loader, opt, crit)
            sched.step()
            
        # Fase 2: Coleta SWAG (LR Constante)
        # Muda otimizador para LR fixo
        opt_swag = torch.optim.SGD(model.parameters(), lr=SWAG_LR, momentum=MOMENTUM, weight_decay=WD)
        
        for ep in range(SWAG_START_EPOCH, EPOCHS):
            train_epoch(model, tr_loader, opt_swag, crit)
            swag_model.collect_model(model)
            
        torch.save(swag_model.state_dict(), path)

    # Inferência SWAG
    def predict_swag(loader):
        probs_all = []
        targets = []
        
        # Precisa coletar targets uma vez
        first_pass = True
        
        # Buffer de predições acumuladas
        list_outs = []
        
        for s in range(SWAG_SAMPLES):
            # 1. Amostra pesos
            swag_model.sample_weights(scale=1.0)
            # 2. Recalibra BN (Crucial!)
            fix_bn(swag_model.base_model, tr_loader)
            
            swag_model.base_model.eval()
            
            epoch_probs = []
            epoch_targets = []
            with torch.no_grad():
                for x, y in loader:
                    x = x.to(DEVICE)
                    p = torch.softmax(swag_model.base_model(x), 1).cpu()
                    epoch_probs.append(p)
                    if first_pass: epoch_targets.append(y)
            
            list_outs.append(torch.cat(epoch_probs))
            if first_pass: 
                targets = torch.cat(epoch_targets)
                first_pass = False
            print(".", end="", flush=True)
            
        print()
        mean_p = torch.stack(list_outs).mean(0)
        return mean_p, targets

    p_id, t_id = predict_swag(ts_loader)
    p_ood, _   = predict_swag(ood_loader)
    return p_id, t_id, p_ood

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    tr, ts, ood = get_loaders()
    
    # Rodar Métodos
    ens_id, ens_t, ens_ood = run_ensembles(tr, ts, ood)
    mcd_id, mcd_t, mcd_ood = run_mcdropout(tr, ts, ood)
    swa_id, swa_t, swa_ood = run_swag(tr, ts, ood)
    
    # Calcular Métricas
    def report(name, pid, tid, pood):
        acc, nll, ece, bri = get_metrics(pid, tid)
        
        # AUROC
        ent_id = entropy(pid)
        ent_ood = entropy(pood)
        y_true = np.concatenate([np.zeros(len(ent_id)), np.ones(len(ent_ood))])
        y_scores = np.concatenate([ent_id, ent_ood])
        auroc = roc_auc_score(y_true, y_scores)
        
        print(f"{name:10s} | {acc*100:6.2f} | {nll:.4f} | {ece:.4f} | {bri:.4f} | {auroc:.4f}")

    print("\n\n==================================================================")
    print(" SOTA BASELINES RESULTS (ResNet-18 @ CIFAR-10 / OOD: SVHN)")
    print("==================================================================")
    print("Method     | Acc %  |  NLL   |  ECE   | Brier  | OOD AUROC")
    print("-----------|--------|--------|--------|--------|----------")
    
    report("Ensemble", ens_id, ens_t, ens_ood)
    report("MC-Drop", mcd_id, mcd_t, mcd_ood)
    report("SWAG", swa_id, swa_t, swa_ood)
    print("==================================================================")