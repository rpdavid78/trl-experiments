import os
import copy
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
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
# CONFIGURAÇÕES
# ==============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = "resnet18_cifar10_map_relu_v2.pth"
BATCH_SIZE = 128
EPOCHS = 50 
LEARNING_RATE = 0.1

print(f"Executando em: {DEVICE}")

# ==============================================================================
# 1. DADOS
# ==============================================================================

def get_data_loaders():
    print(">>> Preparando Dados CIFAR-10 e SVHN...")
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    t_train = transforms.Compose([transforms.RandomCrop(32, 4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(*stats)])
    t_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    trainset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=t_train)
    tr_sub, val_sub = torch.utils.data.random_split(trainset, [45000, 5000], generator=torch.Generator().manual_seed(42))

    train_loader = torch.utils.data.DataLoader(tr_sub, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_sub, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=t_test)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    # SVHN (OOD)
    ood_set = torchvision.datasets.SVHN(root="./data", split='test', download=True, transform=t_test)
    ood_sub = torch.utils.data.Subset(ood_set, range(2000))
    ood_loader = torch.utils.data.DataLoader(ood_sub, batch_size=128, shuffle=False)

    return train_loader, val_loader, test_loader, ood_loader

# ==============================================================================
# 2. MODELO
# ==============================================================================
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)
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
    def __init__(self, num_classes=10):
        super(ResNetCIFAR, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.linear = nn.Linear(512, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
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

def get_resnet(): return ResNetCIFAR(10).to(DEVICE)

def train_or_load(tr, val):
    m = get_resnet()
    if os.path.exists(CHECKPOINT_PATH):
        print(f">>> Loading Checkpoint: {CHECKPOINT_PATH}")
        try: m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        except: pass
        else: return m

    print(">>> Training MAP...")
    opt = torch.optim.SGD(m.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    crit = nn.CrossEntropyLoss()
    best_acc = 0
    
    for ep in range(EPOCHS):
        m.train(); lsum=0; corr=0; tot=0
        for x,y in tr:
            x,y=x.to(DEVICE),y.to(DEVICE)
            opt.zero_grad()
            out = m(x); loss=crit(out,y)
            loss.backward(); opt.step()
            lsum+=loss.item(); _,p=out.max(1); corr+=p.eq(y).sum().item(); tot+=y.size(0)
        sched.step()
        if (ep+1)%10==0:
            m.eval(); vcorr=0; vtot=0
            with torch.no_grad():
                for x,y in val:
                    out=m(x.to(DEVICE)); _,p=out.max(1)
                    vcorr+=p.eq(y.to(DEVICE)).sum().item(); vtot+=y.size(0)
            vacc = 100*vcorr/vtot
            print(f"Ep {ep+1} | Val Acc: {vacc:.4f}")
            if vacc>best_acc: best_acc=vacc; torch.save(m.state_dict(), CHECKPOINT_PATH)
            
    m.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    return m

# ==============================================================================
# 4. TRL UTILS
# ==============================================================================
def flatten_tensors(tensors): 
    return torch.cat([t.contiguous().view(-1) for t in tensors])

def get_hvp_function(model, loader, device, num_batches=8):
    params = [p for p in model.parameters() if p.requires_grad]
    cache = []
    it = iter(loader)
    for _ in range(num_batches): 
        try: cache.append(next(it))
        except: break
    
    def hvp(v_np):
        v = torch.from_numpy(v_np).to(device).float()
        model.zero_grad(); loss=0
        for x,y in cache: 
            loss += nn.CrossEntropyLoss()(model(x.to(device)), y.to(device))
        loss /= len(cache)
        grads = torch.autograd.grad(loss, params, create_graph=True)
        gvec = flatten_tensors(grads)
        prod = torch.dot(gvec, v)
        return flatten_tensors(torch.autograd.grad(prod, params, retain_graph=False)).cpu().numpy()
    return hvp

def fix_bn(model, loader, device, num_batches=15):
    model.train()
    with torch.no_grad():
        it = iter(loader)
        for _ in range(num_batches):
            try: model(next(it)[0].to(device))
            except: break
    model.eval()

class PracticalTRL:
    def __init__(self, model, prior, loader, steps, k_perp, ds, eta, scale):
        self.model = copy.deepcopy(model); self.prior = prior.detach().to(DEVICE)
        self.loader = loader; self.T = steps; self.k = k_perp
        self.ds = ds; self.eta = eta; self.beta = scale
        self.spine = []
        
    def build(self):
        curr = flatten_tensors(list(self.model.parameters())).detach()
        hvp = get_hvp_function(self.model, self.loader, DEVICE)
        op = sla.LinearOperator((curr.numel(),curr.numel()), matvec=hvp)
        ev, ec = sla.eigsh(op, k=self.k+1, which='LA')
        
        idx = np.argsort(ev)[::-1][:self.k]
        N = torch.from_numpy(ec[:,idx].copy()).float().to(DEVICE)
        evals = torch.from_numpy(ev[idx].copy()).float().to(DEVICE)
        v_raw = torch.randn(curr.numel(), device=DEVICE)
        v = v_raw - N@(N.T@v_raw); v=v/(v.norm()+1e-9)
        
        dit = iter(self.loader)
        for _ in range(self.T):
            pproj = torch.sum((N**2)*self.prior.unsqueeze(1), 0)
            L = torch.diag(torch.rsqrt(torch.clamp(evals + pproj, min=1e-6)))
            self.spine.append({'th':curr.clone().cpu(), 'N':N.clone().cpu(), 'L':L.clone().cpu()})
            
            vector_to_parameters(curr, self.model.parameters()); self.model.zero_grad()
            try: xb,yb=next(dit)
            except: dit=iter(self.loader); xb,yb=next(dit)
            ls = nn.CrossEntropyLoss()(self.model(xb.to(DEVICE)), yb.to(DEVICE))
            g = flatten_tensors(torch.autograd.grad(ls, self.model.parameters()))
            
            gp = g - torch.dot(g,v)*v
            nxt = curr + self.ds*v - self.eta*gp
            d = nxt - curr; vn = d/(d.norm()+1e-9)
            np_ = vn@N; nt = N - torch.outer(vn, np_); 
            nn_,_ = torch.linalg.qr(nt, mode='reduced') # Correção aqui
            
            curr,v,N = nxt, vn, nn_ # Correção aqui (era Nn)

    def predict(self, loader, n_samples, fixb, boost=1.0):
        eff_beta = self.beta * boost
        ens = []; t_list = []
        for i in range(n_samples):
            pt = self.spine[np.random.randint(len(self.spine))]
            z = torch.randn(self.k, device=DEVICE)
            th = pt['th'].to(DEVICE) + pt['N'].to(DEVICE) @ (eff_beta * pt['L'].to(DEVICE) @ z)
            vector_to_parameters(th, self.model.parameters())
            fix_bn(self.model, self.loader, DEVICE, fixb)
            ps = []
            get_t = (i==0)
            with torch.no_grad():
                for x, y in loader:
                    ps.append(torch.softmax(self.model(x.to(DEVICE)),1).cpu())
                    if get_t: t_list.append(y.cpu())
            ens.append(torch.cat(ps))
            print(".",end="",flush=True)
        print()
        return torch.stack(ens).mean(0), torch.cat(t_list)

# ==============================================================================
# 5. METRICS & PLOT (CORRIGIDO)
# ==============================================================================
def get_metrics(p, t):
    p = p.clamp(1e-7, 1-1e-7)
    nll = nn.NLLLoss()(torch.log(p), t.long()).item()
    acc = p.argmax(1).eq(t).float().mean().item()
    c, _ = p.max(1); ac = p.argmax(1).eq(t); ece=0; bins=torch.linspace(0,1,16)
    for i in range(15):
        m = (c>bins[i]) & (c<=bins[i+1])
        if m.sum()>0: ece += abs(c[m].mean()-ac[m].float().mean()) * (m.float().sum()/len(p))
    oh = F.one_hot(t.long(), 10).float(); bri=((p-oh)**2).sum(1).mean().item()
    return acc, nll, ece, bri

def entropy(p): return -torch.sum(p*torch.log(p.clamp(1e-9,1)),1).numpy()

# FUNÇÃO CORRIGIDA PARA RECEBER 5 ARGUMENTOS
def plot_final(ps, tg, labels): 
    # Cores fixas baseadas na ordem esperada
    cols = ['gray', 'orange', 'blue', 'green']

    plt.figure(figsize=(12,5)); plt.subplot(1,2,1)
    def crv(p, t):
        c, pr = p.max(1); ac = pr.eq(t); X,Y=[],[]
        bi=np.linspace(0,1,11)
        for i in range(10):
            m=(c>bi[i])&(c<=bi[i+1])
            if m.any(): X.append(c[m].mean().item()); Y.append(ac[m].float().mean().item())
        return X,Y
    
    plt.plot([0,1],[0,1],'k--',alpha=0.3)
    for p,l,c in zip(ps, labels, cols):
        x,y=crv(p,tg); plt.plot(x,y,label=l,color=c,marker='o')
    plt.legend(); plt.title('Reliability'); plt.xlabel('Conf'); plt.ylabel('Acc')
    
    plt.subplot(1,2,2)
    for p,l,c in zip(ps, labels, cols):
        plt.hist(entropy(p), 30, density=True, alpha=1 if 'TRL' in l else 0.3, color=c, label=l, histtype='step' if 'TRL' in l else 'bar')
    plt.legend(); plt.title('Entropy'); plt.savefig('final_cifar.png')
    print(">>> Plot salvo!")

def laplace_pred(la, ld, pt, ns=30):
    ps=[]
    for x,_ in ld:
        out = la(x.to(DEVICE), pred_type=pt, link_approx='mc' if pt=='nn' else 'probit', n_samples=ns)
        ps.append(out.detach().cpu())
    return torch.cat(ps)

# OOD EVAL
def eval_ood_complete(svhn, model, la, trl, p_map, p_ela, p_lla, p_trl_id):
    print("\n>>> Avaliando OOD (SVHN)...")
    p_ood_map=[]; 
    with torch.no_grad():
        for x,_ in svhn: p_ood_map.append(torch.softmax(model(x.to(DEVICE)),1).cpu())
    p_ood_map=torch.cat(p_ood_map)
    
    print("    ELA OOD...")
    p_ood_ela = laplace_pred(la, svhn, 'nn', 30)
    print("    LLA OOD...")
    p_ood_lla = laplace_pred(la, svhn, 'glm', 30)
    
    print("    TRL OOD (Boost x2)...")
    # TRL predict retorna (probs, targets). Pegamos so probs [0]
    p_ood_trl, _ = trl.predict(svhn, 25, 20, boost=2.0)
    
    # AUROC
    def auc(pid, pood):
        y = np.concatenate([np.zeros(len(pid)), np.ones(len(pood))])
        s = np.concatenate([entropy(pid), entropy(pood)])
        return roc_auc_score(y, s)
        
    print("\n---------------- OOD AUROC ----------------")
    print(f"MAP: {auc(p_map, p_ood_map):.4f}")
    print(f"ELA: {auc(p_ela, p_ood_ela):.4f}")
    print(f"LLA: {auc(p_lla, p_ood_lla):.4f}")
    # Usando TRL ID normal vs TRL OOD boosted
    print(f"TRL: {auc(p_trl_id, p_ood_trl):.4f} (OOD Boosted)")
    print("-------------------------------------------")

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == '__main__':
    tr, val, ts, ood = get_data_loaders()
    m = train_or_load(tr, val)
    torch.cuda.empty_cache()
    
    # 2. Laplace Baseline (LL)
    print("\n>>> Fitting Laplace (LL Full)...")
    idx = torch.randperm(len(tr.dataset))[:5000]
    sub = torch.utils.data.Subset(tr.dataset, idx)
    dl = torch.utils.data.DataLoader(sub, 32)
    
    la = Laplace(m, 'classification', subset_of_weights='last_layer', hessian_structure='full')
    la.fit(dl); la.optimize_prior_precision(method='marglik')
    
    # Prior p/ TRL (Boost)
    raw = la.prior_precision.mean().item()
    bst = max(raw*50.0, 10.0)
    p_vec = torch.full((sum(p.numel() for p in m.parameters()),), bst, device=DEVICE)
    print(f"    Prior Boosted: {bst:.2f}")
    
    targets = torch.cat([y for _, y in ts])
    
    # Baselines
    print(">>> Baselines...")
    pe = laplace_pred(la, ts, 'nn', 30)
    pl = laplace_pred(la, ts, 'glm', 30)
    re = get_metrics(pe, targets); rl = get_metrics(pl, targets)
    
    # 3. TRL Grid (Mock - usar best fixo p/ economizar tempo se quiser)
    best_cfg = {'st':20, 'ds':0.03, 'sc':1.0} 
    # run_grid(m, p_vec, tr, val) # Se quiser rodar
    
    print(f"Running TRL {best_cfg}...")
    trl = PracticalTRL(m, p_vec, tr, best_cfg['st'], 10, best_cfg['ds'], 0.1, best_cfg['sc'])
    trl.build()
    pt, _ = trl.predict(ts, 25, 20)
    rt = get_metrics(pt, targets)
    
    # 4. MAP
    pm=[]; m.eval()
    with torch.no_grad():
        for x,_ in ts: pm.append(torch.softmax(m(x.to(DEVICE)),1).cpu())
    pm = torch.cat(pm); rm = get_metrics(pm, targets)
    
    # Report
    print("\n=================================")
    print("FINAL METRICS (ResNet CIFAR-10)")
    print(f"MAP | {rm[0]:.4f} | {rm[1]:.4f} | {rm[2]:.4f}")
    print(f"ELA | {re[0]:.4f} | {re[1]:.4f} | {re[2]:.4f}")
    print(f"LLA | {rl[0]:.4f} | {rl[1]:.4f} | {rl[2]:.4f}")
    print(f"TRL | {rt[0]:.4f} | {rt[1]:.4f} | {rt[2]:.4f}")
    
   #Agrupa as predições numa lista
    probs_list = [pm, pe, pl, pt]
    labels_list = ['MAP', 'ELA', 'LLA', 'TRL']
    
    # Chama com 3 argumentos
    plot_final(probs_list, targets, labels_list) 
    
    # 5. OOD Evaluation
    eval_ood_complete(ood, m, la, trl, pm, pe, pl, pt)