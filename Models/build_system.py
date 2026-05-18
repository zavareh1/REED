# build_system.py
from __future__ import annotations
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from config import Config
from utils.model_adapter import TorchModelAdapter
from clients.client import Client, ClientState
from trainers.torch_trainer import TorchTrainer
from comm.comm import Comm
from comm.gain_norm import NoNorm, PilotNorm, PilotlessNorm, GivenNorm, PathlossNorm
from server.server import Server
from server.sampling import sample_clients
from server.eval import evaluate_global

from data.partition import (
        iid_splits, dirichlet_splits, shards_partition,
        label_skew_n_classes, quantity_skew, majority_class_skew,
    )


def make_gain_norm(cfg: Config):
    mode = cfg.algo.gain_norm.mode
    if mode == "none":       return NoNorm()
    if mode == "pilot":      return PilotNorm()
    if mode == "pilotless":  return PilotlessNorm()
    if mode == "given":
        if cfg.algo.gain_norm.G_given is None:
            raise ValueError("gain_norm.mode='given' requires G_given")
        return GivenNorm(cfg.algo.gain_norm.G_given)
    if mode == "pathloss":   return PathlossNorm(include_M=cfg.algo.gain_norm.use_expected_M_factor)
    raise ValueError("unknown gain_norm.mode")

# Small CNN for MNIST (21840 parameters)

def make_mnist_cnn(device: str = "cpu") -> nn.Module:
    """
    Small CNN matching d = 21840 parameters (Tegin & Duman style).
    Input: 1×28×28
    """
    model = nn.Sequential(
        nn.Conv2d(1, 10, kernel_size=5),  # 1×28×28 -> 10×24×24
        nn.ReLU(),
        nn.MaxPool2d(2),                  # 10×12×12

        nn.Conv2d(10, 20, kernel_size=5), # 20×8×8
        nn.ReLU(),
        nn.MaxPool2d(2),                  # 20×4×4

        nn.Flatten(),                     # 20*4*4 = 320
        nn.Linear(320, 50),
        nn.ReLU(),
        nn.Linear(50, 10),
    )
    return model.to(device)

def make_mnist_mlp(device: str = "cpu", hidden: int = 100) -> nn.Module:
    """
    MLP: Flatten(784) -> FC(hidden) -> ReLU -> FC(10)
    If hidden=100, params = 784*100+100 + 100*10+10 = 79,510.
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 10),
    )
    return model.to(device)


def make_mnist_mlp2(device: str = "cpu", h1: int = 128, h2: int = 64) -> nn.Module:
    """A slightly larger MLP variant."""
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, h1), nn.ReLU(),
        nn.Linear(h1, h2), nn.ReLU(),
        nn.Linear(h2, 10),
    )
    return model.to(device)

MNIST_MODEL_REGISTRY: dict[str, callable] = {
    "cnn_small": make_mnist_cnn,
    "mlp_100":   lambda device="cpu": make_mnist_mlp(device=device, hidden=100),
    "mlp2":      lambda device="cpu": make_mnist_mlp2(device=device, h1=128, h2=64),
}


def make_mnist_model(model_arch: str, device: str = "cpu") -> nn.Module:
    key = (model_arch or "cnn_small").lower()
    if key not in MNIST_MODEL_REGISTRY:
        raise ValueError(f"Unknown MNIST model_arch={model_arch!r}. "
                         f"Available: {sorted(MNIST_MODEL_REGISTRY.keys())}")
    return MNIST_MODEL_REGISTRY[key](device=device)


def build_system_mnist(K, cfg, rng, device="cpu", partition="iid",
                       alpha=0.5, shards_per_client=2, n_classes_per_client=2,
                       qty_alpha=0.3, maj_min=0.8, maj_max=0.95,
                       ch_longterm="lognormal", pl_exp=3.0, dmin=10.0, dmax=200.0,
                       shadow_std_db=6.0, dist_file=None, 
                           # --- new: per-round client learning-rate schedule decided by server ---
                       local_lr0: float = 0.05,
                       local_lr_alpha: float = 0.5,
                       local_lr_t0: float = 1.0,
                       dataset_name: str = "mnist",   # <--- NEW
                       batch_size: int = 64,
                       model_arch: str = "cnn_small", 
                       ):
    
    # --- Data selection: MNIST vs Fashion-MNIST ---
    if dataset_name.lower() == "mnist":
        ds_class = datasets.MNIST
        mean, std = (0.1307,), (0.3081,)
    elif dataset_name.lower() == "fashion":
        ds_class = datasets.FashionMNIST
        # Standard normalization used in the literature; not critical, but nicer
        mean, std = (0.2860,), (0.3530,)
    else:
        raise ValueError(f"Unknown dataset_name={dataset_name!r}; expected 'mnist' or 'fashion'.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_ds = ds_class(root="~/.torch", train=True,  download=True, transform=transform)
    test_ds  = ds_class(root="~/.torch", train=False, download=True, transform=transform)

   
    # --- choose partition ---

    y_train = np.array(train_ds.targets)

    if partition == "iid":
        parts = iid_splits(num_samples=len(train_ds), K=K, rng=rng)
    elif partition == "dirichlet":
        parts = dirichlet_splits(labels=y_train, K=K, alpha=alpha, rng=rng, min_size=1)
    elif partition == "shards":
        parts = shards_partition(labels=y_train, K=K, shards_per_client=shards_per_client, rng=rng)
    elif partition == "qty":
        parts = quantity_skew(num_samples=len(train_ds), K=K, rng=rng, alpha=(qty_alpha or 0.3), min_size=10)
    elif partition == "nclass":
        parts = label_skew_n_classes(labels=y_train, K=K, n_classes_per_client=(n_classes_per_client or 2), rng=rng)
    elif partition == "majority":
        parts = majority_class_skew(labels=y_train, K=K, rng=rng,
                                    p_min=(maj_min or 0.8), p_max=(maj_max or 0.95))
    else:
        raise ValueError(f"unknown partition: {partition}")


    # Model: small MLP
    #model = nn.Sequential(
    #    nn.Flatten(),
    #    nn.Linear(28*28, 128), nn.ReLU(),
    #    nn.Linear(128, 64), nn.ReLU(),
    #    nn.Linear(64, 10),
    #).to(device)
    
     # NEW: small CNN (Tegin & Duman style)
    # Model: small CNN (Tegin & Duman style)
    model = make_mnist_model(model_arch=model_arch, device=device)
    adapter = TorchModelAdapter(model, device=device)
    w0 = adapter.to_vector()

    # Build clients
    clients = []
    nks = []
    # Long-term channel powers (log-normal-ish spread)
    if ch_longterm == "lognormal":
        Omega_vec = np.exp(rng.normal(loc=0.0, scale=0.8, size=K)).astype(float)
    elif ch_longterm == "pathloss":
        if dist_file:
            dists = np.loadtxt(dist_file, delimiter=",").astype(float)
            assert dists.shape[0] >= K, "dist_file must have at least K distances"
            dists = dists[:K]
        else:
            dists = rng.uniform(dmin, dmax, size=K)
        K0 = 1.0
        shadow_lin = 10 ** (rng.normal(0.0, shadow_std_db, size=K) / 10.0)
        Omega_vec = K0 * (np.maximum(dists, 1.0) ** (-pl_exp)) * shadow_lin
    else:
        raise ValueError("ch_longterm must be 'lognormal' or 'pathloss'")


    for k in range(K):

        idxs = parts[k]
    # Each client gets the SAME architecture as the global model
        model_k = make_mnist_model(model_arch=model_arch, device=device)
        adapter_k = TorchModelAdapter(model_k, device=device)
        # one-time batch size choice (not per round)
        n_local = len(idxs)
        bs_k = min(int(batch_size), n_local) if n_local > 0 else int(batch_size)

        trainer = TorchTrainer(
            model=model_k,
            adapter=adapter_k,
            dataset=train_ds,
            idxs=idxs,
            batch_size=bs_k,
            lr=0.05,
            device=device,
        )
        cstate = ClientState(id=k, n_k=int(len(idxs)), Omega_k=float(Omega_vec[k]))
        clients.append(Client(cstate, trainer)); nks.append(len(idxs))

    # Noise
    if cfg.algo.noise.mode == "none":    sigma_w2 = 0.0
    elif cfg.algo.noise.mode == "given": sigma_w2 = cfg.radio.sigma_w2_given or 0.0
    elif cfg.algo.noise.mode == "awgn":  sigma_w2 = cfg.algo.noise.sigma_w2
    else: raise ValueError("noise.mode must be one of {'none','awgn','given'}")

    gain_norm = make_gain_norm(cfg)
    M = getattr(cfg.radio, "M", getattr(cfg.radio, "chips", 32))
    comm = Comm(M=M, sigma_w2=sigma_w2, rng=rng, gain_norm=gain_norm)

    server = Server(
        clients=clients,
        comm=comm,
        d=adapter.total,
        Omega_vec=Omega_vec,
        Pk_vec=np.ones(K) * cfg.radio.pt_W,
        cfg=cfg,
        local_lr0=local_lr0,
        local_lr_alpha=local_lr_alpha,
        local_lr_t0=local_lr_t0,
    )
    server.state.w = w0.copy()  # set to model init

    # TX energy per coordinate per client
    #Tcoord_s = cfg.radio.Tcoord_s
    Pk_vec = np.ones(K) * (cfg.radio.pt_W) # Here we use pt_W as per-coordinate energy, not power

    # Test loader & loss
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    return adapter, model, server, Omega_vec, Pk_vec, test_loader, loss_fn


# --- Add this compact CIFAR-10 CNN ---
class SmallCIFARCNN(nn.Module):
    """Lightweight CNN that trains quickly on CIFAR-10."""
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),  # 4x4
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.classifier(self.features(x))


def _make_cifar_model(arch: str, device: str):
    arch = (arch or "cnn3").lower()
    if arch == "cnn3":
        return SmallCIFARCNN().to(device)
    elif arch == "resnet18":
        from torchvision.models import resnet18
        m = resnet18(weights=None, num_classes=10)
        # CIFAR-friendly tweak: 3x3 conv, stride=1, no initial maxpool
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        return m.to(device)
    else:
        raise ValueError(f"Unknown CIFAR model arch: {arch}")


# --- New: CIFAR-10 builder (same return signature as build_system_mnist) ---
def build_system_cifar(
    K: int,
    cfg: Config,
    rng: np.random.Generator,
    device: str = "cpu",
    partition: str = "iid",
    alpha: float = 0.5,
    shards_per_client: int = 2,
    n_classes_per_client: int = 2,
    qty_alpha: float = 0.3,
    maj_min: float = 0.8,
    maj_max: float = 0.95,
    ch_longterm: str = "lognormal",
    pl_exp: float = 3.0,
    dmin: float = 10.0,
    dmax: float = 200.0,
    shadow_std_db: float = 6.0,
    dist_file: str | None = None,
    # server-chosen per-round client LR schedule (aligned with MNIST builder)
    local_lr0: float = 0.05,
    local_lr_alpha: float = 0.5,
    local_lr_t0: float = 1.0,
    # CIFAR specifics
    model_arch: str = "cnn3",
    augment: bool = True,
):
    # --- CIFAR-10 datasets & transforms ---
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2470, 0.2435, 0.2616)

    t_train = []
    if augment:
        t_train += [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()]
    t_train += [transforms.ToTensor(), transforms.Normalize(mean, std)]
    t_test = [transforms.ToTensor(), transforms.Normalize(mean, std)]

    train_ds = datasets.CIFAR10(root="~/.torch", train=True,  download=True, transform=transforms.Compose(t_train))
    test_ds  = datasets.CIFAR10(root="~/.torch", train=False, download=True, transform=transforms.Compose(t_test))

    # --- choose partition ---
    y_train = np.array(train_ds.targets)
    if partition == "iid":
        parts = iid_splits(num_samples=len(train_ds), K=K, rng=rng)
    elif partition == "dirichlet":
        parts = dirichlet_splits(labels=y_train, K=K, alpha=alpha, rng=rng, min_size=1)
    elif partition == "shards":
        parts = shards_partition(labels=y_train, K=K, shards_per_client=shards_per_client, rng=rng)
    elif partition == "qty":
        parts = quantity_skew(num_samples=len(train_ds), K=K, rng=rng, alpha=(qty_alpha or 0.3), min_size=10)
    elif partition == "nclass":
        parts = label_skew_n_classes(labels=y_train, K=K, n_classes_per_client=(n_classes_per_client or 2), rng=rng)
    elif partition == "majority":
        parts = majority_class_skew(labels=y_train, K=K, rng=rng, p_min=(maj_min or 0.8), p_max=(maj_max or 0.95))
    else:
        raise ValueError(f"unknown partition: {partition}")

    # --- Global model & adapter ---
    model = _make_cifar_model(model_arch, device)
    adapter = TorchModelAdapter(model, device=device)
    w0 = adapter.to_vector().copy()
    # --- Build clients ---
    clients: list[Client] = []

    # Long‑term channel powers (same options as MNIST)
    if ch_longterm == "lognormal":
        Omega_vec = np.exp(rng.normal(loc=0.0, scale=0.8, size=K)).astype(float)
    elif ch_longterm == "pathloss":
        if dist_file:
            dists = np.loadtxt(dist_file, delimiter=", ").astype(float)
            assert dists.shape[0] >= K, "dist_file must have at least K distances"
            dists = dists[:K]
        else:
            dists = rng.uniform(dmin, dmax, size=K)
        K0 = 1.0
        shadow_lin = 10 ** (rng.normal(0.0, shadow_std_db, size=K) / 10.0)
        Omega_vec = K0 * (np.maximum(dists, 1.0) ** (-pl_exp)) * shadow_lin
    else:
        raise ValueError("ch_longterm must be 'lognormal' or 'pathloss'")

    for k in range(K):
        idxs = parts[k]
        model_k = _make_cifar_model(model_arch, device)
        adapter_k = TorchModelAdapter(model_k, device=device)
        trainer = TorchTrainer(
            model=model_k,
            adapter=adapter_k,
            dataset=train_ds,
            idxs=idxs,
            batch_size=128,
            lr=0.1,
            device=device,
        )
        cstate = ClientState(id=k, n_k=int(len(idxs)), Omega_k=float(Omega_vec[k]))
        clients.append(Client(cstate, trainer))

    # --- Noise / gain normalization / comm ---
    if cfg.algo.noise.mode == "none":    sigma_w2 = 0.0
    elif cfg.algo.noise.mode == "given": sigma_w2 = cfg.radio.sigma_w2_given or 0.0
    elif cfg.algo.noise.mode == "awgn":  sigma_w2 = cfg.algo.noise.sigma_w2
    else: raise ValueError("noise.mode must be one of {'none','awgn','given'}")

    gain_norm = make_gain_norm(cfg)
    M = getattr(cfg.radio, "M", getattr(cfg.radio, "chips", 32))
    comm = Comm(M=M, sigma_w2=sigma_w2, rng=rng, gain_norm=gain_norm)

    # --- Server ---
    server = Server(
        clients=clients,
        comm=comm,
        d=adapter.total,
        Omega_vec=Omega_vec,
        Pk_vec=np.ones(K) * cfg.radio.pt_W,
        cfg=cfg,
        local_lr0=local_lr0,
        local_lr_alpha=local_lr_alpha,
        local_lr_t0=local_lr_t0,
    )
    server.state.w = w0.copy()

    # TX energy per coordinate per client (reuse pt_W semantics)
    Pk_vec = np.ones(K) * (cfg.radio.pt_W)

    # --- Test loader & loss ---
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    loss_fn = nn.CrossEntropyLoss()

    return adapter, model, server, Omega_vec, Pk_vec, test_loader, loss_fn


# Optional convenience dispatcher (non-breaking):
# Old usages calling build_system_mnist(...) keep working.
# New code can call build_system(dataset=..., ...)
def build_system(*args, dataset: str = "mnist", **kwargs):
    ds = (dataset or "mnist").lower()
    if ds == "mnist":
        return build_system_mnist(*args, **kwargs)
    elif ds in ("cifar10", "cifar"):
        return build_system_cifar(*args, **kwargs)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")