import argparse, numpy as np, torch
import sys
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
from Models.build_system import build_system_mnist



# add if missing at top of file:
# import argparse
# import numpy as np

def main():
    ap = argparse.ArgumentParser(description="OTA-FL on MNIST", conflict_handler="resolve")

    # --- core sim / algo ---
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--sample", type=int, default=5, help="clients per round")
    ap.add_argument("--local_epochs", type=int, default=1)

    ap.add_argument("--clipping", type=str, default="per_coord",
                    choices=["none", "per_coord", "l2"])
    ap.add_argument("--clip_B", type=float, default=0.5)
    ap.add_argument("--clip_L2", type=float, default=1.0)

    ap.add_argument("--gain_norm", type=str, default="pilotless",
                    choices=["none", "pilot", "pilotless", "given", "pathloss"])
    ap.add_argument("--G_given", type=float, default=None)

    ap.add_argument("--noise", type=str, default="awgn",
                    choices=["none", "awgn", "given"])
    ap.add_argument("--sigma_w2", type=float, default=5e-12)

    # --- radio / waveform ---
    ap.add_argument("--pt_W", type=float, default=2e-7)
    ap.add_argument("--chips", type=int, default=32, help="M chips per coordinate")

    # --- reproducibility ---
    ap.add_argument("--seed", type=int, default=1234)

    # --- channel long-term stats ---
    ap.add_argument("--ch_longterm", type=str, default="lognormal",
                    choices=["lognormal", "pathloss"])
    ap.add_argument("--pl_exp", type=float, default=3.0, help="pathloss exponent")
    ap.add_argument("--dmin", type=float, default=10.0, help="min distance (m)")
    ap.add_argument("--dmax", type=float, default=200.0, help="max distance (m)")
    ap.add_argument("--shadow_std_db", type=float, default=6.0, help="shadowing std (dB)")
    ap.add_argument(
        "--dist-file", "--dist_file",
        dest="dist_file", type=str, default=None, metavar="PATH",
        help="CSV/TXT with one distance per line (meters), length K",
    )

    args, unknown = ap.parse_known_args()
    print("argv:", " ".join(sys.argv))
    print("args.rounds =", args.rounds)
    if unknown:
        print("[argparse] Ignoring unknown args:", " ".join(unknown))

    # trust the CLI for the loop count
    num_rounds = int(args.rounds)

    # --- data partitioning ---
    ap.add_argument("--partition", choices=["iid", "shards", "dirichlet"],
                    default="iid", help="Client data partitioning scheme.")
    ap.add_argument("--shards_per_client", type=int, default=2,
                    help="Used when --partition=shards.")
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="Dirichlet concentration for --partition=dirichlet.")
    # Optional knobs some repos use; safe defaults if your build doesn’t need them:
    ap.add_argument("--n_classes_per_client", type=int, default=None)
    ap.add_argument("--qty_alpha", type=float, default=None)
    ap.add_argument("--maj_min", type=int, default=None)
    ap.add_argument("--maj_max", type=int, default=None)

    # Show unknown args instead of hard-exiting with code 2
    args, unknown = ap.parse_known_args()
    if unknown:
        print("[argparse] Ignoring unknown args:", " ".join(unknown))

    # ---- Build config from args ----
    from config import Config
    cfg = Config()

    # You may have cfg.sim / cfg.algo / cfg.radio sections; map accordingly.
    # Adjust these lines if your dataclass uses slightly different field names.
    cfg.sim.rounds = args.rounds
    cfg.sim.clients = args.clients
    cfg.sim.sample_m = args.sample
    cfg.sim.local_epochs = args.local_epochs

    cfg.algo.clipping.mode = args.clipping
    cfg.algo.clipping.B = args.clip_B
    cfg.algo.clipping.L2_max = args.clip_L2

    cfg.algo.gain_norm.mode = args.gain_norm
    cfg.algo.gain_norm.G_given = args.G_given

    cfg.algo.noise.mode = args.noise
    cfg.algo.noise.sigma_w2 = args.sigma_w2

    cfg.radio.pt_W = args.pt_W
    # If your dataclass uses `chips` (common), prefer this:
    if hasattr(cfg.radio, "chips"):
        cfg.radio.chips = args.chips
    else:
        # fallback if the field is named M in your codebase:
        cfg.radio.M = args.chips

    device = "cpu"
    rng = np.random.default_rng(args.seed)
    print("argv:", " ".join(sys.argv))
    print("args.rounds =", args.rounds)
    print("cfg.sim.rounds =", getattr(cfg.sim, "rounds", None))

    # ---- Build system ----
    adapter, model, server, Omega_vec, Pk_vec, test_loader, loss_fn = build_system_mnist(
        args.clients,
        cfg,
        rng,
        device=device,
        partition=args.partition,
        alpha=args.alpha,
        shards_per_client=args.shards_per_client,
        n_classes_per_client=args.n_classes_per_client,
        qty_alpha=args.qty_alpha,
        maj_min=args.maj_min,
        maj_max=args.maj_max,
        ch_longterm=args.ch_longterm,
        pl_exp=args.pl_exp,
        dmin=args.dmin,
        dmax=args.dmax,
        shadow_std_db=args.shadow_std_db,
        dist_file=args.dist_file,
    )

    # ---- Round loop ----
    print("Running OTA-FL on MNIST...")
    for t in range(cfg.sim.rounds):
        selected = sample_clients(cfg.sim.clients, cfg.sim.sample_m, rng)
        s_hat = server.one_round(
            selected=selected,
            epochs=cfg.sim.local_epochs,
            clip_mode=cfg.algo.clipping.mode,
            clip_B=cfg.algo.clipping.B,
            clip_L2=cfg.algo.clipping.L2_max,
            Omega_vec=Omega_vec,
            Pk_vec=Pk_vec,
            e0_pilot=(cfg.algo.gain_norm.e0_pilot
                      if getattr(cfg.algo.gain_norm, "mode", None) == "pilot"
                      else None),
        )

        # Evaluate
        adapter.from_vector(server.state.w)
        metrics = evaluate_global(adapter, model, test_loader, loss_fn)
        print(
            f"[round {t:02d}] ||ŝ||={np.linalg.norm(s_hat):.3e}  "
           #f"G*={diags.get('G_star', float('nan')):.3e}  "
            f"test_loss={metrics['loss']:.4f} acc={metrics['acc']:.3f}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
