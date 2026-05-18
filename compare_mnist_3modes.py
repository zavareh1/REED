import argparse
import csv
import math
import re
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

# --- Project modules ---
from config import Config
from Models.build_system import build_system_mnist
from server.server import Server
from server.sampling import sample_clients
from server.eval import evaluate_global
from clients.client import Client  # not used directly, but kept for parity


# ----------------------------
# Helpers
# ----------------------------
def _fmt(x):
    if isinstance(x, float):
        s = f"{x:.4g}"
    else:
        s = str(x)
    return s.replace(".", "p").replace("+", "")


def _slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9\-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s


def _parse_modes(s: str) -> list[str]:
    s = (s or "").strip().lower()
    if s in ("all", "", "cmp3"):
        return ["clean", "ota_reed", "CSIT_SELECT"]

    alias = {
        "clean": "clean",
        "fedavg": "clean",
        "ota_reed": "ota_reed",
        "reed": "ota_reed",
        "csit": "CSIT_SELECT",
        "csit_select": "CSIT_SELECT",
    }
    out = []
    for m in s.split(","):
        key = m.strip().lower()
        if not key:
            continue
        out.append(alias.get(key, key))
    return out


def _parse_str_list(s: str | None) -> list[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_float_list(s: str | None) -> list[float]:
    if not s:
        return []
    out = []
    for x in s.split(","):
        x = x.strip()
        if x:
            out.append(float(x))
    return out


def _preselect_clients(T: int, K: int, m: int, seed: int) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [sample_clients(K, m, rng) for _ in range(T)]


def _set_global_seed(seed: int) -> None:
    np.random.seed(seed)
    try:
        import random
        import torch

        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _sigma_w2_from_snr_db(
    pt_W: float,
    snr_db: float,
    ref_power_W: float | None = None,
    dmin: float | None = None,
    pl_exp: float | None = None,
) -> float:
    """
    Convert SNR(dB) to sigma_w2.

    Default convention in this script:
        SNR = pt_W / ((dmin ** pl_exp) * sigma_w2)
    so that
        sigma_w2 = (pt_W / (dmin ** pl_exp)) / 10^(SNR/10).

    This matches the user's current experimental setup when dmin=dmax and
    shadow_std_db=0. If you need a different SNR reference, pass
    --snr_ref_power_W explicitly.
    """
    if ref_power_W is None:
        if dmin is None or pl_exp is None:
            raise ValueError(
                "Need dmin and pl_exp for SNR->sigma_w2 conversion unless "
                "snr_ref_power_W is provided."
            )
        p_ref = pt_W / (float(dmin) ** float(pl_exp))
    else:
        p_ref = ref_power_W
    return float(p_ref / (10.0 ** (snr_db / 10.0)))


def make_run_tag(args, exp_reed: dict, exp_csit: dict, scheme_label: str = "cmp3", max_len: int = 80) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")

    parts = [
        scheme_label,
        f"ds-{getattr(args, 'dataset', 'mnist')}",
        f"K{_fmt(args.clients)}",
        f"m{_fmt(args.sample)}",
        f"E{_fmt(args.local_epochs)}",
        f"lr{_fmt(args.local_lr0)}",
        f"a{_fmt(args.local_lr_alpha)}",
        f"part-{args.partition}",
    ]

    if getattr(args, "partition", None) == "dirichlet":
        parts.append(f"alpha{_fmt(getattr(args, 'alpha', 0.0))}")

    if getattr(args, "comm_q", None) is not None:
        parts.append(f"q{_fmt(args.comm_q)}")

    if getattr(args, "snr_db", None) is not None:
        parts.append(f"snr{_fmt(args.snr_db)}dB")

    parts.extend(
        [
            f"clip-{_slug(exp_reed.get('clipping', 'none'))}",
            f"gn-{_slug(exp_reed.get('gain_norm', 'none'))}",
            f"noise-{_slug(exp_reed.get('noise', 'none'))}",
        ]
    )
    if exp_reed.get("clip_L2"):
        parts.append(f"cL2{_fmt(exp_reed['clip_L2'])}")
    if exp_reed.get("clip_B"):
        parts.append(f"cB{_fmt(exp_reed['clip_B'])}")
    if exp_reed.get("sigma_w2") is not None:
        parts.append(f"sW2{_fmt(exp_reed['sigma_w2'])}")

    if exp_csit.get("thresh") is not None:
        parts.append(f"csit-{_slug(exp_csit.get('thresh_by', 'snr'))}-{_fmt(exp_csit['thresh'])}")
    if exp_csit.get("keep_min"):
        parts.append(f"keep{int(exp_csit['keep_min'])}")
    if exp_csit.get("equal_gain") is not None:
        parts.append("eg1" if exp_csit["equal_gain"] else "eg0")

    if getattr(args, "seed", None) is not None:
        parts.append(f"seed{args.seed}")

    tag = _slug("-".join(str(p) for p in parts if p not in (None, "", "-")))[:max_len]
    return f"{tag}-{ts}"


# ----------------------------
# One full run for a single mode
# ----------------------------
def run_one_mode(
    label: str,
    mode: str,
    args,
    cfg,
    rng_build,
    selections: list[list[int]],
    w0_ref: np.ndarray | None,
    share_eval_loader: tuple | None,
    rep: int,
    rep_seed: int,
):
    adapter, model, server, Omega_vec, Pk_vec, test_loader, loss_fn = build_system_mnist(
        K=args.clients,
        cfg=cfg,
        rng=rng_build,
        device="cpu",
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
        local_lr0=args.local_lr0,
        local_lr_alpha=args.local_lr_alpha,
        local_lr_t0=args.local_lr_t0,
        dataset_name=args.dataset,
        model_arch=args.model_arch,
        batch_size=args.batch_size,
    )

    # lock initial weights across modes
    w0 = adapter.to_vector().copy()
    if w0_ref is not None:
        server.state.w = w0_ref.copy()
    else:
        w0_ref = w0.copy()

    if share_eval_loader is not None:
        test_loader, loss_fn = share_eval_loader

    rows: list[dict] = []

    # evaluate initial model BEFORE any FL rounds -> round = 0
    adapter.from_vector(server.state.w)
    metrics0 = evaluate_global(adapter, model, test_loader, loss_fn)
    row0 = dict(
        rep=rep,
        rep_seed=rep_seed,
        scheme=label,
        mode=mode,
        round=0,
        lr=0.0,
        scale=math.nan,
        cos=math.nan,
        g=math.nan,
        test_loss=float(metrics0.get("loss", math.nan)),
        test_acc=float(metrics0.get("acc", math.nan)),
    )
    rows.append(row0)
    print(f"[{label} r00] init: loss={row0['test_loss']:.4f} acc={row0['test_acc']:.3f}")

    T = int(args.rounds)
    train_kwargs = {"epochs": args.local_epochs} if args.comm_q is None else {"steps": args.comm_q}

    for t in range(T):
        selected = selections[t]
        stats = server.one_round(selected=selected, t=t, mode=mode, **train_kwargs)

        adapter.from_vector(server.state.w)
        metrics = evaluate_global(adapter, model, test_loader, loss_fn)
        r_disp = t + 1

        lr_val = float(
            stats["lr"] if ("lr" in stats and stats["lr"] is not None) else stats.get("eta_t", float("nan"))
        )
        row = dict(
            rep=rep,
            rep_seed=rep_seed,
            scheme=label,
            mode=mode,
            round=r_disp,
            lr=lr_val,
            scale=float(stats.get("scale", math.nan)),
            cos=float(stats.get("cos", math.nan)),
            g=float(stats.get("g", math.nan)),
            test_loss=float(metrics.get("loss", math.nan)),
            test_acc=float(metrics.get("acc", math.nan)),
        )
        if "norm_sum" in stats:
            row["norm_sum"] = float(stats["norm_sum"])
        if "norm_apply" in stats:
            row["norm_apply"] = float(stats["norm_apply"])
        rows.append(row)

        print(
            f"[{label} r{t:02d}] scale={row['scale']:.4f} cos={row['cos']:.4f} g={row['g']:.4f} "
            f"lr={row['lr']:.4f} loss={row['test_loss']:.4f} acc={row['test_acc']:.3f}"
        )

    return rows, w0_ref, (test_loader, loss_fn)


# ----------------------------
# Config builders
# ----------------------------
def _build_exp_dicts(args):
    exp_common = {
        "rounds": args.rounds,
        "clients": args.clients,
        "sample_m": args.sample,
        "local_epochs": args.local_epochs,
        "comm_q": args.comm_q,
        "pt_W": args.pt_W,
        "chips": args.chips,
        "partition": args.partition,
        "shards_per_client": args.shards_per_client,
        "alpha": args.alpha,
        "n_classes_per_client": args.n_classes_per_client,
        "qty_alpha": args.qty_alpha,
        "maj_min": args.maj_min,
        "maj_max": args.maj_max,
        "ch_longterm": args.ch_longterm,
        "pl_exp": args.pl_exp,
        "dmin": args.dmin,
        "dmax": args.dmax,
        "shadow_std_db": args.shadow_std_db,
        "dist_file": args.dist_file,
        "local_lr0": args.local_lr0,
        "local_lr_alpha": args.local_lr_alpha,
        "local_lr_t0": args.local_lr_t0,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "model_arch": args.model_arch,
        "dataset": args.dataset,
        "repeats": args.repeats,
        "snr_db": getattr(args, "snr_db", None),
        "sigma_w2": args.sigma_w2,
    }
    exp_reed = {
        "clipping": args.clipping,
        "clip_B": args.clip_B,
        "clip_L2": args.clip_L2,
        "gain_norm": args.gain_norm,
        "G_given": args.G_given,
        "noise": args.noise,
        "sigma_w2": args.sigma_w2,
    }
    exp_csit = {
        "thresh_by": args.csit_thresh_by,
        "thresh": args.csit_thresh,
        "keep_min": args.csit_keep_min,
        "equal_gain": bool(args.csit_equal_gain),
        "norm_mode": args.csit_norm_mode,
        **exp_reed,
    }
    return exp_common, exp_reed, exp_csit


def _cfg_from_args_ota_reed(args):
    cfg = Config()
    cfg.sim.rounds = args.rounds
    cfg.sim.clients = args.clients
    cfg.sim.sample_m = args.sample
    cfg.sim.local_epochs = args.local_epochs

    cfg.algo.agg_mode = "fedavg"

    cfg.algo.clipping.mode = args.clipping
    cfg.algo.clipping.B = args.clip_B
    cfg.algo.clipping.L2_max = args.clip_L2

    cfg.algo.gain_norm.mode = args.gain_norm
    cfg.algo.gain_norm.G_given = args.G_given

    cfg.algo.noise.mode = args.noise
    cfg.algo.noise.sigma_w2 = args.sigma_w2

    cfg.radio.pt_W = args.pt_W
    if hasattr(cfg.radio, "chips"):
        cfg.radio.chips = args.chips
    else:
        cfg.radio.M = args.chips

    return cfg


def _cfg_from_args_clean(args):
    cfg = Config()
    cfg.sim.rounds = args.rounds
    cfg.sim.clients = args.clients
    cfg.sim.sample_m = args.sample
    cfg.sim.local_epochs = args.local_epochs

    cfg.algo.agg_mode = "fedavg"

    cfg.algo.clipping.mode = "none"
    cfg.algo.clipping.B = 0.0
    cfg.algo.clipping.L2_max = 0.0
    cfg.algo.gain_norm.mode = "none"
    cfg.algo.noise.mode = "none"
    cfg.algo.noise.sigma_w2 = 0.0

    cfg.radio.pt_W = args.pt_W
    if hasattr(cfg.radio, "chips"):
        cfg.radio.chips = args.chips
    else:
        cfg.radio.M = args.chips

    return cfg


def _cfg_from_args_ota_csit(args):
    cfg = _cfg_from_args_ota_reed(args)

    csit = getattr(cfg.algo, "csit", None)
    if csit is not None:
        csit.thresh_by = args.csit_thresh_by
        csit.thresh = args.csit_thresh
        csit.keep_min = args.csit_keep_min
        csit.equal_gain = bool(args.csit_equal_gain)
        csit.norm_mode = args.csit_norm_mode

    cfg.algo.csit_thresh_by = args.csit_thresh_by
    cfg.algo.csit_thresh = args.csit_thresh
    cfg.algo.csit_keep_min = args.csit_keep_min
    cfg.algo.csit_equal_gain = bool(args.csit_equal_gain)
    cfg.algo.csit_norm_mode = args.csit_norm_mode

    return cfg


# ----------------------------
# Summaries / outputs
# ----------------------------
def summarize(rows, scheme_name):
    acc_by_round = defaultdict(list)
    loss_by_round = defaultdict(list)

    for r in rows:
        if r["scheme"] != scheme_name:
            continue
        acc_by_round[int(r["round"])] .append(float(r["test_acc"]))
        loss_by_round[int(r["round"])] .append(float(r["test_loss"]))

    rounds = sorted(acc_by_round.keys())
    out = []
    for rd in rounds:
        accs = np.array(acc_by_round[rd], dtype=float)
        losses = np.array(loss_by_round[rd], dtype=float)
        out.append(
            dict(
                scheme=scheme_name,
                round=rd,
                n=len(accs),
                acc_mean=float(np.mean(accs)),
                acc_std=float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0,
                loss_mean=float(np.mean(losses)),
                loss_std=float(np.std(losses, ddof=1)) if len(losses) > 1 else 0.0,
            )
        )
    return out


def _summary_map(summary_rows, scheme):
    rs = [r for r in summary_rows if r["scheme"] == scheme]
    xs = np.array([r["round"] for r in rs], dtype=int)
    mu = np.array([r["acc_mean"] for r in rs], dtype=float)
    sd = np.array([r["acc_std"] for r in rs], dtype=float)
    n = np.array([r["n"] for r in rs], dtype=int)
    return xs, mu, sd, n


def _write_single_outputs(args, rows_all, summary_rows, exp_common, exp_reed, exp_csit):
    run_tag = _slug(args.tag) if args.tag else make_run_tag(args, exp_reed, exp_csit)

    base_csv = Path(args.out_csv)
    base_csv.parent.mkdir(parents=True, exist_ok=True)
    out_csv = base_csv.with_name(f"{base_csv.stem}_{run_tag}{base_csv.suffix}")

    metric_cols = [
        "rep",
        "rep_seed",
        "scheme",
        "mode",
        "round",
        "lr",
        "scale",
        "cos",
        "g",
        "test_loss",
        "test_acc",
        "norm_sum",
        "norm_apply",
    ]
    param_cols_common = list(exp_common.keys())
    param_cols_reed = [f"reed:{k}" for k in exp_reed.keys()]
    param_cols_csit = [f"csit:{k}" for k in exp_csit.keys()]
    fieldnames = param_cols_common + param_cols_reed + param_cols_csit + metric_cols

    exp_clean = {
        k: ("none" if "noise" in k or "gain_norm" in k else (0.0 if "clip" in k else None))
        for k in exp_reed.keys()
    }
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_all:
            if r["scheme"] == "CLEAN":
                reed_params = exp_clean
                csit_params = {**exp_csit}
            elif r["scheme"] == "OTA_REED":
                reed_params = exp_reed
                csit_params = {**exp_csit}
            else:
                reed_params = exp_reed
                csit_params = {**exp_csit}
            w.writerow(
                {
                    **exp_common,
                    **{f"reed:{k}": reed_params.get(k) for k in exp_reed.keys()},
                    **{f"csit:{k}": csit_params.get(k) for k in exp_csit.keys()},
                    **r,
                }
            )
    print(f"[OK] Wrote CSV to {out_csv}")

    out_csv_summary = out_csv.with_name(f"{out_csv.stem}_SUMMARY{out_csv.suffix}")
    with out_csv_summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scheme", "round", "n", "acc_mean", "acc_std", "loss_mean", "loss_std"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
    print(f"[OK] Wrote SUMMARY CSV to {out_csv_summary}")

    styles = {
        "OTA_REED": dict(marker="o", linestyle="-", linewidth=2.0, markersize=3),
        "CLEAN": dict(marker="s", linestyle="--", linewidth=2.0, markersize=3),
        "CSIT_SELECT": dict(marker="^", linestyle="-.", linewidth=2.0, markersize=3),
    }
    legend_labels = {
        "OTA_REED": "REED",
        "CLEAN": "FedAvg",
        "CSIT_SELECT": "CSIT",
    }

    plt.figure()
    for scheme in ["OTA_REED", "CLEAN", "CSIT_SELECT"]:
        xs, mu, sd, n = _summary_map(summary_rows, scheme)
        if len(xs) == 0:
            continue
        plt.plot(xs, mu, label=legend_labels[scheme], **styles[scheme])
        if args.repeats > 1 and args.band != "none":
            band = sd / np.sqrt(np.maximum(n, 1)) if args.band == "sem" else sd
            plt.fill_between(xs, mu - band, mu + band, alpha=0.2)

    title_bits = [args.dataset.upper(), args.partition]
    if args.partition == "dirichlet":
        title_bits.append(f"alpha={args.alpha}")
    if getattr(args, "snr_db", None) is not None:
        title_bits.append(f"SNR={args.snr_db:g} dB")
    plt.title(" | ".join(title_bits))
    plt.xlabel("Round")
    plt.ylabel("Test Accuracy")
    plt.grid(True)
    plt.legend(loc="lower right")

    base_png = Path(args.out_png)
    base_png.parent.mkdir(parents=True, exist_ok=True)
    out_png = base_png.with_name(f"{base_png.stem}_{run_tag}{base_png.suffix}")
    plt.savefig(out_png, bbox_inches="tight", dpi=300)
    out_pdf = out_png.with_suffix(".pdf")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"[OK] Wrote plot to {out_png}")
    print(f"[OK] Wrote plot to {out_pdf}")

    return {
        "out_csv": out_csv,
        "out_csv_summary": out_csv_summary,
        "out_png": out_png,
        "out_pdf": out_pdf,
        "run_tag": run_tag,
    }


def _write_sweep_outputs(args, sweep_results, snr_list, partition_list):
    base_csv = Path(args.out_csv)
    base_csv.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base_tag = _slug(args.tag) if args.tag else f"{args.dataset}-sweep-{ts}"

    combined_summary = []
    for res in sweep_results:
        part = res["partition"]
        alpha = res["alpha"]
        snr_db = res["snr_db"]
        sigma_w2 = res["sigma_w2"]
        for row in res["summary_rows"]:
            combined_summary.append(
                {
                    "dataset": args.dataset,
                    "partition": part,
                    "alpha": alpha,
                    "snr_db": snr_db,
                    "sigma_w2": sigma_w2,
                    **row,
                }
            )

    combined_csv = base_csv.with_name(f"{base_csv.stem}_{base_tag}_SWEEP_SUMMARY{base_csv.suffix}")
    with combined_csv.open("w", newline="") as f:
        fieldnames = [
            "dataset",
            "partition",
            "alpha",
            "snr_db",
            "sigma_w2",
            "scheme",
            "round",
            "n",
            "acc_mean",
            "acc_std",
            "loss_mean",
            "loss_std",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in combined_summary:
            w.writerow(row)
    print(f"[OK] Wrote SWEEP SUMMARY CSV to {combined_csv}")

    styles = {
        "OTA_REED": dict(marker="o", linestyle="-", linewidth=2.0, markersize=3),
        "CLEAN": dict(marker="s", linestyle="--", linewidth=2.0, markersize=3),
        "CSIT_SELECT": dict(marker="^", linestyle="-.", linewidth=2.0, markersize=3),
    }
    legend_labels = {
        "OTA_REED": "REED",
        "CLEAN": "FedAvg",
        "CSIT_SELECT": "CSIT",
    }

    n_rows = len(partition_list)
    n_cols = len(snr_list)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3.75 * n_rows), sharex=False, sharey=True)
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for i, part in enumerate(partition_list):
        for j, snr_db in enumerate(snr_list):
            ax = axes[i, j]
            match = None
            for res in sweep_results:
                if res["partition"] == part and float(res["snr_db"]) == float(snr_db):
                    match = res
                    break
            if match is None:
                ax.set_visible(False)
                continue

            for scheme in ["OTA_REED", "CLEAN", "CSIT_SELECT"]:
                xs, mu, sd, n = _summary_map(match["summary_rows"], scheme)
                if len(xs) == 0:
                    continue
                ax.plot(xs, mu, label=legend_labels[scheme], **styles[scheme])
                if args.repeats > 1 and args.band != "none":
                    band = sd / np.sqrt(np.maximum(n, 1)) if args.band == "sem" else sd
                    ax.fill_between(xs, mu - band, mu + band, alpha=0.2)

            part_title = part
            if part == "dirichlet":
                part_title = f"dirichlet α={match['alpha']}"
            ax.set_title(f"{part_title}, SNR={snr_db:g} dB")
            ax.grid(True)
            if i == n_rows - 1:
                ax.set_xlabel("Round")
            if j == 0:
                ax.set_ylabel("Test Accuracy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"{args.dataset.upper()} sweep: partitions × SNR", y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])

    base_png = Path(args.out_png)
    base_png.parent.mkdir(parents=True, exist_ok=True)
    combined_png = base_png.with_name(f"{base_png.stem}_{base_tag}_SWEEP{base_png.suffix}")
    fig.savefig(combined_png, bbox_inches="tight", dpi=300)
    combined_pdf = combined_png.with_suffix(".pdf")
    fig.savefig(combined_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Wrote SWEEP plot to {combined_png}")
    print(f"[OK] Wrote SWEEP plot to {combined_pdf}")

    return {"combined_csv": combined_csv, "combined_png": combined_png, "combined_pdf": combined_pdf}


# ----------------------------
# Single experiment driver
# ----------------------------
def run_experiment(args):
    exp_common, exp_reed, exp_csit = _build_exp_dicts(args)
    cfg_clean = _cfg_from_args_clean(args)
    cfg_reed = _cfg_from_args_ota_reed(args)
    cfg_csit = _cfg_from_args_ota_csit(args)

    _set_global_seed(args.seed)

    modes = _parse_modes(args.modes)
    rows_all = []
    shared_eval = None

    selections_fixed = None
    if args.fixed_selections:
        selections_fixed = _preselect_clients(args.rounds, args.clients, args.sample, seed=args.seed + 999)

    for rep in range(int(args.repeats)):
        rep_seed = int(args.seed) + rep * 10000
        _set_global_seed(rep_seed)

        selections = selections_fixed if selections_fixed is not None else _preselect_clients(
            args.rounds, args.clients, args.sample, seed=rep_seed + 999
        )

        rng_clean = np.random.default_rng(rep_seed)
        rng_reed = np.random.default_rng(rep_seed)
        rng_csit = np.random.default_rng(rep_seed)

        w0_ref = None
        shared_eval_rep = shared_eval

        if "clean" in modes:
            rows_c, w0_ref, shared_eval_rep = run_one_mode(
                label="CLEAN",
                mode="clean",
                args=args,
                cfg=cfg_clean,
                rng_build=rng_clean,
                selections=selections,
                w0_ref=w0_ref,
                share_eval_loader=shared_eval_rep,
                rep=rep,
                rep_seed=rep_seed,
            )
            rows_all.extend(rows_c)

        if "ota_reed" in modes:
            rows_r, w0_ref, shared_eval_rep = run_one_mode(
                label="OTA_REED",
                mode="ota_reed",
                args=args,
                cfg=cfg_reed,
                rng_build=rng_reed,
                selections=selections,
                w0_ref=w0_ref,
                share_eval_loader=shared_eval_rep,
                rep=rep,
                rep_seed=rep_seed,
            )
            rows_all.extend(rows_r)

        if "CSIT_SELECT" in modes:
            rows_s, w0_ref, shared_eval_rep = run_one_mode(
                label="CSIT_SELECT",
                mode="CSIT_SELECT",
                args=args,
                cfg=cfg_csit,
                rng_build=rng_csit,
                selections=selections,
                w0_ref=w0_ref,
                share_eval_loader=shared_eval_rep,
                rep=rep,
                rep_seed=rep_seed,
            )
            rows_all.extend(rows_s)

        if shared_eval is None and shared_eval_rep is not None:
            shared_eval = shared_eval_rep

    summary_rows = []
    for sch in ["CLEAN", "OTA_REED", "CSIT_SELECT"]:
        summary_rows.extend(summarize(rows_all, sch))

    outputs = _write_single_outputs(args, rows_all, summary_rows, exp_common, exp_reed, exp_csit)
    return {
        "rows_all": rows_all,
        "summary_rows": summary_rows,
        "partition": args.partition,
        "alpha": args.alpha,
        "snr_db": getattr(args, "snr_db", None),
        "sigma_w2": args.sigma_w2,
        **outputs,
    }


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        "MNIST/Fashion 3-mode comparison: clean vs ota_reed vs CSIT_SELECT",
        conflict_handler="resolve",
    )
    # Core sim
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--clients", type=int, default=10)
    ap.add_argument("--sample", type=int, default=5)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument(
        "--comm_q",
        type=int,
        default=None,
        help="If set, one FL round = Q minibatches per selected client (instead of --local_epochs epochs).",
    )
    ap.add_argument("--batch_size", type=int, default=64, help="Local client DataLoader batch size.")
    ap.add_argument("--model_arch", type=str, default="cnn_small", choices=["cnn_small", "mlp_100", "mlp2"])

    # Repeats & plotting
    ap.add_argument("--repeats", type=int, default=1, help="Number of independent runs to average.")
    ap.add_argument("--band", type=str, default="std", choices=["std", "sem", "none"])
    ap.add_argument("--fixed_selections", action="store_true", help="Reuse the same client-selection schedule across repeats.")

    # Dataset
    ap.add_argument("--dataset", type=str, choices=["mnist", "fashion"], default="mnist")

    # FedAvg LR schedule (server-dictated)
    ap.add_argument("--local_lr0", type=float, default=0.05)
    ap.add_argument("--local_lr_alpha", type=float, default=0.5)
    ap.add_argument("--local_lr_t0", type=float, default=1.0)

    # Partition & data
    ap.add_argument("--partition", choices=["iid", "shards", "dirichlet"], default="iid")
    ap.add_argument("--shards_per_client", type=int, default=2)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--n_classes_per_client", type=int, default=None)
    ap.add_argument("--qty_alpha", type=float, default=None)
    ap.add_argument("--maj_min", type=float, default=None)
    ap.add_argument("--maj_max", type=float, default=None)

    # Channel long-term stats
    ap.add_argument("--ch_longterm", choices=["lognormal", "pathloss"], default="lognormal")
    ap.add_argument("--pl_exp", type=float, default=3.0)
    ap.add_argument("--dmin", type=float, default=10.0)
    ap.add_argument("--dmax", type=float, default=200.0)
    ap.add_argument("--shadow_std_db", type=float, default=6.0)
    ap.add_argument("--dist_file", type=str, default=None)

    # Radio / PHY
    ap.add_argument("--pt_W", type=float, default=2e-7)
    ap.add_argument("--chips", type=int, default=32)

    # OTA shared knobs
    ap.add_argument("--clipping", type=str, default="per_coord", choices=["none", "per_coord", "l2", "L2"])
    ap.add_argument("--clip_B", type=float, default=0.5)
    ap.add_argument("--clip_L2", type=float, default=1.0)
    ap.add_argument("--gain_norm", type=str, default="pilotless", choices=["none", "pilot", "pilotless", "given", "pathloss"])
    ap.add_argument("--G_given", type=float, default=None)
    ap.add_argument("--noise", type=str, default="awgn", choices=["none", "awgn", "given"])
    ap.add_argument("--sigma_w2", type=float, default=5e-12)

    # SNR helpers / sweep mode
    ap.add_argument("--snr_db", type=float, default=None, help="If set, override sigma_w2 using the SNR helper.")
    ap.add_argument(
        "--snr_db_list",
        type=str,
        default=None,
        help="Comma-separated SNR list in dB for sweep mode, e.g. '0,-10,-20'.",
    )
    ap.add_argument(
        "--sweep_partitions",
        type=str,
        default=None,
        help="Comma-separated partitions for sweep mode, e.g. 'iid,dirichlet'.",
    )
    ap.add_argument(
        "--snr_ref_power_W",
        type=float,
        default=None,
        help="Reference receive power used by the SNR->sigma_w2 conversion. By default, the script uses pt_W/(dmin^pl_exp).",
    )

    # CSIT selection knobs
    ap.add_argument("--csit_thresh_by", type=str, default="snr", choices=["snr", "hnorm", "beta"])
    ap.add_argument("--csit_thresh", type=float, default=None)
    ap.add_argument("--csit_keep_min", type=int, default=1)
    ap.add_argument("--csit_equal_gain", action="store_true")
    ap.add_argument("--csit_norm_mode", type=str, default="meancoeff", choices=["none", "meancoeff", "sumcoeff"])

    # Multi-run control
    ap.add_argument("--modes", type=str, default="all", help="comma list or 'all' for clean,ota_reed,CSIT_SELECT")

    # Repro + outputs
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out_csv", type=str, default="mnist_compare_3modes.csv")
    ap.add_argument("--out_png", type=str, default="mnist_compare_3modes.png")
    ap.add_argument("--tag", type=str, default=None)

    args, unknown = ap.parse_known_args()
    if unknown:
        print("[argparse] Ignoring unknown args:", " ".join(unknown))

    # Single-run SNR convenience
    if args.snr_db is not None:
        args.sigma_w2 = _sigma_w2_from_snr_db(
            args.pt_W, args.snr_db, args.snr_ref_power_W, args.dmin, args.pl_exp
        )
        print(f"[SNR helper] snr_db={args.snr_db:g} -> sigma_w2={args.sigma_w2:.6g}")

    # Sweep mode: run all combinations and make a combined summary/plot.
    snr_list = _parse_float_list(args.snr_db_list)
    partition_list = _parse_str_list(args.sweep_partitions)
    sweep_active = bool(snr_list) or bool(partition_list)

    if not sweep_active:
        run_experiment(args)
        return

    if not snr_list:
        snr_list = [args.snr_db if args.snr_db is not None else None]
    if not partition_list:
        partition_list = [args.partition]

    sweep_results = []
    for part in partition_list:
        for snr_db in snr_list:
            if snr_db is None:
                raise ValueError("Sweep mode requires an explicit SNR list or --snr_db.")

            job_args = deepcopy(args)
            job_args.partition = part
            job_args.snr_db = float(snr_db)
            job_args.sigma_w2 = _sigma_w2_from_snr_db(
                job_args.pt_W, job_args.snr_db, job_args.snr_ref_power_W, job_args.dmin, job_args.pl_exp
            )

            # Make per-case tags readable while still writing a dedicated CSV/PNG for each case.
            base_tag = job_args.tag or "sweep"
            part_tag = part if part != "dirichlet" else f"dirichlet-a{_fmt(job_args.alpha)}"
            job_args.tag = f"{base_tag}-{job_args.dataset}-{part_tag}-snr{_fmt(job_args.snr_db)}dB"

            print("\n" + "=" * 90)
            print(
                f"[SWEEP] dataset={job_args.dataset} partition={job_args.partition} "
                f"alpha={job_args.alpha} snr_db={job_args.snr_db:g} sigma_w2={job_args.sigma_w2:.6g}"
            )
            print("=" * 90)
            sweep_results.append(run_experiment(job_args))

    _write_sweep_outputs(args, sweep_results, snr_list, partition_list)


if __name__ == "__main__":
    main()
