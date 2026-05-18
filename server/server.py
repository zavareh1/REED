from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from clients.client import Client
from comm.comm import Comm

from config import Config

@dataclass
class ServerState:
    w: np.ndarray        # global model params (vectorized)
    t: int = 0

@dataclass
class RoundBatch:
    t: int
    selected: list[int]
    eta_t: float                 # client LR actually used this round
    U_list: list[np.ndarray]     # weighted client updates u_k = w_k * Δ_k
    U_sum: np.ndarray            # sum_k u_k
    K_act: int
    
 

class Server:
    def __init__(self,
                 clients: List[Client],
                 comm: Comm,
                 d: int,
                 Omega_vec: np.ndarray,
                 Pk_vec:    np.ndarray,
                 cfg: Config,
                 agg_mode: str = "fedavg",
                 e0_pilot:  float | None = None,
                 # --- per-round client LR chosen by the server (used by trainers) ---
                 local_lr0: float = 0.05,
                 local_lr_alpha: float = 0.6,   # lr_t = lr0 / (t + t0)^alpha alpha = 0 means constant LR
                 local_lr_t0: float = 0.0,
                 # --- server-side SGD step (only if agg_mode == "sgd") ---
                 sgd_eta0: float = 0.1,
                 sgd_t0: float = 10.0,
                 sgd_exp: float = 0.60,
                 clip_mode: str = "none",
                 clip_B: float = 0.5,
                 clip_L2: float = 1.0,):
        self.clients = clients
        self.comm = comm
        self.cfg = cfg

        # Server state
        self.state = ServerState(w=np.zeros(d))
        # Agg mode: how the server uses the aggregate (FedAvg with model delta vs SGD-on-gradient)
        agg_mode = getattr(cfg.algo, "agg_mode", "fedavg")
        if agg_mode not in ("fedavg", "sgd"):
            raise ValueError("agg_mode must be 'fedavg' or 'sgd'")
        self.agg_mode = agg_mode
        
        # client LR schedule
        self.local_lr0 = float(local_lr0)
        self.local_lr_alpha = float(local_lr_alpha)
        self.local_lr_t0 = float(local_lr_t0)

        # radio / long-term channel + per-coordinate TX energy (a.k.a. "Pk")
        self.Omega_vec = np.asarray(Omega_vec, dtype=np.float64)  # shape (Ktotal,)
        self.Pk_vec    = np.asarray(Pk_vec,    dtype=np.float64)  # shape (Ktotal,)

        # Gain normalization: pilot / pilotless / given / pathloss, etc.
        # e0_pilot is only meaningful if we use pilot gain normalization.
        gn_cfg = cfg.algo.gain_norm
        self.e0_pilot = (
            float(getattr(gn_cfg, "e0_pilot", 1.0))
            if getattr(gn_cfg, "mode", "none") == "pilot"
            else None
        )

        # Server-side SGD schedule (only used if agg_mode == "sgd")
        self.eta0 = float(sgd_eta0)
        self.t0 = float(sgd_t0)
        self.exp = float(sgd_exp)
        self.clip_mode = str(clip_mode)
        self.clip_B    = float(clip_B)
        self.clip_L2   = float(clip_L2)

        # CSIT selection config (optional in cfg)
        csit_cfg = getattr(cfg.algo, "csit", None)

        if csit_cfg is not None:
            self.csit_sel_thresh_by = str(getattr(csit_cfg, "thresh_by", "snr"))
            self.csit_sel_thresh = getattr(csit_cfg, "thresh", None)
            self.csit_sel_keep_min = int(getattr(csit_cfg, "keep_min", 1))
            self.csit_sel_equal_gain = bool(getattr(csit_cfg, "equal_gain", True))
            self.csit_sel_norm_mode = str(getattr(csit_cfg, "norm_mode", "meancoeff"))
        else:
            # Fallbacks to previous hard-coded defaults
            self.csit_sel_thresh_by = "snr"      # {"snr", "hnorm", "beta"}
            self.csit_sel_thresh = None          # float or None
            self.csit_sel_keep_min = 1           # >=1
            self.csit_sel_equal_gain = True      # fairness among kept
            self.csit_sel_norm_mode = "meancoeff"        
        self.csit_sel_norm_mode  = "meancoeff"
        
        self._agg_dispatch = {
            "clean":       self.aggregate_clean,
            "ota_reed":    self.aggregate_ota_reed,
            "CSIT_SELECT": self.aggregate_csit_select,
        }

    def _sgd_lr(self, t: int) -> float:
        return self.eta0 / (t + self.t0) ** self.exp

    def _round_local_lr(self, t: int) -> float:
        """Round-wise client LR decided by the server (used inside clients)."""
        tt = max(1.0, self.local_lr_t0) + float(t)
        return self.local_lr0 / (tt ** self.local_lr_alpha)
    
    def set_client_weights(self, selected: List[int] | None = None):
        if selected is None:
            selected = list(range(len(self.clients)))
        N = sum(self.clients[i].s.n_k for i in selected)
        for i in selected:
            c = self.clients[i]
            c.s.weight_k = c.s.n_k / N

    def _collect_round_batch(self,selected: List[int],
                      epochs: int | None = None,
                      steps: int | None = None,
                  )-> RoundBatch:
        t = self.state.t
        eta_t = self._round_local_lr(t)

        self.set_client_weights(selected)

        # 2) collect weighted updates once, with the SAME eta_t
        
        U_list = []
        for i in selected:
            c = self.clients[i]
            # NOTE: pass lr_override=eta_t so clean/OTA use identical local LR
            u_k, mu_k = c.form_update(
                w_global=self.state.w,
                clip_mode=self.clip_mode,
                clip_B=self.clip_B,
                clip_L2=self.clip_L2,
                epochs=epochs,
                steps=steps,
                mode="fedavg",
                round_lr=eta_t,
            )
            # drop non-finite client update defensively
            if not np.all(np.isfinite(u_k)):
                continue
            U_list.append(u_k)
            
           
        

        if len(U_list) == 0:
            # fall back safely
            U_sum = np.zeros_like(self.state.w)
            return RoundBatch(t, selected=[], eta_t=eta_t, U_list=[], U_sum=U_sum, K_act=0)

        U_sum = np.sum(np.stack(U_list, axis=0), axis=0)
        

        return RoundBatch(t, selected, eta_t, U_list, U_sum, K_act=len(U_list))
    
    
    def aggregate_clean(self, batch: RoundBatch) -> np.ndarray:
        # Clean FedAvg update is just the sum (clients already weighted)
        return batch.U_sum           

    def aggregate_ota_reed(self, batch: RoundBatch) -> np.ndarray:
        # Use the SAME U_list / selected as clean, only the PHY differs
        # NOTE: keep your previously working normalization (G* pilotless) here.
        U_list=batch.U_list,
        selected=batch.selected,
        Omega_sel = self.Omega_vec[batch.selected]
        Pk_sel    = self.Pk_vec[batch.selected]
        
        s_hat = self.comm.aggregate_reed_vectorized(
            U_list=batch.U_list,
            Omega_vec=Omega_sel,
            Pk_vec=Pk_sel,
            e0_pilot=self.e0_pilot,
        )

        return s_hat


    def one_round(
        self,
        selected: list[int],
        t: int,
        epochs: int | None = None,
        steps: int | None = None,
        mode: str = "ota_reed",
    ) -> dict:
        self.state.t = t
        batch = self._collect_round_batch(selected, epochs=epochs, steps=steps)
    # Dispatcher: build only what's needed
        try:
            agg_fn = self._agg_dispatch[mode]
        except KeyError:
            raise ValueError(f"unknown aggregation mode: {mode}")
        s_apply = agg_fn(batch)

        # Metrics vs. the same reference U_sum
        eps = 1e-12
        num = float(np.linalg.norm(s_apply))
        den = float(np.linalg.norm(batch.U_sum)) + eps
        scale = num / den
        cos = float(np.dot(s_apply, batch.U_sum) / ((num * den) + eps)) if num > 0.0 else 0.0
        g = scale * cos

        # Apply the chosen update
        self.state.w += s_apply

        out = dict(
            t=batch.t,
            K_act=batch.K_act,
            eta_t=batch.eta_t,
            lr=batch.eta_t,                 # same value, kept for backward compatibility
            scale=scale, cos=cos, g=g,
            norm_sum=float(np.linalg.norm(batch.U_sum)),
            norm_apply=num,                 # <-- correct: norm of the applied aggregate
            mode=mode,
        )
        return out

    def aggregate_csit_select(self, batch: RoundBatch) -> np.ndarray:
       
        Omega_sel = self.Omega_vec[batch.selected]
        Pk_sel    = self.Pk_vec[batch.selected]

        s0, kept_idx, diags = self.comm.aggregate_csit_select_coherent_vectorized(
            U_list=batch.U_list,
            Omega_vec=Omega_sel,
            Pk_vec=Pk_sel,
            thresh_by=self.csit_sel_thresh_by,
            thresh=self.csit_sel_thresh,
            keep_min=self.csit_sel_keep_min,
            equal_gain_among_kept=True,          # recommended to keep direction unbiased
            norm_mode="meancoeff",               # only used if equal_gain_among_kept=False
        )

        # --- post-weighting: renormalize to a kept-set average ---
        # batch.w_vec are the FedAvg weights over the SELECTED set; sum(batch.w_vec) == 1
        sumw_kept = 0.0
        for idx in kept_idx:
            i_client = batch.selected[idx]
            c = self.clients[i_client]
            sumw_kept += c.s.weight_k

        renorm = 1.0 / (sumw_kept + 1e-12)      # equals m_sel/m_keep if weights are equal
        s_hat = s0 * renorm

        # optional diagnostics
        diags.update({
            "sumw_kept": sumw_kept,
            "renorm": renorm,
            "k_act": int(batch.K_act),
        })

        return s_hat