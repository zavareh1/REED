from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Union, Tuple
from comm.gain_norm import BaseGainNormalizer

@dataclass
class ERGResult:
    G_star: float
    a_vec: np.ndarray

class Comm:
    def __init__(self, M: int, sigma_w2: float, rng: np.random.Generator,
                 gain_norm: BaseGainNormalizer | None):
        self.M = M
        self.sigma_w2 = sigma_w2
        self.rng = rng
        self.gain_norm = gain_norm

    def set_sigma_w2(self, sigma_w2: float):
        self.sigma_w2 = float(sigma_w2)

    def erg_power_control(self, mu_vec: np.ndarray, Omega_vec: np.ndarray, Pk_vec: np.ndarray) -> ERGResult:
        safe_mu = np.maximum(mu_vec, 1e-23)
        ratios = Omega_vec * Pk_vec / safe_mu
        G_star = np.min(ratios)
        a_vec = G_star / Omega_vec
        return ERGResult(G_star=float(G_star), a_vec=a_vec)

    def _draw_block_channels(self, K: int, Omega_vec: np.ndarray) -> np.ndarray:
        real = self.rng.normal(scale=np.sqrt(Omega_vec[:, None]/2.0), size=(K, self.M)) 
        imag = self.rng.normal(scale=np.sqrt(Omega_vec[:, None]/2.0), size=(K, self.M))
        return real + 1j * imag

 #   def _const_env_waveform(self, total_energy: np.ndarray) -> np.ndarray:
 #       # total_energy is per-coordinate (per client)
 #       amp = np.sqrt(np.maximum(total_energy, 0.0) / self.M)[:, None]  # (K,1) broadcasts over M
 #       # phases = 2 * np.pi * self.rng.random(size=(total_energy.shape[0], self.M)) # check if just doing a constant phase works. 
 #       phases = 2 * np.pi * self.rng.random(size=(total_energy.shape[0], self.M)) # check if just doing a constant phase works. 
 #       return amp * np.exp(1j * phases)
    
    def _const_env_waveform(self, E):
        # E: (K,) or (K, d)
        M = self.M
        E = np.asarray(E)
        if E.ndim == 1:
            amp = np.sqrt(np.maximum(E, 0.0) / M)[:, None]          # (K, 1)
            phases = 2*np.pi*self.rng.random(size=(E.shape[0], M))  # (K, M)
            #Testing out constant phase, for example, 0 for all cooridnates. It should improve the performance, at least in terms of cosine similarity.
            #phases = 0*np.pi*self.rng.random(size=(E.shape[0], M))  
            return amp * np.exp(1j*phases)                          # (K, M)
        elif E.ndim == 2:
            K, d = E.shape
            amp = np.sqrt(np.maximum(E, 0.0) / M)[..., None]        # (K, d, 1)
            phases = 2*np.pi*self.rng.random(size=(K, d, M))        # (K, d, M)

            #Testing out constant phase, for example, 0 for all cooridnates. It should improve the performance, at least in terms of cosine similarity.
            #phases = 0*np.pi*self.rng.random(size=(K, d, M))        # (K, d, M)
            return amp * np.exp(1j*phases)                          # (K, d, M)
        else:
            raise ValueError("E must have shape (K,) or (K,d)")

    #def _awgn(self, size: int):
    #   if self.sigma_w2 <= 0.0:
    #        return np.zeros(size, dtype=np.complex128)
    #    return (self.rng.normal(scale=np.sqrt(self.sigma_w2/2), size=size)
    #            + 1j*self.rng.normal(scale=np.sqrt(self.sigma_w2/2), size=size))
    
    def _awgn(self, size: Union[int, Tuple[int, ...]], dtype=np.complex128) -> np.ndarray:
        """
        Complex AWGN with total per-sample variance sigma_w2.
        Works with size=(M,), (B, M), (d, M), etc.
        """
        if self.sigma_w2 <= 0.0:
            return np.zeros(size, dtype=dtype)
        std = float(np.sqrt(self.sigma_w2 / 2.0))
        # Match dtype of your waveforms/channels to avoid upcasting surprises
        if dtype == np.complex64:
            r = self.rng.normal(scale=std, size=size).astype(np.float32, copy=False)
            i = self.rng.normal(scale=std, size=size).astype(np.float32, copy=False)
            return r + 1j * i
        else:
            r = self.rng.normal(scale=std, size=size)
            i = self.rng.normal(scale=std, size=size)
            return r + 1j * i

    def aggregate(self,
                  U_list: List[np.ndarray],
                  mu_vec: np.ndarray,
                  Omega_vec: np.ndarray,
                  Pk_vec: np.ndarray,
                  e0_pilot: float | None = None) -> tuple[np.ndarray, dict]:
        K = len(U_list)
        d = U_list[0].shape[0]
        U = np.stack(U_list, axis=0)

        erg = self.erg_power_control(mu_vec, Omega_vec, Pk_vec)
        a = erg.a_vec
        # Should be: G* = min_k Omega_k * P_k / mu_k
        G_star_check = float(np.min(Omega_vec * Pk_vec / mu_vec))
        print(f"[COMM] G*_erg={erg.G_star:.3e}  G*_check={G_star_check:.3e}  relerr={(erg.G_star/G_star_check - 1):.3e}")
        U_pos, U_neg = np.maximum(U, 0.0), np.maximum(-U, 0.0)
        e_plus, e_minus = a[:, None] * U_pos, a[:, None] * U_neg

        s_hat = np.zeros(d, dtype=np.float64)
        h = self._draw_block_channels(K, Omega_vec)  # (K, M) block fading

        if self.gain_norm is None:
            Gnorm = 1.0
        else:
            Gnorm =  erg.G_star 

        for i in range(d):
            
            x_plus = self._const_env_waveform(e_plus[:, i])
            x_minus = self._const_env_waveform(e_minus[:, i])

            y_plus = (h * x_plus).sum(axis=0) + self._awgn(self.M)
            y_minus = (h * x_minus).sum(axis=0) + self._awgn(self.M)

            Delta = float(np.sum(np.abs(y_plus)**2) - np.sum(np.abs(y_minus)**2))

            s_hat[i] = Delta / Gnorm

        diags = dict(G_star=erg.G_star, a=a, mu=mu_vec)
        return s_hat, diags
    
    def aggregate_reed_vectorized(self,
                  U_list: List[np.ndarray],
                  Omega_vec: np.ndarray,
                  Pk_vec: np.ndarray,
                  e0_pilot: float | None = None, 
                  tile = 4096) -> tuple[np.ndarray, dict]:
        K = len(U_list)
        d = U_list[0].shape[0]
        U = np.stack(U_list, axis=0)
        mu_vec = np.mean(np.abs(U), axis=1)                          # (K_act,)

        # REED aggregation
        G_star = float(np.min(Omega_vec * Pk_vec /np.maximum(mu_vec, 1e-12)))
        a = G_star / Omega_vec                                       # (K_act,)
        # Should be: G* = min_k Omega_k * P_k / mu_k
        # G_star_check = float(np.min(Omega_vec * Pk_vec / mu_vec))
        #print(f"[COMM] G*_erg={erg.G_star:.3e}  G*_check={G_star_check:.3e}  relerr={(erg.G_star/G_star_check - 1):.3e}")
        U_pos, U_neg = np.maximum(U, 0.0), np.maximum(-U, 0.0)
        e_plus, e_minus = a[:, None] * U_pos, a[:, None] * U_neg

        s_hat = np.zeros(d, dtype=np.float64)
        h = self._draw_block_channels(K, Omega_vec)  # (K, M) block fading

      
        if self.gain_norm is None:
            Gnorm = 1.0
        else:
            Gnorm = G_star 
        Gnorm = max(Gnorm, 1e-12)  # numeric floor

        for t0 in range(0, d, tile):
            sl = slice(t0, min(t0+tile, d))
            B = sl.stop - sl.start  # block size
        # bypass the const_env waveform function
            xp = self._const_env_waveform(e_plus[:, sl])            # (K, B, M)
            xm = self._const_env_waveform(e_minus[:, sl])           # (K, B, M)
            # amplitudes per client/coord/chip:
            #amp_p = np.sqrt(np.maximum(e_plus[:, sl],  0.0) / self.M)[..., None]   # (K,B,1)
            #amp_m = np.sqrt(np.maximum(e_minus[:, sl], 0.0) / self.M)[..., None]   # (K,B,1)
            # draw phases ONCE and reuse for both + and −:
            #phi   = 2*np.pi*self.rng.random(size=(K, B, self.M))
            #xp    = amp_p * np.exp(1j*phi)                                         # (K,B,M)
            #xm    = amp_m * np.exp(1j*phi)                                         # (K,B,M)

            yp = (h[:, None, :] * xp).sum(axis=0) + self._awgn((B,self.M))
            ym = (h[:, None, :] * xm).sum(axis=0) + self._awgn((B,self.M))
            Delta = np.sum(np.abs(yp)**2 - np.abs(ym)**2, axis=1)   # (B,)
            s_hat[sl] = Delta / Gnorm
        

        # diags = dict(G_star=erg.G_star, a=a, mu=mu_vec)

        if not np.all(np.isfinite(s_hat)):
            U_sum = np.sum(np.stack(U_list, 0), axis=0)
            print("[WARN] non-finite ŝ; falling back to noiseless sum for this round.")
            s_hat = np.nan_to_num(U_sum, nan=0.0, posinf=0.0, neginf=0.0)
        return s_hat

    def aggregate_csit_select_coherent_vectorized(
        self,
        U_list,                     # list[np.ndarray] of length K, each (d,)
        Omega_vec: np.ndarray,      # (K,)
        Pk_vec:    np.ndarray,      # (K,) per-coordinate TX energy budgets
        # --- selection knob (choose exactly one metric) ---
        thresh_by: str = "snr",     # {"snr","hnorm","beta"}
        thresh: float | None = None,# threshold on the chosen metric; None => no drop
        keep_min: int = 1,          # ensure at least this many users are kept
        # --- normalization knob ---
        equal_gain_among_kept: bool = True,  # fairness within active set
        norm_mode: str = "meancoeff",        # used only if equal_gain_among_kept == False; {"none","meancoeff","sumcoeff"}
        eps: float = 1e-12,
    ):
        """
        Coherent CSIT aggregation with (i) metric-based dropping, (ii) keep-min guarantee,
        and (iii) optional equal-gain normalization among the kept users.

        - Per-user energy budget Pk_vec follows the per-coordinate convention (pt_W).
        - Noise is injected as a scalar shortcut post-combiner with std sqrt(M*sigma_w2/2),
        consistent with the AWGN semantics and coherent combining.
        """
   
        K = len(U_list)
        if K == 0:
            return np.zeros_like(U_list[0])
        d = U_list[0].shape[0]
        U = np.stack(U_list, axis=0).astype(np.float64, copy=False)     # (K,d)

        # Per-client second moment (drives energy usage for linear signaling)
        mu2 = np.mean(U * U, axis=1)                                    # (K,)

        # Draw instantaneous channels and norms
        h = self._draw_block_channels(K, Omega_vec)                      # (K,M)
        hnorms = np.linalg.norm(h, axis=1) + eps                          # (K,)

        # “Power-limited” effective beta if we DIDN'T do equal-gain:
        # alpha_pow = sqrt(Pk / mu2), so beta_pow = alpha_pow * ||h||.
        alpha_pow = np.sqrt(np.maximum(Pk_vec, 0.0) / (mu2 + eps))       # (K,)
        beta_pow  = alpha_pow * hnorms                                    # (K,)

        # SNR proxy per client for selection (post-combiner scalar model):
        # SNR_k ≈ (beta_pow^2 * Var[u]) / (M*sigma_w2) = Pk * ||h||^2 / (M*sigma_w2)
        if self.sigma_w2 <= 0.0:
            snr_k = np.full(K, np.inf, dtype=np.float64)
        else:
            snr_k = (Pk_vec * (hnorms**2)) / (max(self.M, 1.0) * float(self.sigma_w2))  # (K,)

        # --- Build the selection metric ---
        if   thresh_by == "snr":   metric = snr_k
        elif thresh_by == "hnorm": metric = hnorms
        elif thresh_by == "beta":  metric = beta_pow
        else:
            raise ValueError("thresh_by must be one of {'snr','hnorm','beta'}")

        # Initial keep mask by threshold (if provided)
        if thresh is None:
            keep = np.ones(K, dtype=bool)
        else:
            keep = metric >= float(thresh)

        # Guarantee at least keep_min users (by strongest metric)
        k_kept = int(np.sum(keep))
        if k_kept < keep_min:
            order = np.argsort(-metric)                       # strongest first
            keep[:] = False
            keep[order[:max(1, keep_min)]] = True

        # If (still) none kept, fall back to best single user
        if not np.any(keep):
            best = int(np.argmax(metric))
            keep[best] = True

        kept_idx = np.nonzero(keep)[0]
        U_kept   = U[kept_idx, :]                             # (Kkept, d)
        norms_k  = hnorms[kept_idx]
        mu2_k    = mu2[kept_idx]
        Pk_k     = Pk_vec[kept_idx]

        # --- Coherent combining ---
        noise_std = float(np.sqrt(self.sigma_w2 * self.M / 2.0)) if self.sigma_w2 > 0.0 else 0.0

        if equal_gain_among_kept:
            # Equal-gain among kept: choose β* s.t. alpha_k^2 * mu2_k <= Pk_k with alpha_k = β*/||h_k||
            # => (β*)^2 <= min_k Pk_k * ||h_k||^2 / mu2_k
            ratios = Pk_k * (norms_k**2) / (mu2_k + eps)
            beta_star = float(np.sqrt(max(eps, np.min(ratios))))

            # y = β* * sum_k u_k + n; unbiased estimate after dividing by β*
            y = beta_star * np.sum(U_kept, axis=0)                         # (d,)
            if noise_std > 0.0:
                y = y + self.rng.normal(scale=noise_std, size=d)
            s_hat = y / max(beta_star, eps)
            # Decide how you want to log "G"
            # Option A: per-user gain
            G = max(beta_star, eps)
            # Option B: total summed gain
            # G = max(beta_star * len(kept_idx), eps)

        else:
            # No equal-gain: use power-limited coefficients β_k = sqrt(Pk/mu2_k) * ||h_k||
            beta_eff = (np.sqrt(np.maximum(Pk_k, 0.0) / (mu2_k + eps)) * norms_k)  # (Kkept,)
            y = (beta_eff[:, None] * U_kept).sum(axis=0)                           # (d,)
            if noise_std > 0.0:
                y = y + self.rng.normal(scale=noise_std, size=d)

            if   norm_mode == "sumcoeff":
                G = float(np.sum(beta_eff)); G = max(G, eps)
                s_hat = (y / G) * len(kept_idx)
            elif norm_mode == "meancoeff":
                G = float(np.mean(beta_eff)); G = max(G, eps)
                s_hat = y / G
            elif norm_mode == "none":
                s_hat = y
                G = 1
            else:
                raise ValueError("norm_mode must be one of {'none','meancoeff','sumcoeff'}")

        diags = {
            "m_sel": int(K),
            "m_keep": int(len(kept_idx)),
            "G": float(G),
            "noise_std": noise_std,
            "mode": f"no_equal_gain/{norm_mode}"
        }

        # Safety
        s0 = s_hat
        if not np.all(np.isfinite(s0)):
            U_sum = np.sum(U_kept, axis=0) if U_kept.size else np.zeros(d, dtype=np.float64)
            s0 = np.nan_to_num(U_sum, nan=0.0, posinf=0.0, neginf=0.0)

        return s0, kept_idx, diags
