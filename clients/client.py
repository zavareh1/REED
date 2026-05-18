from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from utils.clipping import apply_clipping

@dataclass
class ClientState:
    id: int
    n_k: int
    Omega_k: float
    weight_k: float = 0.0

class BaseTrainer:
    def train(self, w_global: np.ndarray, epochs: int) -> np.ndarray:
        """
        Load w_global into the local model, run local training for `epochs`,
        and return the NEW flattened weight vector w_new (same shape as w_global).
        """
        raise NotImplementedError

    # --- Backwards-compat shim (optional, but handy while migrating) ---
    def compute_delta(self, w_global: np.ndarray, epochs: int) -> np.ndarray:
        """
        Deprecated: prefer `train(...)` and compute delta outside.
        """
        w_new = self.train(w_global, epochs=epochs)
        return w_new - w_global


class Client:
    def __init__(self, state: ClientState, trainer: BaseTrainer):
        self.s = state
        self.trainer = trainer
        self.adapter = trainer.adapter      # <-- expose adapter (fixes AttributeError)


    def form_update(
        self,
        w_global: np.ndarray,
        clip_mode: str,
        clip_B: float,
        clip_L2: float,
        epochs: int | None = None,
        steps: int | None = None,
        mode: str = "fedavg",
        round_lr: float | None = None,   # NEW
    ) -> tuple[np.ndarray, float]:
        # Trainer returns NEW weights now
        self.adapter.from_vector(w_global)          # sync params + BN buffers
        w_new = self.trainer.train(w_global, epochs=epochs, steps=steps, lr_override=round_lr)

        # Delta is computed here (outside the trainer)
        w_vec = self.adapter.to_vector()            # SAME length as w_global
        delta = w_vec  - w_global
        """
        # --- DEBUG START ---
        delta = delta.astype(np.float64, copy=False)
        w_raw_mu   = float(np.mean(np.abs(delta)))
        w_norm     = float(np.linalg.norm(delta)) # L2 norm
        wk         = float(self.s.weight_k)

        u_before   = wk * delta
        mu_before  = float(np.mean(np.abs(u_before)))

        u_k = apply_clipping(u_before, clip_mode, clip_B, clip_L2)
        mu_after   = float(np.mean(np.abs(u_k)))
        u_norm     = float(np.linalg.norm(u_k)) # L2 norm
        
        print(
            f"[DEBUG client {self.s.id}] "
            f"n_k={self.s.n_k} weight_k={wk:.6f} "
            f"||Δw||={w_norm:.3e} μ_raw={w_raw_mu:.3e} "
            f"μ_weighted={mu_before:.3e} ||u||={u_norm:.3e} μ_after_clip={mu_after:.3e} "
            f"clip=({clip_mode}, B={clip_B}, L2={clip_L2})"
        )
        # --- DEBUG END ---
        """
        # Apply weighting and clipping before communication
        
        # u_k =  delta
        u_k = self.s.weight_k * apply_clipping(delta, clip_mode, clip_B, clip_L2)

        # Any scalar you report upstream (unchanged)
        mu_k = float(np.mean(np.abs(u_k)))
        return u_k, mu_k
    
