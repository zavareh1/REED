import numpy as np

def apply_clipping(u: np.ndarray, mode: str, B: float, L2_max: float) -> np.ndarray:
    if mode == "none":
        return u
    if mode == "per_coord":
        return np.clip(u, -B, B)
    if mode == "l2":
        norm = np.linalg.norm(u)
        if norm <= L2_max or norm == 0.0:
            return u
        return u * (L2_max / norm)
    raise ValueError("clipping mode must be one of {'none','per_coord','l2'}")
