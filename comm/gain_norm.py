from __future__ import annotations
import numpy as np

class BaseGainNormalizer:
    def norm_scalar(self, *, h: np.ndarray | None, a: np.ndarray, M: int,
                    sigma_w2: float, rng, e0: float | None,
                    G_star: float, Omega_vec: np.ndarray) -> float:
        raise NotImplementedError

class NoNorm(BaseGainNormalizer):
    def norm_scalar(self, **kwargs) -> float:
        return 1.0

class PilotNorm(BaseGainNormalizer):
    def norm_scalar(self, *, h, a, M, sigma_w2, rng, e0, **_):
        amp = np.sqrt(e0 / M)
        phases = 2*np.pi*rng.random(size=(h.shape[0], M))
        x_ref = amp * np.exp(1j*phases)
        n_ref = (rng.normal(scale=np.sqrt(sigma_w2/2), size=M)
                 + 1j*rng.normal(scale=np.sqrt(sigma_w2/2), size=M))
        y_ref = (h * (a[:, None] * x_ref)).sum(axis=0) + n_ref
        Y_ref = np.sum(np.abs(y_ref)**2)
        return float((Y_ref - M*sigma_w2) / e0)

class PilotlessNorm(BaseGainNormalizer):
    def norm_scalar(self, *, G_star, **_):
        return float(G_star)


class GivenNorm(BaseGainNormalizer):
    def __init__(self, G_given: float):
        self.G_given = G_given
    def norm_scalar(self, **_):
        return float(self.G_given)

class PathlossNorm(BaseGainNormalizer):
    def __init__(self, include_M: bool = True):
        self.include_M = include_M
    def norm_scalar(self, *, a, M, Omega_vec, **_):
        factor = M if self.include_M else 1.0
        return float(np.sum(a * Omega_vec) * factor)
