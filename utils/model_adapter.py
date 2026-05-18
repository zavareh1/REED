
# utils/model_adapter.py
from __future__ import annotations
import numpy as np
import torch
from torch import nn

class TorchModelAdapter:
    """
    Flattens/restores a PyTorch model to/from a single numpy vector.

    Synchronizes:
      - Trainable parameters (weights and biases)
      - BatchNorm running buffers (running_mean, running_var)
    Skips:
      - Non-floating buffers (e.g., num_batches_tracked counters)
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model  = model.to(device)
        self.device = device

        # Parameters
        self._params     = [p for p in self.model.parameters()]
        self._p_shapes   = [tuple(p.shape) for p in self._params]
        self._p_sizes    = [int(p.numel()) for p in self._params]
        self.param_total = int(sum(self._p_sizes))

        # BN running buffers
        self._buffers, self._b_shapes, self._b_sizes = [], [], []
        for name, buf in self.model.named_buffers():
            if not buf.is_floating_point():
                continue
            if name.endswith("running_mean") or name.endswith("running_var"):
                self._buffers.append(buf)
                self._b_shapes.append(tuple(buf.shape))
                self._b_sizes.append(int(buf.numel()))
        self.buffer_total = int(sum(self._b_sizes))

        # Total flattened dimension used by the server/comm stack
        self.total = self.param_total + self.buffer_total

    def to_vector(self) -> np.ndarray:
        with torch.no_grad():
            parts = []
            if self.param_total:
                parts.append(torch.cat([p.detach().reshape(-1).to("cpu") for p in self._params]))
            if self.buffer_total:
                parts.append(torch.cat([b.detach().reshape(-1).to("cpu") for b in self._buffers]))
            flat = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32)
        # Use float64 for numerical stability in aggregation/CSV
        return flat.numpy().astype(np.float64, copy=False)

    def from_vector(self, vec: np.ndarray) -> None:
        if vec.size != self.total:
            raise ValueError(f"vector size {vec.size} != expected {self.total}")
        with torch.no_grad():
            off = 0
            for p, n, shp in zip(self._params, self._p_sizes, self._p_shapes):
                chunk = torch.from_numpy(vec[off:off+n]).view(shp).to(self.device, dtype=p.dtype)
                p.copy_(chunk); off += n
            for b, n, shp in zip(self._buffers, self._b_sizes, self._b_shapes):
                chunk = torch.from_numpy(vec[off:off+n]).view(shp).to(self.device, dtype=b.dtype)
                b.copy_(chunk); off += n
