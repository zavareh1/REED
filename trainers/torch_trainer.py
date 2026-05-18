import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from clients.client import BaseTrainer
from torch.nn.utils import parameters_to_vector
import numpy as np, torch
from torch.utils.data import DataLoader, Subset
from clients.client import BaseTrainer

class TorchTrainer(BaseTrainer):
    def __init__(self, model, adapter, dataset, idxs, batch_size=64, lr=0.05, device="cpu"):
        self.model, self.adapter = model, adapter
        self.loader = DataLoader(Subset(dataset, idxs), batch_size=batch_size, shuffle=True)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.device = device
        self.model, self.adapter = model, adapter
        # Ensure adapter points to the *same* model the trainer uses:
        self.adapter.model = self.model
        self.adapter.device = self.device  # keep device in sync too

    def train(
        self,
        w_global: np.ndarray,
        epochs: int | None = None,
        steps: int | None = None,
        lr_override: float | None = None,
    ) -> np.ndarray:
        """Train for either epochs or minibatch steps. Exactly one must be provided."""
        if (epochs is None) == (steps is None):
            raise ValueError("Provide exactly one of epochs or steps")

        # Sync from global
        assert self.adapter.model is self.model, "Adapter bound to different model instance"
        self.adapter.from_vector(w_global)
        self.model.train()

        # IMPORTANT: rebind optimizer params in case adapter reallocated tensors
        for g in self.opt.param_groups:
            g["params"] = list(self.model.parameters())

        base_lr = float(self.opt.param_groups[0]["lr"])
        if lr_override is not None:
            self.opt.param_groups[0]["lr"] = float(lr_override)

        if steps is not None:
            total_steps = int(steps)
            if total_steps < 0:
                raise ValueError("steps must be >= 0")
            if len(self.loader) == 0:
                return parameters_to_vector(self.model.parameters()).detach().cpu().numpy()

            it = iter(self.loader)
            for _ in range(total_steps):
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(self.loader)
                    xb, yb = next(it)

                xb, yb = xb.to(self.device), yb.to(self.device)
                self.opt.zero_grad()
                loss = self.loss_fn(self.model(xb), yb)
                loss.backward()
                self.opt.step()
        else:
            total_epochs = int(epochs)
            if total_epochs < 0:
                raise ValueError("epochs must be >= 0")
            for _ in range(total_epochs):
                for xb, yb in self.loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    self.opt.zero_grad()
                    loss = self.loss_fn(self.model(xb), yb)
                    loss.backward()
                    self.opt.step()

        if lr_override is not None:
            self.opt.param_groups[0]["lr"] = base_lr

        return parameters_to_vector(self.model.parameters()).detach().cpu().numpy()


    # --- Backwards-compat shim (kept temporarily) ---
    def compute_delta(self, w_global: np.ndarray, epochs: int) -> np.ndarray:
        w_new = self.train(w_global, epochs=epochs)
        return w_new - w_global
