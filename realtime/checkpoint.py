"""Load the trained SONIX head for the live engine.

The head was trained on Z-SCORED embeddings, so the checkpoint's mu/sd must be
applied before the linear layers. We bake them into the module so callers can
just do model(embeddings) and cannot forget the normalisation -- forgetting it
produces plausible-looking but meaningless scores.
"""

import numpy as np
import torch
import torch.nn as nn


class StandardisedHead(nn.Module):
    """Trained MLP head with the checkpoint's own input standardiser built in."""

    def __init__(self, cfg, state_dict, mu, sd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg["in_dim"], cfg["hidden"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["hidden"], 1),
        )
        self.net.load_state_dict(state_dict)
        self.register_buffer("mu", torch.from_numpy(np.asarray(mu, np.float32)))
        self.register_buffer("sd", torch.from_numpy(np.asarray(sd, np.float32)))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def load_checkpoint(path, device="cuda"):
    """Return (model, config). Raises loudly rather than returning None."""
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    for k in ("config", "state_dict", "mu", "sd"):
        if k not in ckpt:
            raise ValueError(f"{path} is missing '{k}' -- not a SONIX head checkpoint")

    sd = np.asarray(ckpt["sd"], np.float32).copy()
    sd[sd == 0] = 1.0                    # a constant feature would give inf/nan

    model = StandardisedHead(ckpt["config"], ckpt["state_dict"], ckpt["mu"], sd)
    model.eval().to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt["config"]
