#!/usr/bin/env python
"""
Permutation‑invariance sanity check for a minimal DeepSets branch
(aggregations: mean, sum, attention) versus a standard MLP branch.
"""

import os
import math
import inspect
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# 1.  Small helpers
# ----------------------------------------------------------------------
class AttentionPool(nn.Module):
    """
    k‑token multi‑head attention aggregator.
    If n_tokens == 1 this is the classic single‑token 'Set Transformer' pool.
    """
    def __init__(self, d_model: int, n_heads: int = 4, n_tokens: int = 4):
        super().__init__()
        self.n_tokens = n_tokens
        self.query_tokens = nn.Parameter(torch.randn(1, n_tokens, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : (B, N, d_model)   encoded sensors
        returns pooled : (B, n_tokens * d_model)
        """
        B = x.size(0)
        q = self.query_tokens.expand(B, -1, -1)    # (B, k, d)
        pooled, _ = self.attn(q, x, x)             # (B, k, d)
        return pooled.flatten(1)                   # (B, k·d)


# ----------------------------------------------------------------------
# 2.  Minimal DeepSets branch
# ----------------------------------------------------------------------
class DeepSetsBranch(nn.Module):
    """
    DeepSets branch that maps a SET of (x_i, u_i) pairs to a latent
    coefficient vector.  Aggregation can be mean, sum or attention.
    """
    def __init__(self,
                 x_dim: int = 1,
                 u_dim: int = 1,
                 latent_dim: int = 64,
                 phi_hidden: int = 128,
                 rho_hidden: int = 128,
                 activation=nn.ReLU,
                 aggregation: str = "mean",
                 attn_tokens: int = 4):
        super().__init__()
        self.aggregation = aggregation.lower()
        assert self.aggregation in {"mean", "sum", "attention"}

        phi_in = x_dim + u_dim
        self.phi = nn.Sequential(
            nn.Linear(phi_in, phi_hidden),
            activation(),
            nn.Linear(phi_hidden, phi_hidden),
            activation(),
            nn.Linear(phi_hidden, latent_dim),
        )

        if self.aggregation == "attention":
            self.pool = AttentionPool(latent_dim, n_tokens=attn_tokens)
            rho_in = latent_dim * attn_tokens
        else:
            rho_in = latent_dim

        self.rho = nn.Sequential(
            nn.Linear(rho_in, rho_hidden),
            activation(),
            nn.Linear(rho_hidden, latent_dim),
        )

    def forward(self, xs: torch.Tensor, us: torch.Tensor) -> torch.Tensor:
        """
        xs : (B, N, x_dim)
        us : (B, N, u_dim)
        returns (B, latent_dim)
        """
        B, N, _ = xs.shape
        # concat → (B·N, x_dim + u_dim)
        feat = torch.cat([xs, us], dim=-1).view(B * N, -1)
        phi_out = self.phi(feat).view(B, N, -1)              # (B, N, d)

        if self.aggregation == "mean":
            agg = phi_out.mean(dim=1)                        # (B, d)
        elif self.aggregation == "sum":
            agg = phi_out.sum(dim=1)                         # (B, d)
        else:                                                # attention
            agg = self.pool(phi_out)                         # (B, k·d)

        return self.rho(agg)                                 # (B, latent_dim)


# ----------------------------------------------------------------------
# 3.  Standard flatten‑and‑MLP branch (not permutation invariant)
# ----------------------------------------------------------------------
class MLPBranch(nn.Module):
    def __init__(self, n_sensors: int, latent_dim: int = 64,
                 hidden: int = 128, activation=nn.ReLU):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_sensors, hidden),
            activation(),
            nn.Linear(hidden, hidden),
            activation(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, y_flat: torch.Tensor) -> torch.Tensor:
        """
        y_flat : (B, n_sensors)
        """
        return self.net(y_flat)


# ----------------------------------------------------------------------
# 4.  Synthetic quadratic data
# ----------------------------------------------------------------------
def quadratic(coeffs, x):
    a, b, c = [coeffs[:, i:i+1] for i in range(3)]     # (B,1)
    return a * x**2 + b * x + c                        # (B, N)


def generate_data(batch=1, n_sensors=50):
    rng = np.random.default_rng()
    coeffs = rng.uniform(-1.5, 1.5, size=(batch, 3)).astype(np.float32)
    x = np.linspace(-10, 10, n_sensors, dtype=np.float32).reshape(-1, 1)  # (N,1)
    y = quadratic(coeffs, x.T).astype(np.float32)        # (B,N)

    perm = rng.permutation(n_sensors)
    x_perm = x[perm]
    y_perm = y[:, perm]

    return (torch.from_numpy(x), torch.from_numpy(y),
            torch.from_numpy(x_perm), torch.from_numpy(y_perm))


# ----------------------------------------------------------------------
# 5.  PI test util
# ----------------------------------------------------------------------
def run_pi_test(model,
                x, y,
                x_perm, y_perm,
                network_type: str,
                name: str,
                device="cpu",
                tol=1e-4):
    model.eval()
    with torch.no_grad():
        if network_type == "deepsets":
            # shapes: xs → (1, N, 1) , us → (1, N, 1)
            xs_orig = x.unsqueeze(0).to(device)             # (1, N, 1)
            us_orig = y.unsqueeze(-1).to(device)            # (1, N, 1)
            xs_perm = x_perm.unsqueeze(0).to(device)        # (1, N, 1)
            us_perm = y_perm.unsqueeze(-1).to(device)       # (1, N, 1)

            coefs_orig = model(xs_orig, us_orig).cpu()
            coefs_perm = model(xs_perm, us_perm).cpu()
        else:  # mlp
            coefs_orig = model(y.to(device)).cpu()
            coefs_perm = model(y_perm.to(device)).cpu()

    diff = (coefs_orig - coefs_perm).abs()
    max_d = diff.max().item()
    mean_d = diff.mean().item()

    passed = max_d < tol if network_type == "deepsets" else max_d >= tol
    flag = "✅" if passed else "❌"

    print(f"\n--- {name} ---")
    print(f"max |Δ| = {max_d:.3e}, mean |Δ| = {mean_d:.3e} --> {flag}")

    return coefs_orig.numpy().ravel(), coefs_perm.numpy().ravel(), max_d, name


# ----------------------------------------------------------------------
# 6.  Main
# ----------------------------------------------------------------------
def main():
    torch.manual_seed(0)
    np.random.seed(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # data
    x, y, x_p, y_p = generate_data(batch=1, n_sensors=50)
    x = x.to(device); x_p = x_p.to(device)
    y_flat = y.clone().to(device)            # (1,N) for MLP
    y_p_flat = y_p.clone().to(device)

    # models
    latent = 64
    branches = [
        ("DeepSets‑MEAN",    DeepSetsBranch(aggregation="mean"     , latent_dim=latent).to(device), "deepsets"),
        ("DeepSets‑SUM",     DeepSetsBranch(aggregation="sum"      , latent_dim=latent).to(device), "deepsets"),
        ("DeepSets‑ATTENTION",    DeepSetsBranch(aggregation="attention", latent_dim=latent).to(device), "deepsets"),
        ("Standard MLP",     MLPBranch(n_sensors=y.shape[1], latent_dim=latent).to(device),        "mlp"),
    ]

    results = []
    for name, model, kind in branches:
        results.append(
            run_pi_test(model,
                        x,  y,      # original
                        x_p, y_p,   # permuted
                        network_type=kind,
                        name=name,
                        device=device)
        )

    # scatter plot
    cols = len(results)
    fig, ax = plt.subplots(1, cols, figsize=(4 * cols, 4))
    if cols == 1:
        ax = [ax]

    for i, (orig, perm, max_d, name) in enumerate(results):
        ax[i].scatter(orig, perm, s=8)
        ax[i].plot([orig.min(), orig.max()], [orig.min(), orig.max()],
                   'r--', lw=1)
        ax[i].set_title(f"{name}\nmax Δ={max_d:.1e}")
        ax[i].set_xlabel("original")
        ax[i].set_ylabel("permuted")
        ax[i].set_aspect("equal")

    plt.tight_layout()
    plt.savefig("pi_test_scratch.png")
    print("\nSaved plot → pi_test_scratch.png")


if __name__ == "__main__":
    main() 