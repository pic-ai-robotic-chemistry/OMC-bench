
from types import SimpleNamespace
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

from .geom_basis_ext import geom_basis_tail


def compute_bessel_math(
    d: torch.Tensor,
    r_max: float,
    freq: torch.Tensor,
) -> torch.Tensor:
    d_scaled = d / r_max
    prefactor = (2.0 / r_max) ** 0.5
    return prefactor * torch.sin(freq * d_scaled) / (d + 1e-6)


def compute_envelope_math(d: torch.Tensor, r_cut: float) -> torch.Tensor:
    inv_r_cut = 1.0 / r_cut
    x = d * inv_r_cut
    x = torch.clamp(x, min=0.0, max=1.0)

    x2 = x * x
    x3 = x2 * x
    return 1.0 - x3 * (10.0 - 15.0 * x + 6.0 * x2)


class BesselBasis(nn.Module):
    def __init__(self, r_max: float, num_basis: int = 8) -> None:
        super().__init__()
        self.r_max = float(r_max)
        self.num_basis = int(num_basis)
        self.register_buffer('freq', torch.arange(1, num_basis + 1).float() * np.pi)

    def forward(self, d: torch.Tensor) -> torch.Tensor:
        return compute_bessel_math(d, self.r_max, self.freq)


class PolynomialEnvelope(nn.Module):
    def __init__(self, r_cut: float, p: int = 5) -> None:
        super().__init__()
        self.r_cutoff = float(r_cut)
        self.p = int(p)

    def forward(self, d_ij: torch.Tensor) -> torch.Tensor:
        return compute_envelope_math(d_ij, self.r_cutoff)


class GeometricBasisEinsumRef(nn.Module):
    '''Reference implementation matching the original logic.'''

    def __init__(self, config) -> None:
        super().__init__()
        self.cfg = config
        self.rbf = BesselBasis(config.cutoff, config.num_rbf)
        self.envelope = PolynomialEnvelope(r_cut=config.cutoff)
        self.rbf_mlp = nn.Sequential(
            nn.Linear(config.num_rbf, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        eye_scaled = torch.eye(3) / 3.0
        self.register_buffer('_eye_scaled', eye_scaled.view(1, 3, 3))

    def forward(
        self,
        vec_ij: torch.Tensor,
        d_ij: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        bessel = self.rbf(d_ij.unsqueeze(-1))
        raw_rbf = self.rbf_mlp(bessel)
        env = self.envelope(d_ij)
        rbf_feat = raw_rbf * env.unsqueeze(-1)

        r_hat = vec_ij / (d_ij.unsqueeze(-1) + 1e-6)
        basis: Dict[int, torch.Tensor] = {0: rbf_feat}

        if self.cfg.use_L1 or self.cfg.use_L2:
            basis[1] = torch.einsum('ef,ei->eif', rbf_feat, r_hat)

        if self.cfg.use_L2:
            outer = torch.einsum('ei,ej->eij', r_hat, r_hat)
            trace_less = outer - self._eye_scaled.to(dtype=outer.dtype, device=outer.device)
            basis[2] = torch.einsum('ef,eij->eijf', rbf_feat, trace_less)

        return basis, r_hat


class GeometricBasisFused(nn.Module):
    '''Drop-in GeometricBasis replacement with a fused CUDA tail.

    This keeps Bessel / envelope / MLP in PyTorch and only fuses:
      - r_hat
      - basis[1]
      - basis[2]
    '''

    def __init__(self, config) -> None:
        super().__init__()
        self.cfg = config
        self.rbf = BesselBasis(config.cutoff, config.num_rbf)
        self.envelope = PolynomialEnvelope(r_cut=config.cutoff)
        self.rbf_mlp = nn.Sequential(
            nn.Linear(config.num_rbf, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # Keep the same buffer name as the original module so state_dict stays compatible.
        eye_scaled = torch.eye(3) / 3.0
        self.register_buffer('_eye_scaled', eye_scaled.view(1, 3, 3))

    def forward(
        self,
        vec_ij: torch.Tensor,
        d_ij: torch.Tensor,
    ) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
        bessel = self.rbf(d_ij.unsqueeze(-1))
        raw_rbf = self.rbf_mlp(bessel)
        env = self.envelope(d_ij)
        rbf_feat = raw_rbf * env.unsqueeze(-1)

        need_l1 = bool(self.cfg.use_L1 or self.cfg.use_L2)
        need_l2 = bool(self.cfg.use_L2)

        r_hat, basis1, basis2 = geom_basis_tail(
            vec_ij,
            d_ij,
            rbf_feat,
            need_l1=need_l1,
            need_l2=need_l2,
        )

        basis: Dict[int, torch.Tensor] = {0: rbf_feat}
        if need_l1:
            basis[1] = basis1
        if need_l2:
            basis[2] = basis2
        return basis, r_hat


def make_demo_config(
    cutoff: float = 6.0,
    num_rbf: int = 12,
    hidden_dim: int = 128,
    use_L1: bool = True,
    use_L2: bool = True,
):
    return SimpleNamespace(
        cutoff=cutoff,
        num_rbf=num_rbf,
        hidden_dim=hidden_dim,
        use_L1=use_L1,
        use_L2=use_L2,
    )
