"""Model factory from config (Transolver first, docs/methods.md §4).

Every Transolver constructor argument is passed explicitly: in PhysicsNeMo 2.1.1 the defaults are
use_te=True, n_hidden=256, n_head=8, which must never be picked up by accident.
"""

from __future__ import annotations

from torch import nn

from plaid_surrogates.config import ModelConfig

COORD_DIM = 2  # 2D meshes: (x, y) go in the embedding


def build_transolver(cfg: ModelConfig, functional_dim: int, out_dim: int) -> nn.Module:
    """Unstructured-mesh Transolver: forward(fx (B,N,C_in), embedding=(B,N,2)) -> (B,N,C_out)."""
    from physicsnemo.models.transolver import Transolver

    return Transolver(
        functional_dim=functional_dim,
        out_dim=out_dim,
        embedding_dim=COORD_DIM,
        n_layers=cfg.n_layers,
        n_hidden=cfg.n_hidden,
        dropout=cfg.dropout,
        n_head=cfg.n_head,
        act=cfg.act,
        mlp_ratio=cfg.mlp_ratio,
        slice_num=cfg.slice_num,
        unified_pos=False,
        ref=8,  # only used with unified_pos=True
        structured_shape=None,
        use_te=cfg.use_te,
        time_input=False,
        plus=cfg.plus,
    )


def count_params(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
