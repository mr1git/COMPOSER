"""Linear-algebra helpers used across the package.

Kept intentionally small: safe eigendecomposition of small Hermitian
matrices, vector/ matrix normalization, near-zero detection.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "safe_normalize",
    "hermitian_eig",
    "is_unitary",
    "is_hermitian",
    "is_antihermitian",
    "top_left_block",
]


def safe_normalize(v: np.ndarray, *, tol: float = 1e-14) -> tuple[np.ndarray, float]:
    """Return (v / ||v||, ||v||); if the norm is below tol, returns the
    zero vector and 0.0 rather than raising. The original caller can
    use the returned norm to guard against degenerate cases.
    """
    norm = float(np.linalg.norm(v))
    if norm < tol:
        return np.zeros_like(v), 0.0
    return v / norm, norm


def hermitian_eig(h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hermitian eigendecomposition; wraps numpy.linalg.eigh after
    symmetrizing ``h`` against round-off (so tiny anti-Hermitian dust
    doesn't push eigh into generic eig).
    """
    h_sym = 0.5 * (h + h.conj().T)
    return np.linalg.eigh(h_sym)


def is_unitary(u: np.ndarray, *, atol: float = 1e-10) -> bool:
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        return False
    n = u.shape[0]
    return np.allclose(u.conj().T @ u, np.eye(n), atol=atol)


def is_hermitian(h: np.ndarray, *, atol: float = 1e-10) -> bool:
    return h.ndim == 2 and h.shape[0] == h.shape[1] and np.allclose(h, h.conj().T, atol=atol)


def is_antihermitian(a: np.ndarray, *, atol: float = 1e-10) -> bool:
    return a.ndim == 2 and a.shape[0] == a.shape[1] and np.allclose(a, -a.conj().T, atol=atol)


def top_left_block(u: np.ndarray, block_dim: int) -> np.ndarray:
    """Extract the top-left ``block_dim x block_dim`` corner of ``u``.

    Used to verify block encodings: if ``W`` is a block encoding of
    ``A/alpha`` on ``n_a`` ancilla qubits then
    ``top_left_block(W, 2**n_sys) == A / alpha``.
    """
    if u.ndim != 2 or u.shape[0] != u.shape[1]:
        raise ValueError(f"expected square matrix, got shape {u.shape}")
    if block_dim > u.shape[0]:
        raise ValueError(f"block_dim={block_dim} exceeds matrix dim {u.shape[0]}")
    return u[:block_dim, :block_dim]
