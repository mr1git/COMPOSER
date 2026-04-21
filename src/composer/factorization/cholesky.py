"""Pivoted Cholesky factorization (Sec II.B, Eq 11).

For two-electron integrals in physicist notation ``<pq|rs>`` with the
standard 8-fold real symmetry, the matricized form

    M[(p, r), (q, s)] = <pq|rs>

is (real) symmetric positive semi-definite. Pivoted Cholesky
produces factors ``L^mu`` of shape ``(n, n)`` such that

    <pq|rs> ~= sum_{mu=1}^{K} L^mu_{pr} * conj(L^mu_{qs})

to any prescribed trace-norm tolerance. The number of factors ``K``
grows roughly linearly with ``n`` for smooth integrals.

This module exposes
* ``pivoted_cholesky_psd(M, threshold)`` for a generic Hermitian PSD
  matrix, and
* ``cholesky_eri(eri, threshold)`` that wraps the reshape + call and
  returns the rank-3 tensor ``L`` of shape ``(K, n, n)``.

This factorization primitive is broader than the Hamiltonian pool built
in ``operators/hamiltonian.py``: it also supports complex Hermitian-PSD
matricized ERIs. The higher-level pool / LCU path currently uses only
the real-valued electronic-integral subset where the resulting
``L^mu`` are real symmetric matrices.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "pivoted_cholesky_psd",
    "cholesky_eri",
    "reconstruct_eri",
]


def pivoted_cholesky_psd(M: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    """Pivoted Cholesky for a Hermitian PSD matrix M.

    Returns ``L`` of shape ``(K, N)`` with
    ``M ~= sum_{mu=0}^{K-1} L[mu, :] @ L[mu, :].conj().T``, i.e.,
    ``M ~= L.conj().T @ L`` as an ``(N x N)`` rank-K expansion.

    The Harbrecht / Koch-Kirsch algorithm: at each step, pick the
    diagonal entry of the current residual with largest magnitude; take
    that column divided by ``sqrt(diag)``; update the residual. The
    residual remains Hermitian PSD throughout (modulo round-off, which
    we symmetrize against).
    """
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"M must be square, got {M.shape}")
    N = M.shape[0]
    R = M.astype(complex, copy=True)
    # symmetrize against any round-off asymmetry on input
    R = 0.5 * (R + R.conj().T)
    L_rows: list[np.ndarray] = []
    while True:
        diag = np.real(np.diag(R))
        p_star = int(np.argmax(diag))
        if diag[p_star] <= threshold:
            break
        pivot = diag[p_star]
        col = R[:, p_star] / np.sqrt(pivot)
        L_rows.append(col)
        R = R - np.outer(col, col.conj())
        # Keep Hermitian under accumulated floating error
        R = 0.5 * (R + R.conj().T)
    if not L_rows:
        return np.zeros((0, N), dtype=complex)
    return np.array(L_rows)


def cholesky_eri(eri: np.ndarray, threshold: float = 1e-10) -> np.ndarray:
    """Factorize physicist-order ERIs eri[p,q,r,s] = <pq|rs>.

    Returns ``L`` of shape ``(K, n, n)`` such that
    ``eri[p,q,r,s] ~= sum_{mu} L[mu, p, r] * conj(L[mu, q, s])``.

    For real electronic-integral tensors with the usual 8-fold
    symmetry, the reshaped factors are real symmetric. Generic complex
    Hermitian-PSD inputs do not satisfy that stronger property.
    """
    n = eri.shape[0]
    if eri.shape != (n, n, n, n):
        raise ValueError(f"eri must have shape (n,n,n,n), got {eri.shape}")
    # Matricize: axes (p, r) along rows, (q, s) along cols.
    M = eri.transpose(0, 2, 1, 3).reshape(n * n, n * n)
    L_mat = pivoted_cholesky_psd(M, threshold=threshold)
    K = L_mat.shape[0]
    return L_mat.reshape(K, n, n)


def reconstruct_eri(L: np.ndarray) -> np.ndarray:
    """Inverse of ``cholesky_eri``: given L of shape (K, n, n) return
    ``eri[p,q,r,s] = sum_mu L[mu, p, r] * conj(L[mu, q, s])``.
    """
    # einsum for clarity; small n so speed is not a concern here
    return np.einsum("mpr,mqs->pqrs", L, L.conj())
