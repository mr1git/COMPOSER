"""Weighted subspace-overlap diagnostic (App E.2, Eq E7-E8).

For singular-value ordered rank-one manifolds ``A`` and ``B``, the paper
defines the rank-``r`` overlap

    ov(r) = (1 / r) || B_r^dagger \tilde{B}_r ||_F^2

where ``B_r`` and ``\tilde{B}_r`` collect the first ``r`` vectorized
rank-one basis vectors. The weighted average overlap is then

    wAUC(R) = sum_{r=1..R} w_r ov(r),
    w_r = s_r^2 / sum_{k=1..R} s_k^2,

with ``s_r`` the reference singular values.

In the repo, ``reference`` is the ranked reference manifold and
``truncated`` is typically a shorter mask-selected prefix/subset. We
therefore evaluate Eq. (E7) against the available truncated prefix at
each rank:

    ov_repo(r) = (1 / r) || B_r^dagger \tilde{B}_{min(r, |B|)} ||_F^2.

This preserves the paper's cumulative-rank weighting and keeps
``wAUC(reference, reference) == 1`` while penalizing missing higher-rank
channels in a shorter truncated list.

Also exposed:

* ``channel_overlap_matrix(A, B)`` — the ``|A| x |B|`` matrix of pair
  overlaps ``|<b_mu, \tilde{b}_nu>|^2``.
* ``rdm1_drift`` — an ``||D_A - D_B||_F`` difference for 1-RDM-like
  tensors (App E.4 style), used as a secondary sanity check.
"""
from __future__ import annotations

import numpy as np

from ..factorization.pair_svd import PairChannel

__all__ = ["channel_overlap_matrix", "wauc", "rdm1_drift"]


def _pair_overlap(U1: np.ndarray, U2: np.ndarray) -> float:
    """Absolute inner product of two antisymmetric matrices treated as
    vectors in their upper-triangular indexing.

    The pair-SVD convention in ``factorization/pair_svd.py`` makes
    ``||U||_F = sqrt(2)`` (off-diagonal double counting); we therefore
    divide by 2 so the normalized overlap is in [0, 1].
    """
    # Frobenius inner product
    ip = np.vdot(U1.ravel(), U2.ravel())
    norm1 = np.linalg.norm(U1)
    norm2 = np.linalg.norm(U2)
    if norm1 < 1e-15 or norm2 < 1e-15:
        return 0.0
    return float(abs(ip) / (norm1 * norm2))


def channel_overlap_matrix(
    A: list[PairChannel],
    B: list[PairChannel],
) -> np.ndarray:
    """Return an ``(len(A), len(B))`` matrix of pair overlaps in [0, 1].

    Entry ``[mu, nu]`` is ``|<U^A_mu, U^B_nu>|^2 * |<V^A_mu, V^B_nu>|^2``.
    """
    m, n = len(A), len(B)
    O = np.zeros((m, n))
    for mu in range(m):
        for nu in range(n):
            ou = _pair_overlap(A[mu].U, B[nu].U)
            ov = _pair_overlap(A[mu].V, B[nu].V)
            O[mu, nu] = ou**2 * ov**2
    return O


def wauc(
    reference: list[PairChannel],
    truncated: list[PairChannel],
) -> float:
    """Weighted AUC of ``truncated`` against ``reference``.

    The inputs are assumed to be singular-value ordered, as produced by
    ``pair_svd_decompose``. The score is the App-E.2 weighted sum of the
    cumulative-rank overlaps ``ov(r)``.
    """
    if not reference:
        return 0.0
    if not truncated:
        return 0.0
    sigma_sq_ref = np.array([abs(ch.sigma) ** 2 for ch in reference], dtype=float)
    total_sigma_sq = float(sigma_sq_ref.sum())
    if total_sigma_sq < 1e-15:
        return 0.0
    weights = sigma_sq_ref / total_sigma_sq
    O = channel_overlap_matrix(reference, truncated)
    n_trunc = len(truncated)
    ov_by_rank = np.zeros(len(reference), dtype=float)
    for r in range(1, len(reference) + 1):
        cols = min(r, n_trunc)
        ov_by_rank[r - 1] = float(O[:r, :cols].sum() / r)
    return float(np.dot(weights, ov_by_rank))


def rdm1_drift(D_reference: np.ndarray, D_test: np.ndarray) -> float:
    """Frobenius distance between two 1-RDM-shaped tensors.

    Thin wrapper so the diagnostic naming matches App E.4's "1-RDM
    drift" metric. No symmetrization — the caller should pass density
    matrices already in compatible shape.
    """
    if D_reference.shape != D_test.shape:
        raise ValueError("shape mismatch between reference and test RDMs")
    return float(np.linalg.norm(D_reference - D_test))
