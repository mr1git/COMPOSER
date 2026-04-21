"""App E.3 mask-selection: one-shot cumulative-coverage selector mask.

Given the pair-SVD rank-one pool of a surrogate T2 (typically the MP2
amplitudes from ``mp2.py``), rank channels by the paper's ladder weight

    w_s^MP2 = |omega_s|^2 ||U^(s)||_F^2 ||V^(s)||_F^2

and keep the smallest prefix whose cumulative weight reaches a
user-specified coverage fraction ``rho in (0, 1]``. The resulting
``ChannelMask`` is a *selector mask*: chosen channels get weight ``1``
and everyone else gets ``0``.

The compile-once selector width (``ASSUMPTION #10``, null branch) is
preserved by giving the caller the option to re-dial the mask via
``ChannelMask.with_alpha_bar(alpha_bar)`` after selection.

Cumulative coverage metric:

    coverage(k)  =  ( sum_{mu=1..k} w_mu^MP2 )  /  ( sum_mu w_mu^MP2 )

This is the "one-shot" variant App E.3 names explicitly; iterative /
residual-greedy variants are documented as follow-up in
``IMPLEMENTATION_LOG.md``.
"""
from __future__ import annotations

import numpy as np

from ..factorization.pair_svd import PairChannel
from ..operators.mask import ChannelMask

__all__ = ["channel_weights_mp2", "channel_weights_by_sigma", "cumulative_coverage_mask"]


def channel_weights_mp2(channels: list[PairChannel]) -> np.ndarray:
    """Return the App-E.3 MP2 ladder weights ``w_s^MP2``.

    For pair-SVD channels from ``pair_svd_decompose``, ``||U||_F`` and
    ``||V||_F`` are both ``sqrt(2)``, so this reduces to a global factor
    times ``sigma^2``. The full expression is kept here so the diagnostic
    remains correct even if the caller supplies channels with a different
    normalization.
    """
    return np.array(
        [
            (abs(ch.sigma) ** 2) * (np.linalg.norm(ch.U) ** 2) * (np.linalg.norm(ch.V) ** 2)
            for ch in channels
        ],
        dtype=float,
    )


def channel_weights_by_sigma(channels: list[PairChannel]) -> np.ndarray:
    """Compatibility alias for the App-E.3 MP2 ladder weights."""
    return channel_weights_mp2(channels)


def cumulative_coverage_mask(
    channels: list[PairChannel],
    coverage: float,
) -> ChannelMask:
    """Selector mask keeping the smallest prefix whose cumulative
    MP2 ladder weight exceeds ``coverage * sum(w_s^MP2)``.

    Parameters
    ----------
    channels : rank-one pair pool (``pair_svd_decompose`` output).
    coverage : target cumulative-coverage fraction in ``(0, 1]``.
    """
    if not 0.0 < coverage <= 1.0 + 1e-12:
        raise ValueError(f"coverage must be in (0, 1], got {coverage}")
    weights_mp2 = channel_weights_mp2(channels)
    total = weights_mp2.sum()
    if total < 1e-15:
        return ChannelMask(weights=np.zeros_like(weights_mp2))
    order = np.argsort(weights_mp2)[::-1]
    cum = 0.0
    k_selected = 0
    target = coverage * total
    for k, idx in enumerate(order, start=1):
        cum += weights_mp2[idx]
        if cum + 1e-12 >= target:
            k_selected = k
            break
    else:
        k_selected = len(weights_mp2)
    selector = np.zeros_like(weights_mp2)
    selector[order[:k_selected]] = 1.0
    return ChannelMask(weights=selector)
