"""Classical mask ``M^(m)`` over the σ̂ rank-one pool (Sec IV.C).

In the paper the mask is a classical subset / bit string over the full
rank-one generator pool. In this repo we represent it slightly more
generally as a per-channel non-negative weight vector over that full
pool. For the current dense verification path, the weights rescale the
pair-SVD doubles channels of ``sigma_hat`` directly.

The paper's null branch is represented explicitly as ``null_weight``.
For selector-style masks, ``weights`` are usually binary, but the repo
allows general non-negative values so the mask can also re-dial PREP
amplitudes over a fixed compiled pool. When the oracle needs a compiled
normalization in branch-specific units, use
``with_compiled_alpha_bar(...)``.

The compile-once claim supported here is structural but now more
literal than a topology-only proxy: changing the mask may change
compiled PREP amplitudes and oracle matrices, but it should not change
the fixed selector width or the ordered gate schedule tracked by the
circuit signatures used in the similarity-sandwich tests.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ChannelMask", "uniform_mask", "top_k_mask"]


@dataclass
class ChannelMask:
    """Per-channel mask weights plus a null-branch residual.

    Attributes
    ----------
    weights : np.ndarray, shape (ell,)
        Non-negative coefficient for each rank-one channel.
    null_weight : float
        Residual carried by the null branch so ``alpha_bar`` stays
        invariant as the mask changes. Computed to make
        ``sum(weights) + null_weight`` constant for a given pool.
    """

    weights: np.ndarray
    null_weight: float = 0.0

    def __post_init__(self) -> None:
        self.weights = np.asarray(self.weights, dtype=float).ravel()
        if np.any(self.weights < -1e-15):
            raise ValueError("mask weights must be non-negative")
        if self.null_weight < -1e-15:
            raise ValueError("null_weight must be non-negative")

    @property
    def total(self) -> float:
        return float(np.sum(self.weights) + self.null_weight)

    def with_alpha_bar(self, alpha_bar: float) -> "ChannelMask":
        """Return a new mask whose total equals ``alpha_bar``, filling
        the null branch with the residual ``alpha_bar - sum(weights)``.
        Raises if ``alpha_bar < sum(weights)``.

        This keeps the raw selector-weight total fixed. For the actual
        compiled sigma oracle, where each branch can carry its own
        normalization scale, use ``with_compiled_alpha_bar`` instead.
        """
        s = float(np.sum(self.weights))
        if alpha_bar + 1e-12 < s:
            raise ValueError(f"alpha_bar={alpha_bar} too small for mask sum {s}")
        return ChannelMask(weights=self.weights.copy(), null_weight=max(alpha_bar - s, 0.0))

    def compiled_weight_sum(self, branch_scales: np.ndarray) -> float:
        """Return ``sum_s weights[s] * branch_scales[s]``.

        This is the PREP-weight sum for a compiled oracle whose
        branch-``s`` block encoding carries normalization
        ``branch_scales[s]``.
        """
        scales = np.asarray(branch_scales, dtype=float).ravel()
        if scales.shape != self.weights.shape:
            raise ValueError(
                "branch_scales must match mask weights shape: "
                f"got {scales.shape}, expected {self.weights.shape}"
            )
        if np.any(scales < -1e-15):
            raise ValueError("branch_scales must be non-negative")
        return float(np.dot(self.weights, scales))

    def with_compiled_alpha_bar(
        self,
        branch_scales: np.ndarray,
        alpha_bar: float | None = None,
    ) -> "ChannelMask":
        """Return a mask whose ``null_weight`` fixes a compiled PREP total.

        Parameters
        ----------
        branch_scales
            Per-branch normalization factors used by the compiled
            oracle. The active PREP weight is
            ``sum_s weights[s] * branch_scales[s]``.
        alpha_bar
            Target total compiled PREP weight. If omitted, use the
            paper-style full-pool total ``sum_s branch_scales[s]``,
            appropriate for selector masks with ``0 <= weights[s] <= 1``.
        """
        scales = np.asarray(branch_scales, dtype=float).ravel()
        compiled_sum = self.compiled_weight_sum(scales)
        target = float(np.sum(scales)) if alpha_bar is None else float(alpha_bar)
        if target + 1e-12 < compiled_sum:
            raise ValueError(f"alpha_bar={target} too small for compiled mask sum {compiled_sum}")
        return ChannelMask(weights=self.weights.copy(), null_weight=max(target - compiled_sum, 0.0))


def uniform_mask(n_channels: int) -> ChannelMask:
    """All-ones mask — every channel included at unit weight."""
    return ChannelMask(weights=np.ones(n_channels, dtype=float))


def top_k_mask(sigma: np.ndarray, k: int) -> ChannelMask:
    """Keep the top-``k`` channels; zero everyone else.

    ``sigma`` is only used to rank the channels. The returned weights are
    binary selectors, matching the paper's classical mask semantics and
    the App E.3 cumulative-coverage selector in
    ``diagnostics/mask_selection.py``.
    """
    sigma = np.asarray(sigma, dtype=float).ravel()
    weights = np.zeros_like(sigma)
    if k > 0:
        order = np.argsort(sigma)[::-1]
        weights[order[:k]] = 1.0
    return ChannelMask(weights=weights)
