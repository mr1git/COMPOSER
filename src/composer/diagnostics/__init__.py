"""composer.diagnostics subpackage.

Low-level diagnostics that inform mask selection (App E.3) and the
post-hoc subspace-coverage analysis (App E.2).

* ``mp2`` - MP2 doubles amplitudes as a surrogate for T2 (Eq E9).
* ``subspace`` - the rank-cumulative wAUC overlap metric and helpers.
* ``mask_selection`` - App-E.3 MP2-weight ranking plus one-shot
  cumulative-coverage selector masks.
"""
from .mask_selection import channel_weights_by_sigma, channel_weights_mp2, cumulative_coverage_mask
from .mp2 import mp2_doubles_amplitudes, mp2_energy
from .subspace import channel_overlap_matrix, rdm1_drift, wauc

__all__ = [
    "channel_overlap_matrix",
    "channel_weights_by_sigma",
    "channel_weights_mp2",
    "cumulative_coverage_mask",
    "mp2_doubles_amplitudes",
    "mp2_energy",
    "rdm1_drift",
    "wauc",
]
