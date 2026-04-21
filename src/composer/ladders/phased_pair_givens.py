"""Phased pair-Givens primitive (Eq. 32).

G_{pq, rs}(theta, phi) is a number-conserving rotation on the
two-electron subspace ``span{|e_{pq}>, |e_{rs}>}`` for any distinct
unordered pairs ``{p, q} != {r, s}`` with ``p < q`` and ``r < s``. The
pairs may overlap in one orbital; Eq. (32) does not require four
distinct indices. In the ordered basis ``(|e_{rs}>, |e_{pq}>)`` the
rotation is

    G(theta, phi) |e_{rs}> = e^{i phi} sin(theta) |e_{pq}> + cos(theta) |e_{rs}>
    G(theta, phi) |e_{pq}> = cos(theta) |e_{pq}> - e^{-i phi} sin(theta) |e_{rs}>

which is the Eq. (32) convention used by the paper.

The generator is

    G_theta_phi = exp(theta * (e^{i phi} a_p^dag a_q^dag a_s a_r
                              - e^{-i phi} a_r^dag a_s^dag a_q a_p))

which is explicitly anti-Hermitian. The module exposes a full
``2**n x 2**n`` dense matrix of this evolution, used for
verification.

A fast 4-qubit decomposition into single Givens + controlled-phase is
possible and is documented in ``ASSUMPTIONS.md`` ("pair-Givens
decomposition"); the dense-matrix form here is what the tests use.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..circuits.gate import Gate
from ..utils import fermion as jw

__all__ = ["phased_pair_givens_matrix", "phased_pair_givens_gate"]


def _validate_pair(a: int, b: int, n_qubits: int, name: str) -> tuple[int, int]:
    if not (0 <= a < n_qubits and 0 <= b < n_qubits):
        raise ValueError(f"{name} indices {(a, b)} out of range for n_qubits={n_qubits}")
    if a == b:
        raise ValueError(f"{name} must contain two distinct orbitals, got {(a, b)}")
    if a > b:
        raise ValueError(f"{name} must be ordered with the smaller orbital first, got {(a, b)}")
    return a, b


def phased_pair_givens_matrix(
    theta: float, phi: float, p: int, q: int, r: int, s: int, n_qubits: int
) -> np.ndarray:
    """Full 2**n x 2**n matrix of the phased pair-Givens rotation."""
    p, q = _validate_pair(p, q, n_qubits, "target pair")
    r, s = _validate_pair(r, s, n_qubits, "pivot pair")
    if (p, q) == (r, s):
        raise ValueError("target pair and pivot pair must be distinct")
    adag = [jw.jw_a_dagger(k, n_qubits) for k in range(n_qubits)]
    a_ = [jw.jw_a(k, n_qubits) for k in range(n_qubits)]
    op = (
        np.exp(1j * phi) * (adag[p] @ adag[q] @ a_[s] @ a_[r])
        - np.exp(-1j * phi) * (adag[r] @ adag[s] @ a_[q] @ a_[p])
    )
    return expm(theta * op)


def phased_pair_givens_gate(
    theta: float, phi: float, p: int, q: int, r: int, s: int, n_qubits: int
) -> Gate:
    """Gate wrapper; qubits set to all n so the simulator applies the
    full dense matrix. Topology kind names the logical pair-pair.
    """
    mat = phased_pair_givens_matrix(theta, phi, p, q, r, s, n_qubits)
    return Gate(
        name=f"PhasedPairGivens({p},{q};{r},{s};{theta:.3f},{phi:.3f})",
        qubits=tuple(range(n_qubits)),
        matrix=mat,
        kind=f"PhasedPairGivens({p},{q};{r},{s})",
    )
