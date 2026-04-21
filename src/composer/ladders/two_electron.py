"""Two-electron ladder primitives (Sec. III.B, App. A.2).

Given an antisymmetric ``n x n`` matrix ``u`` with unit norm on its
strict upper triangle, the paper's two-electron ladder fixes a pivot
pair ``(r, s)`` and builds a strictly number-conserving unitary
``U_u`` such that

    U_u |e_{rs}> = sum_{p<q} u_{pq} |e_{pq}>,

where ``|e_{pq}> = a_p^dag a_q^dag |vac>``.

The preparation form is obtained by injecting the pivot pair first:

    U_prep(u) = U_u X_r X_s,

so that ``U_prep(u) |vac> = sum_{p<q} u_{pq} |e_{pq}>``.

Appendix A.2 gives a direct ladder over the unordered pair basis:
one phased pair-Givens rotation ``G_{pq,rs}(theta, phi)`` per
non-pivot pair. This module implements that construction directly.

The historical helper ``build_rank2_ladder`` is retained as a
compatibility alias, but it now dispatches to the full App. A.2 ladder
rather than the old rank-2-only orbital-rotation shortcut.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from ..circuits.circuit import Circuit
from ..circuits.gate import Gate
from ..utils import fermion as jw
from ..utils.antisymmetric import index_to_pair, num_pairs, pair_index, pairs_from_matrix
from .one_electron import x_gate
from .phased_pair_givens import phased_pair_givens_gate

__all__ = [
    "PairLadderAngles",
    "solve_angles",
    "build_number_conserving_ladder",
    "build_ladder",
    "orbital_rotation_unitary",
    "build_rank2_ladder",
    "rank2_decomposition",
]


def orbital_rotation_unitary(V: np.ndarray) -> np.ndarray:
    """Return the number-conserving orbital rotation ``U(V)`` on Fock space.

    This utility is still used elsewhere in the repository, but it is
    not the paper's App. A.2 state-preparation ladder.
    """
    n = V.shape[0]
    if V.shape != (n, n):
        raise ValueError("V must be square")
    if not np.allclose(V @ V.conj().T, np.eye(n), atol=1e-10):
        raise ValueError("V must be unitary")
    from scipy.linalg import logm

    X = logm(V)
    X = 0.5 * (X - X.conj().T)
    adag = [jw.jw_a_dagger(p, n) for p in range(n)]
    a_ = [jw.jw_a(p, n) for p in range(n)]
    generator = np.zeros((2**n, 2**n), dtype=complex)
    for p in range(n):
        for q in range(n):
            if abs(X[p, q]) < 1e-15:
                continue
            generator += X[p, q] * (adag[p] @ a_[q])
    return expm(generator)


def rank2_decomposition(U_asym: np.ndarray, tol: float = 1e-8) -> tuple[np.ndarray, np.ndarray] | None:
    """Detect the special rank-2 form ``U = u v^T - v u^T`` when present."""
    Uu, s, _ = np.linalg.svd(U_asym)
    nz = np.sum(s > tol * max(s.max(), 1.0))
    if nz != 2:
        return None
    u1 = Uu[:, 0]
    u2 = Uu[:, 1]
    block = Uu.conj().T @ U_asym @ Uu
    a = block[0, 1]
    if abs(a) < tol:
        return None
    phase = a / abs(a)
    u = u1.copy()
    v = u2 * np.conj(phase)
    return u, v


class PairLadderAngles:
    """Resolved angles for the App. A.2 two-electron ladder.

    Attributes
    ----------
    pivot : tuple[int, int]
        Pivot pair ``(r, s)`` with ``r < s``.
    order : list[tuple[int, int]]
        Non-pivot pairs in lexicographic order. The circuit applies the
        corresponding pair-Givens rotations in reverse order so the
        realized matrix product matches the paper's rightmost-first
        convention.
    thetas : list[float]
        Real rotation angles aligned with ``order``.
    phases : list[float]
        Relative phases aligned with ``order``.
    """

    def __init__(
        self,
        pivot: tuple[int, int],
        order: list[tuple[int, int]],
        thetas: list[float],
        phases: list[float],
    ) -> None:
        self.pivot = pivot
        self.order = order
        self.thetas = thetas
        self.phases = phases


def _validate_target(u_asym: np.ndarray) -> np.ndarray:
    u_asym = np.asarray(u_asym, dtype=complex)
    if u_asym.ndim != 2 or u_asym.shape[0] != u_asym.shape[1]:
        raise ValueError("u_asym must be a square matrix")
    if not np.allclose(u_asym, -u_asym.T, atol=1e-10):
        raise ValueError("u_asym must be antisymmetric")
    pair_vec = pairs_from_matrix(u_asym)
    norm = np.linalg.norm(pair_vec)
    if not np.isclose(norm, 1.0, atol=1e-10):
        raise ValueError(f"u_asym must have unit pair norm, got {norm}")
    return u_asym


def _pair_order(n: int, pivot: tuple[int, int]) -> list[tuple[int, int]]:
    return [index_to_pair(k, n) for k in range(num_pairs(n)) if index_to_pair(k, n) != pivot]


def solve_angles(
    u_asym: np.ndarray, pivot: tuple[int, int] | None = None
) -> PairLadderAngles:
    """Solve the App. A.2 classical recursion for the 2e ladder."""
    u_asym = _validate_target(u_asym)
    n = u_asym.shape[0]
    pair_vec = pairs_from_matrix(u_asym)
    if pivot is None:
        pivot = index_to_pair(int(np.argmax(np.abs(pair_vec))), n)
    r, s = pivot
    if not (0 <= r < s < n):
        raise ValueError(f"invalid pivot pair {pivot} for n={n}")
    pivot_amp = u_asym[r, s]
    if abs(pivot_amp) < 1e-14:
        raise ValueError(f"|u[pivot]| too small: {pivot_amp}")

    order = _pair_order(n, pivot)
    thetas: list[float] = []
    phases: list[float] = []
    residual_sq = abs(pivot_amp) ** 2
    pivot_phase = np.angle(pivot_amp)
    for p, q in order:
        amp = u_asym[p, q]
        thetas.append(float(np.arctan2(abs(amp), np.sqrt(residual_sq))))
        dphi = np.angle(amp) - pivot_phase
        dphi = ((dphi + np.pi) % (2 * np.pi)) - np.pi
        phases.append(float(dphi))
        residual_sq += abs(amp) ** 2
    return PairLadderAngles(pivot=pivot, order=order, thetas=thetas, phases=phases)


def build_number_conserving_ladder(
    u_asym: np.ndarray,
    *,
    n_qubits: int | None = None,
    pivot: tuple[int, int] | None = None,
) -> Circuit:
    """Build the strictly number-conserving two-electron ladder ``U_u``."""
    angles = solve_angles(u_asym, pivot=pivot)
    n_target = np.asarray(u_asym).shape[0]
    n = n_target if n_qubits is None else int(n_qubits)
    if n < n_target:
        raise ValueError(f"n_qubits={n} < target size {n_target}")
    c = Circuit(num_qubits=n)
    r, s = angles.pivot
    for (p, q), theta, phi in zip(
        reversed(angles.order), reversed(angles.thetas), reversed(angles.phases)
    ):
        if abs(theta) < 1e-14:
            continue
        c.append(phased_pair_givens_gate(theta, phi, p, q, r, s, n))
    return c


def build_ladder(
    u_asym: np.ndarray,
    *,
    n_qubits: int | None = None,
    pivot: tuple[int, int] | None = None,
) -> Circuit:
    """Build the preparation-form two-electron ladder ``U_u X_r X_s``."""
    angles = solve_angles(u_asym, pivot=pivot)
    n_target = np.asarray(u_asym).shape[0]
    n = n_target if n_qubits is None else int(n_qubits)
    if n < n_target:
        raise ValueError(f"n_qubits={n} < target size {n_target}")
    r, s = angles.pivot
    c = Circuit(num_qubits=n)
    c.append(x_gate(r))
    c.append(x_gate(s))
    c.extend(build_number_conserving_ladder(u_asym, n_qubits=n, pivot=angles.pivot).gates)
    return c


def build_rank2_ladder(U_asym: np.ndarray, pivot: tuple[int, int] = (0, 1)) -> Circuit:
    """Compatibility alias for the full App. A.2 ladder.

    The historical implementation only supported rank-2 antisymmetric
    targets via a dense orbital-rotation shortcut. The paper's actual
    two-electron ladder applies to any normalized antisymmetric target,
    so this wrapper now dispatches to ``build_ladder``.
    """
    return build_ladder(U_asym, pivot=pivot)

