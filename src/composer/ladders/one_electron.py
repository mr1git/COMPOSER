"""One-electron ladder primitives (Sec III.A, App A.1).

Given a complex unit vector ``u`` of length ``n`` with a designated
pivot orbital ``r`` (``u[r] != 0``), produce a circuit ``U_u`` such
that

    U_u |e_r>  =  sum_p u_p |e_p>

where ``|e_p> = a_p^dag |vac>``. The strictly number-conserving ladder
``U_u`` is the primitive reused inside rank-one block encodings. The
state-preparation form is obtained by preceding it with pivot
injection, ``U_u X_r``.

The ladder itself is a product of ``n-1`` fermionic Givens rotations
between each non-pivot orbital and the pivot, plus optional per-qubit
``Rz`` phases for complex ``u``.

Angle recursion (App A.1, adapted to our ``G_pr`` convention):

    tan(theta_k) = |u_{p_k}| / sqrt(|u_r|^2 + sum_{i<k} |u_{p_i}|^2)
    sign(theta_k) absorbs the sign of the real amplitude
    per-qubit phase phi_p = arg(u_p) - arg(u_r) applied as Rz on qubit p

The preparation-form circuit returned by ``build_ladder`` contains:

1. X on the pivot qubit (to create |e_r> from |vac>).
2. n-1 Givens gates ``G_{p_k, r}(theta_k)`` in order
   k = n-2, n-3, ..., 0 (innermost first in matrix order; see
   derivation in ``ASSUMPTIONS.md``).
3. Optional per-qubit ``Rz`` phase corrections.

See ``tests/test_one_electron_ladder.py`` for separate verification of
the preparation and number-conserving forms.
"""
from __future__ import annotations

import numpy as np

from ..circuits.circuit import Circuit
from ..circuits.gate import Gate
from ..circuits.simulator import unitary as circuit_unitary
from .givens import givens_gate

__all__ = [
    "LadderAngles",
    "solve_angles",
    "build_number_conserving_ladder",
    "build_ladder",
    "mode_rotation_unitary",
    "x_gate",
    "rz_gate",
]


def x_gate(qubit: int) -> Gate:
    return Gate(
        name="X",
        qubits=(qubit,),
        matrix=np.array([[0, 1], [1, 0]], dtype=complex),
        kind="X",
    )


def rz_gate(qubit: int, phi: float) -> Gate:
    """Rz(phi) = diag(exp(-i phi/2), exp(i phi/2))."""
    return Gate(
        name=f"Rz({phi:.4f})",
        qubits=(qubit,),
        matrix=np.diag([np.exp(-1j * phi / 2), np.exp(1j * phi / 2)]),
        kind=f"Rz({qubit})",
    )


class LadderAngles:
    """Resolved angles for a one-electron ladder.

    Attributes
    ----------
    pivot : int
    order : list[int]
        The non-pivot indices ``p_0, p_1, ..., p_{n-2}`` in the order
        they appear **in the target sum**. The *circuit* applies
        ``G_{p_{n-2}, r}``, then ``G_{p_{n-3}, r}``, ..., ending with
        ``G_{p_0, r}``.
    thetas : list[float]
        Real rotation angles, aligned with ``order``.
    phases : list[tuple[int, float]]
        ``(qubit, phi)`` Rz corrections to apply after Givens.
    """

    def __init__(
        self,
        pivot: int,
        order: list[int],
        thetas: list[float],
        phases: list[tuple[int, float]],
    ) -> None:
        self.pivot = pivot
        self.order = order
        self.thetas = thetas
        self.phases = phases


def solve_angles(u: np.ndarray, pivot: int | None = None) -> LadderAngles:
    """Solve the classical recursion shared by both 1e ladder forms.

    The pivot defaults to ``argmax |u_p|`` for best numerical
    conditioning (ASSUMPTION: "pivot selection").
    """
    u = np.asarray(u, dtype=complex)
    if u.ndim != 1:
        raise ValueError("u must be 1-D")
    n = int(u.shape[0])
    if not np.isclose(np.linalg.norm(u), 1.0, atol=1e-10):
        raise ValueError(f"u must be unit norm, got {np.linalg.norm(u)}")
    if pivot is None:
        pivot = int(np.argmax(np.abs(u)))
    if not (0 <= pivot < n):
        raise ValueError(f"pivot {pivot} out of range for n={n}")
    if np.abs(u[pivot]) < 1e-14:
        raise ValueError(f"|u[pivot]| too small: {u[pivot]}")

    # Work with magnitudes and phases separately.
    mags = np.abs(u)
    args = np.angle(u)
    # Order: non-pivot indices in ascending order
    order = [p for p in range(n) if p != pivot]
    thetas: list[float] = []
    # residual = magnitude remaining on the pivot at each step
    # Starting residual^2 = |u_r|^2 (before adding any p_k contribution)
    residual_sq = mags[pivot] ** 2
    for p in order:
        # tan theta = |u_p| / sqrt(residual_sq)
        theta = float(np.arctan2(mags[p], np.sqrt(residual_sq)))
        thetas.append(theta)
        residual_sq += mags[p] ** 2
    # Phase corrections. After the magnitude ladder, state is
    # sum_p |u_p| |e_p>. Applying Rz_q(phi_q) on each qubit q gives
    # |e_p> phase phi_p - (1/2) sum_q phi_q; we want this to equal
    # arg(u_p) up to a global phase. Choosing phi_p = arg(u_p) - arg(u_r)
    # (with phi_r = 0) satisfies the condition (differences of phi_p
    # match the target alpha_p differences).
    phases: list[tuple[int, float]] = []
    for p in range(n):
        dphi = args[p] - args[pivot]
        # Normalize to (-pi, pi]
        dphi = ((dphi + np.pi) % (2 * np.pi)) - np.pi
        if abs(dphi) > 1e-14:
            phases.append((p, dphi))
    return LadderAngles(pivot=pivot, order=order, thetas=thetas, phases=phases)

def build_number_conserving_ladder(
    u: np.ndarray, *, n_qubits: int | None = None, pivot: int | None = None
) -> Circuit:
    """Build the strictly number-conserving ladder ``U_u`` on ``n`` qubits.

    On the one-electron sector this satisfies
    ``U_u |e_pivot> = sum_p u_p |e_p>``. The same angles and phases are
    used by the preparation-form circuit; only the initial pivot
    injection is absent.
    """
    angles = solve_angles(u, pivot=pivot)
    n = int(u.shape[0]) if n_qubits is None else int(n_qubits)
    if n < u.shape[0]:
        raise ValueError(f"n_qubits={n} < len(u)={u.shape[0]}")
    c = Circuit(num_qubits=n)
    # Reverse application order so the realized matrix product matches
    # the paper's rightmost-factor-acts-first product convention.
    for p, theta in zip(reversed(angles.order), reversed(angles.thetas)):
        if abs(theta) < 1e-14:
            continue
        c.append(givens_gate(theta, p, angles.pivot, n))
    for q, phi in angles.phases:
        c.append(rz_gate(q, phi))
    return c


def build_ladder(u: np.ndarray, *, n_qubits: int | None = None, pivot: int | None = None) -> Circuit:
    """Build the preparation-form ladder ``U_u X_r`` on ``n`` qubits.

    ``n_qubits`` defaults to ``len(u)``; passing a larger value embeds
    the ladder into a wider register with the extra qubits unused
    (still ``|0>``).
    """
    angles = solve_angles(u, pivot=pivot)
    n = int(u.shape[0]) if n_qubits is None else int(n_qubits)
    if n < u.shape[0]:
        raise ValueError(f"n_qubits={n} < len(u)={u.shape[0]}")
    c = Circuit(num_qubits=n)
    c.append(x_gate(angles.pivot))
    c.extend(build_number_conserving_ladder(u, n_qubits=n, pivot=angles.pivot).gates)
    return c


def mode_rotation_unitary(
    u: np.ndarray, *, n_qubits: int | None = None, pivot: int | None = None
) -> tuple[np.ndarray, int]:
    """Return the dense number-conserving orbital rotation ``U_u``.

    The returned unitary satisfies

        ``U_u |e_pivot> = sum_p u_p |e_p>``

    on the one-electron sector, and therefore

        ``U_u n_pivot U_u^dag = n[u]``

    on the full Fock space. This is the rotated-mode primitive reused
    by the more literal Appendix B.2 occupation-flag construction.
    """
    angles = solve_angles(u, pivot=pivot)
    n = int(u.shape[0]) if n_qubits is None else int(n_qubits)
    if n < u.shape[0]:
        raise ValueError(f"n_qubits={n} < len(u)={u.shape[0]}")
    U = circuit_unitary(
        build_number_conserving_ladder(u, n_qubits=n, pivot=angles.pivot)
    )
    return U, angles.pivot
