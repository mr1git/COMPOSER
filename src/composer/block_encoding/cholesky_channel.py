"""Lemma 2 (Eq. 34): rotated-mode block encoding of one Cholesky channel.

Appendix B.2 constructs ``O_mu^2`` in two steps:

1. Prepare an index register with amplitudes proportional to
   ``sqrt(|lambda_xi|)`` and use an occupation-flag gadget to block
   encode the commuting rotated-mode projectors ``n_{mu xi}``.
2. Apply a constant-overhead degree-2 transform to that block encoding
   to obtain ``(O_mu / Gamma_mu)^2``.

The main builder in this module now follows that structure explicitly.
Dense reflection helpers are retained only as compatibility wrappers
for older callers elsewhere in the repository.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..circuits.circuit import Circuit
from ..circuits.gate import Gate
from ..circuits.simulator import unitary as circuit_unitary
from ..ladders.one_electron import mode_rotation_unitary, x_gate
from ..qsp.qsvt_poly import degree_two_projector_transform
from ..utils import fermion as jw

__all__ = [
    "CholeskyChannelBlockEncoding",
    "apply_x_squared_qsvt",
    "build_cholesky_channel_block_encoding",
    "cholesky_channel_block_encoding",
    "hermitian_one_body_block_encoding",
    "x_squared_qsvt_unitary",
]


def _reflection_block_encoding(A: np.ndarray) -> np.ndarray:
    """Return ``[[A, sqrt(I-A^2)], [sqrt(I-A^2), -A]]`` for a Hermitian contraction."""
    A = np.asarray(A, dtype=complex)
    if not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError("A must be Hermitian")
    eigvals, eigvecs = np.linalg.eigh(A)
    if np.max(np.abs(eigvals)) > 1.0 + 1e-10:
        raise ValueError("A must be a contraction")
    rad = np.sqrt(np.clip(1.0 - eigvals**2, 0.0, 1.0))
    S = (eigvecs * rad) @ eigvecs.conj().T
    S = 0.5 * (S + S.conj().T)
    dim = A.shape[0]
    W = np.zeros((2 * dim, 2 * dim), dtype=complex)
    W[:dim, :dim] = A
    W[:dim, dim:] = S
    W[dim:, :dim] = S
    W[dim:, dim:] = -A
    return W


def _cnot_matrix_lsb_first(qubits: tuple[int, int], control: int, target: int) -> np.ndarray:
    if set(qubits) != {control, target}:
        raise ValueError("qubits must be exactly the control/target pair")
    bit_c = qubits.index(control)
    bit_t = qubits.index(target)
    mat = np.zeros((4, 4), dtype=complex)
    for i in range(4):
        c_val = (i >> bit_c) & 1
        t_val = (i >> bit_t) & 1
        t_new = t_val ^ c_val
        j = i
        if t_new != t_val:
            j ^= 1 << bit_t
        mat[j, i] = 1.0
    return mat


def _cnot_gate(control: int, target: int) -> Gate:
    if control == target:
        raise ValueError("control and target must differ")
    lo, hi = (control, target) if control < target else (target, control)
    qubits = (lo, hi)
    return Gate(
        name=f"CNOT({control}->{target})",
        qubits=qubits,
        matrix=_cnot_matrix_lsb_first(qubits, control, target),
        kind=f"CNOT({control}->{target})",
    )


def _top_left_block(unitary: np.ndarray, n_system: int) -> np.ndarray:
    dim_sys = 1 << n_system
    return unitary[:dim_sys, :dim_sys]


def _prep_unitary(weights_abs: np.ndarray, n_ancilla: int) -> np.ndarray:
    """Return a dense PREP with the requested amplitudes on ``|0...0>``."""
    if n_ancilla < 0:
        raise ValueError(f"n_ancilla must be non-negative, got {n_ancilla}")
    dim = 1 << n_ancilla
    if dim == 1:
        return np.ones((1, 1), dtype=complex)
    total = float(np.sum(weights_abs))
    if total <= 0.0:
        raise ValueError("PREP weights must sum to a positive value")
    amps = np.zeros(dim, dtype=complex)
    amps[: weights_abs.shape[0]] = np.sqrt(weights_abs / total)
    M = np.zeros((dim, dim), dtype=complex)
    M[:, 0] = amps
    for j in range(1, dim):
        col = np.zeros(dim, dtype=complex)
        col[j] = 1.0
        for k in range(j):
            col = col - np.vdot(M[:, k], col) * M[:, k]
        norm = np.linalg.norm(col)
        if norm < 1e-12:
            for alt in range(dim):
                cand = np.zeros(dim, dtype=complex)
                cand[alt] = 1.0
                for k in range(j):
                    cand = cand - np.vdot(M[:, k], cand) * M[:, k]
                cand_norm = np.linalg.norm(cand)
                if cand_norm > 1e-6:
                    col = cand / cand_norm
                    break
        else:
            col = col / norm
        M[:, j] = col
    return M


def _select_unitary(
    branches: list[np.ndarray],
    *,
    n_index: int,
    n_system: int,
) -> np.ndarray:
    """Return index-controlled SELECT over flag+system branches."""
    sub_dim = 1 << (n_system + 1)
    if n_index == 0:
        return branches[0]
    sel_dim = 1 << n_index
    full_dim = sel_dim * sub_dim
    select = np.zeros((full_dim, full_dim), dtype=complex)
    for idx in range(sel_dim):
        start = idx * sub_dim
        stop = start + sub_dim
        if idx < len(branches):
            select[start:stop, start:stop] = branches[idx]
        else:
            select[start:stop, start:stop] = np.eye(sub_dim, dtype=complex)
    return select


def _occupation_flag_unitary(
    u_mode: np.ndarray,
    *,
    pivot: int,
    n_system: int,
) -> np.ndarray:
    """Return the flag gadget whose top-left block equals ``n[u_mode]``."""
    flag = n_system
    circuit = Circuit(num_qubits=n_system + 1)
    circuit.append(_cnot_gate(control=pivot, target=flag))
    circuit.append(x_gate(flag))
    flag_projector = circuit_unitary(circuit)
    lifted_rotation = np.kron(np.eye(2, dtype=complex), u_mode)
    with np.errstate(all="ignore"):
        return lifted_rotation @ flag_projector @ lifted_rotation.conj().T


def _one_body_dense(L: np.ndarray, n_qubits: int) -> np.ndarray:
    """Return ``sum_pq L_pq a_p^dag a_q`` as a dense JW matrix."""
    n_orb = L.shape[0]
    adag = [jw.jw_a_dagger(p, n_qubits) for p in range(n_orb)]
    a_ = [jw.jw_a(p, n_qubits) for p in range(n_orb)]
    dim = jw.fock_dim(n_qubits)
    O = np.zeros((dim, dim), dtype=complex)
    for p in range(n_orb):
        for q in range(n_orb):
            coef = L[p, q]
            if abs(coef) < 1e-15:
                continue
            O += coef * (adag[p] @ a_[q])
    return 0.5 * (O + O.conj().T)


@dataclass
class CholeskyChannelBlockEncoding:
    """Structured output of the Appendix B.2 Lemma-2 construction."""

    unitary: np.ndarray
    alpha: float
    n_system: int
    n_index: int
    one_body_unitary: np.ndarray
    prep_unitary: np.ndarray
    select_unitary: np.ndarray
    branch_unitaries: tuple[np.ndarray, ...]
    select_branch_unitaries: tuple[np.ndarray, ...]
    eigenvalues: np.ndarray
    orbitals: np.ndarray
    pivots: tuple[int, ...]

    @property
    def n_flag(self) -> int:
        return 1

    @property
    def n_signal(self) -> int:
        return 1

    @property
    def n_one_body_ancilla(self) -> int:
        return self.n_index + self.n_flag

    @property
    def n_ancilla(self) -> int:
        return self.n_one_body_ancilla + self.n_signal

    def one_body_top_left_block(self) -> np.ndarray:
        return _top_left_block(self.one_body_unitary, self.n_system)

    def top_left_block(self) -> np.ndarray:
        return _top_left_block(self.unitary, self.n_system)

    def branch_top_left_block(self, index: int, *, include_phase: bool = False) -> np.ndarray:
        branches = self.select_branch_unitaries if include_phase else self.branch_unitaries
        if not (0 <= index < len(branches)):
            raise IndexError(
                f"branch index {index} out of range for {len(branches)} retained modes"
            )
        return _top_left_block(branches[index], self.n_system)


def build_cholesky_channel_block_encoding(
    L: np.ndarray,
    n_qubits: int | None = None,
    *,
    spectral_tol: float = 1e-12,
) -> CholeskyChannelBlockEncoding:
    """Build the explicit Appendix B.2 Lemma-2 construction.

    The returned unitary acts on

    * ``n_system`` system qubits,
    * ``n_index = ceil(log2 R_mu)`` index qubits,
    * one occupation-flag qubit, and
    * one degree-2 signal qubit.

    Qubit ordering follows the repo's standard ancilla-as-MSB layout:
    system qubits are least significant, then flag, then index, then
    the signal qubit.
    """
    L = np.asarray(L, dtype=complex)
    if L.ndim != 2 or L.shape[0] != L.shape[1]:
        raise ValueError(f"L must be square, got shape {L.shape}")
    if not np.allclose(L, L.conj().T, atol=1e-10):
        raise ValueError("L must be Hermitian")
    n_orb = int(L.shape[0])
    n = n_orb if n_qubits is None else int(n_qubits)
    if n < n_orb:
        raise ValueError(f"n_qubits={n} < L.shape[0]={n_orb}")

    eigvals, eigvecs = np.linalg.eigh(L)
    keep = np.flatnonzero(np.abs(eigvals) > spectral_tol)

    if keep.size == 0:
        alpha = 1.0
        n_index = 0
        prep = np.ones((1, 1), dtype=complex)
        zero_branch = np.kron(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), np.eye(1 << n))
        one_body = zero_branch
        full = degree_two_projector_transform(one_body, n_block_ancilla=1)
        return CholeskyChannelBlockEncoding(
            unitary=full,
            alpha=alpha,
            n_system=n,
            n_index=n_index,
            one_body_unitary=one_body,
            prep_unitary=prep,
            select_unitary=zero_branch,
            branch_unitaries=tuple(),
            select_branch_unitaries=tuple(),
            eigenvalues=np.zeros(0, dtype=float),
            orbitals=np.zeros((0, n_orb), dtype=complex),
            pivots=tuple(),
        )

    retained_vals = eigvals[keep]
    retained_vecs = eigvecs[:, keep]
    order = np.argsort(-np.abs(retained_vals), kind="stable")
    retained_vals = retained_vals[order]
    retained_vecs = retained_vecs[:, order]

    alpha = float(np.sum(np.abs(retained_vals)))
    n_index = int(np.ceil(np.log2(retained_vals.shape[0]))) if retained_vals.shape[0] > 1 else 0

    branch_unitaries: list[np.ndarray] = []
    select_branches: list[np.ndarray] = []
    pivots: list[int] = []
    orbitals: list[np.ndarray] = []
    for lam, vec in zip(retained_vals, retained_vecs.T):
        U_mode, pivot = mode_rotation_unitary(vec, n_qubits=n)
        branch = _occupation_flag_unitary(U_mode, pivot=pivot, n_system=n)
        phase = np.exp(1j * np.angle(lam))
        branch_unitaries.append(branch)
        select_branches.append(phase * branch)
        pivots.append(pivot)
        orbitals.append(np.asarray(vec, dtype=complex))

    prep = _prep_unitary(np.abs(retained_vals), n_index)
    select = _select_unitary(select_branches, n_index=n_index, n_system=n)
    prep_full = np.kron(prep, np.eye(1 << (n + 1), dtype=complex))
    with np.errstate(all="ignore"):
        one_body = prep_full.conj().T @ select @ prep_full
    full = degree_two_projector_transform(one_body, n_block_ancilla=n_index + 1)

    return CholeskyChannelBlockEncoding(
        unitary=full,
        alpha=alpha,
        n_system=n,
        n_index=n_index,
        one_body_unitary=one_body,
        prep_unitary=prep,
        select_unitary=select,
        branch_unitaries=tuple(branch_unitaries),
        select_branch_unitaries=tuple(select_branches),
        eigenvalues=np.asarray(retained_vals, dtype=float),
        orbitals=np.asarray(orbitals, dtype=complex),
        pivots=tuple(pivots),
    )


def hermitian_one_body_block_encoding(
    L: np.ndarray, n_qubits: int | None = None
) -> tuple[np.ndarray, float]:
    """Compatibility helper returning a compact reflection block encoding.

    This dense one-ancilla wrapper is retained for existing callers that
    still expect a fixed-width unitary. The literal Appendix B.2
    construction is exposed by ``build_cholesky_channel_block_encoding``.
    """
    O = _one_body_dense(np.asarray(L, dtype=complex), L.shape[0] if n_qubits is None else n_qubits)
    eigvals = np.linalg.eigvalsh(O)
    alpha = float(max(np.max(np.abs(eigvals)), 1e-16))
    A = O / alpha
    return _reflection_block_encoding(A), alpha


def apply_x_squared_qsvt(W: np.ndarray) -> np.ndarray:
    """Compatibility helper returning the exact transformed top-left block."""
    dim = W.shape[0] // 2
    A = W[:dim, :dim]
    A2 = A @ A
    return 0.5 * (A2 + A2.conj().T)


def x_squared_qsvt_unitary(W: np.ndarray) -> np.ndarray:
    """Compatibility helper returning a compact block encoding of ``A^2``."""
    return _reflection_block_encoding(apply_x_squared_qsvt(W))


def cholesky_channel_block_encoding(
    L: np.ndarray,
    n_qubits: int | None = None,
    *,
    spectral_tol: float = 1e-12,
) -> CholeskyChannelBlockEncoding:
    """Build and return the explicit Lemma-2 channel block encoding."""
    return build_cholesky_channel_block_encoding(
        L,
        n_qubits=n_qubits,
        spectral_tol=spectral_tol,
    )
