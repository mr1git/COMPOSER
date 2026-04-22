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
from ..circuits.gate import (
    AncillaZeroReflectionGate,
    CircuitCall,
    Gate,
    MultiplexedGate,
    SelectGate,
    StatePreparationGate,
)
from ..circuits.simulator import unitary as circuit_unitary
from ..ladders.one_electron import build_number_conserving_ladder, x_gate
from ..qsp.qsvt_poly import degree_two_projector_transform
from ..utils import fermion as jw

__all__ = [
    "CholeskyChannelBlockEncoding",
    "HermitianOneBodyBlockEncoding",
    "apply_x_squared_qsvt",
    "build_hermitian_one_body_block_encoding",
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


def _identity_circuit(width: int) -> Circuit:
    return Circuit(num_qubits=width)


def _hadamard_gate(qubit: int) -> Gate:
    return Gate(
        name="H",
        qubits=(qubit,),
        matrix=np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0),
        kind="H",
    )


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


def _occupation_flag_circuit(
    orbital: np.ndarray,
    *,
    pivot: int,
    n_system: int,
) -> Circuit:
    """Return the explicit rotated-mode occupation-flag circuit."""
    flag = n_system
    ladder = build_number_conserving_ladder(orbital, n_qubits=n_system, pivot=pivot)
    circuit = Circuit(num_qubits=n_system + 1)
    circuit.append(
        CircuitCall(
            name="U_mode^dag",
            qubits=tuple(range(n_system)),
            subcircuit=ladder.inverse(),
            kind="U_mode_rotation",
        )
    )
    circuit.append(_cnot_gate(control=pivot, target=flag))
    circuit.append(x_gate(flag))
    circuit.append(
        CircuitCall(
            name="U_mode",
            qubits=tuple(range(n_system)),
            subcircuit=ladder,
            kind="U_mode_rotation",
        )
    )
    return circuit


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


def _degree_two_projector_transform_circuit(
    block_circuit: Circuit,
    *,
    n_block_ancilla: int,
    n_system: int,
) -> Circuit:
    """Structural counterpart of ``degree_two_projector_transform``."""
    if block_circuit.num_qubits != n_system + n_block_ancilla:
        raise ValueError("block_circuit width must equal system plus block ancillas")
    signal = block_circuit.num_qubits
    block_ancillas = tuple(range(n_system, n_system + n_block_ancilla))
    reflection_branch = Circuit(num_qubits=n_block_ancilla)
    reflection_branch.append(
        AncillaZeroReflectionGate(
            name="REFLECT_block_zero",
            qubits=tuple(range(n_block_ancilla)),
            system_width=0,
            kind="REFLECT_block_zero",
        )
    )

    circuit = Circuit(num_qubits=block_circuit.num_qubits + 1)
    circuit.append(
        CircuitCall(
            name="U_one_body",
            qubits=tuple(range(block_circuit.num_qubits)),
            subcircuit=block_circuit,
            kind="U_one_body",
        )
    )
    circuit.append(_hadamard_gate(signal))
    circuit.append(
        SelectGate(
            name="SELECT_projector_zero",
            qubits=block_ancillas + (signal,),
            zero_circuit=_identity_circuit(n_block_ancilla),
            one_circuit=reflection_branch,
            kind="SELECT_projector_zero",
        )
    )
    circuit.append(_hadamard_gate(signal))
    circuit.append(
        CircuitCall(
            name="U_one_body",
            qubits=tuple(range(block_circuit.num_qubits)),
            subcircuit=block_circuit,
            kind="U_one_body",
        )
    )
    return circuit


@dataclass
class CholeskyChannelBlockEncoding:
    """Structured output of the Appendix B.2 Lemma-2 construction."""

    circuit: Circuit
    unitary: np.ndarray
    alpha: float
    n_system: int
    n_index: int
    one_body_circuit: Circuit
    one_body_unitary: np.ndarray
    select_circuit: Circuit
    prep_unitary: np.ndarray
    select_unitary: np.ndarray
    branch_circuits: tuple[Circuit, ...]
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


@dataclass
class HermitianOneBodyBlockEncoding:
    """Explicit PREP-SELECT-PREP† encoding of ``O / Gamma``."""

    circuit: Circuit
    unitary: np.ndarray
    alpha: float
    n_system: int
    n_index: int
    select_circuit: Circuit
    prep_unitary: np.ndarray
    select_unitary: np.ndarray
    branch_circuits: tuple[Circuit, ...]
    branch_unitaries: tuple[np.ndarray, ...]
    select_branch_unitaries: tuple[np.ndarray, ...]
    eigenvalues: np.ndarray
    orbitals: np.ndarray
    pivots: tuple[int, ...]

    @property
    def n_flag(self) -> int:
        return 1

    @property
    def n_ancilla(self) -> int:
        return self.n_index + self.n_flag

    def top_left_block(self) -> np.ndarray:
        return _top_left_block(self.unitary, self.n_system)

    def branch_top_left_block(self, index: int, *, include_phase: bool = False) -> np.ndarray:
        branches = self.select_branch_unitaries if include_phase else self.branch_unitaries
        if not (0 <= index < len(branches)):
            raise IndexError(
                f"branch index {index} out of range for {len(branches)} retained modes"
            )
        return _top_left_block(branches[index], self.n_system)


def build_hermitian_one_body_block_encoding(
    L: np.ndarray,
    n_qubits: int | None = None,
    *,
    spectral_tol: float = 1e-12,
) -> HermitianOneBodyBlockEncoding:
    """Build the explicit rotated-mode PREP-SELECT-PREP† encoding of ``O / Gamma``."""
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
        circuit = Circuit(num_qubits=n + 1)
        circuit.append(x_gate(n))
        unitary = circuit_unitary(circuit)
        return HermitianOneBodyBlockEncoding(
            circuit=circuit,
            unitary=unitary,
            alpha=1.0,
            n_system=n,
            n_index=0,
            select_circuit=circuit,
            prep_unitary=np.ones((1, 1), dtype=complex),
            select_unitary=unitary,
            branch_circuits=tuple(),
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
    prep_amplitudes = np.zeros(1 << n_index, dtype=complex)
    prep_amplitudes[: retained_vals.shape[0]] = np.sqrt(np.abs(retained_vals) / alpha)

    branch_circuits: list[Circuit] = []
    branch_unitaries: list[np.ndarray] = []
    select_branch_unitaries: list[np.ndarray] = []
    branch_phases: list[complex] = []
    pivots: list[int] = []
    orbitals: list[np.ndarray] = []
    for lam, vec in zip(retained_vals, retained_vecs.T):
        pivot = int(np.argmax(np.abs(vec)))
        branch_circuit = _occupation_flag_circuit(vec, pivot=pivot, n_system=n)
        branch_unitary = circuit_unitary(branch_circuit)
        phase = np.exp(1j * np.angle(lam))
        branch_circuits.append(branch_circuit)
        branch_unitaries.append(branch_unitary)
        select_branch_unitaries.append(phase * branch_unitary)
        branch_phases.append(phase)
        pivots.append(pivot)
        orbitals.append(np.asarray(vec, dtype=complex))

    selector_qubits = tuple(range(n + 1, n + 1 + n_index))
    select_circuit = Circuit(num_qubits=n + 1 + n_index)
    select_circuit.append(
        MultiplexedGate(
            name="SELECT_O",
            qubits=tuple(range(n + 1 + n_index)),
            selector_width=n_index,
            branch_circuits=tuple(branch_circuits),
            default_circuit=_identity_circuit(n + 1),
            branch_phases=tuple(branch_phases),
            kind="SELECT_O",
        )
    )

    circuit = Circuit(num_qubits=n + 1 + n_index)
    if n_index > 0:
        circuit.append(
            StatePreparationGate(
                name="PREP_O",
                qubits=selector_qubits,
                amplitudes=prep_amplitudes,
                kind="PREP_O",
            )
        )
    circuit.append(
        CircuitCall(
            name="SELECT_O",
            qubits=tuple(range(n + 1 + n_index)),
            subcircuit=select_circuit,
            kind="SELECT_O",
        )
    )
    if n_index > 0:
        circuit.append(
            StatePreparationGate(
                name="PREP_O^dag",
                qubits=selector_qubits,
                amplitudes=prep_amplitudes,
                kind="PREP_O",
                adjoint=True,
            )
        )

    return HermitianOneBodyBlockEncoding(
        circuit=circuit,
        unitary=circuit_unitary(circuit),
        alpha=alpha,
        n_system=n,
        n_index=n_index,
        select_circuit=select_circuit,
        prep_unitary=_prep_unitary(np.abs(retained_vals), n_index),
        select_unitary=circuit_unitary(select_circuit),
        branch_circuits=tuple(branch_circuits),
        branch_unitaries=tuple(branch_unitaries),
        select_branch_unitaries=tuple(select_branch_unitaries),
        eigenvalues=np.asarray(retained_vals, dtype=float),
        orbitals=np.asarray(orbitals, dtype=complex),
        pivots=tuple(pivots),
    )


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
    one_body_be = build_hermitian_one_body_block_encoding(
        L,
        n_qubits=n_qubits,
        spectral_tol=spectral_tol,
    )
    full_circuit = _degree_two_projector_transform_circuit(
        one_body_be.circuit,
        n_block_ancilla=one_body_be.n_ancilla,
        n_system=one_body_be.n_system,
    )
    full = circuit_unitary(full_circuit)
    return CholeskyChannelBlockEncoding(
        circuit=full_circuit,
        unitary=full,
        alpha=one_body_be.alpha,
        n_system=one_body_be.n_system,
        n_index=one_body_be.n_index,
        one_body_circuit=one_body_be.circuit,
        one_body_unitary=one_body_be.unitary,
        select_circuit=one_body_be.select_circuit,
        prep_unitary=one_body_be.prep_unitary,
        select_unitary=one_body_be.select_unitary,
        branch_circuits=one_body_be.branch_circuits,
        branch_unitaries=one_body_be.branch_unitaries,
        select_branch_unitaries=one_body_be.select_branch_unitaries,
        eigenvalues=one_body_be.eigenvalues,
        orbitals=one_body_be.orbitals,
        pivots=one_body_be.pivots,
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
