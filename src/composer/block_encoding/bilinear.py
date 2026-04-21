"""Lemma 1 (Eq 33): restricted-subspace block encoding of a bilinear dyad.

Given ``L = lambda a^dag[u] a[v]`` with ``|u| = |v| = 1`` and
``lambda >= 0`` (ASSUMPTION #4: phase absorbed into ``u``), build a
circuit ``W`` on ``n + 1`` qubits — qubits ``0..n-1`` are the system,
qubit ``n`` is the single ancilla — whose top-left ``2**n x 2**n``
block (ancilla projected onto ``|0>``) matches ``L / alpha`` only
after restricting to the single-excitation subspace ``H_{N=1}``.

Derivation
----------
Let ``U_u`` be a *number-conserving* orbital rotation with
``U_u |e_0> = |u>``, built from the unitary whose first column is
``u`` (``orbital_rotation_unitary`` in ``ladders/two_electron.py``).
Similarly ``U_v``.  Let ``P_{q_0=1} = |1><1|_{qubit 0}`` be the
projector onto qubit 0 being 1; on ``H_{N=1}`` this equals
``|e_0><e_0|``. Then

    U_u P_{q_0=1} U_v^dag |_{H_{N=1}}  =  U_u |e_0><e_0| U_v^dag |_{H_{N=1}}
                                       =  |u><v|.

The ``P_{q_0=1}`` projector is implemented with a single ancilla and
one CNOT: take ``(X_a ⊗ I) CNOT_{q_0 -> a}`` and project the ancilla
onto ``|0>``. The full circuit is therefore

    W  =  (I ⊗ U_u) (X_a ⊗ I) CNOT_{q_0 -> a} (I ⊗ U_v^dag)

and ``(<0|_a ⊗ I) W (|0>_a ⊗ I) = U_u P_{q_0=1} U_v^dag``.

This is not the literal Appendix B.1 vacuum-projector construction.
Instead, it is an equivalent number-conserving realization of the same
restricted ``H_{N=1}`` dyad identity. Using the *full* state-prep
ladder ``build_ladder(u)`` here would be incorrect, because that
ladder takes ``|vac>`` to ``|u>`` and does not act as a
number-conserving rotation on ``H_{N=1}``. The rotation factor of the
ladder (Givens chain + Rz) *without* the leading ``X_pivot`` is what is
needed. For clarity we bypass the ladder and build the rotation
``U_u`` directly by extending ``u`` to a unitary basis.

The ancilla lives at qubit index ``n`` (MSB).
"""
from __future__ import annotations

import numpy as np

from ..circuits.circuit import Circuit
from ..circuits.gate import Gate
from ..circuits.simulator import unitary as circuit_unitary
from ..ladders.two_electron import orbital_rotation_unitary
from ..operators.rank_one import BilinearRankOne

__all__ = [
    "build_bilinear_block_encoding",
    "BilinearBlockEncoding",
    "orbital_rotation_first_column",
    "cnot_gate",
]


def orbital_rotation_first_column(u: np.ndarray) -> np.ndarray:
    """Return the ``2**n x 2**n`` number-conserving unitary ``U_u`` with
    ``U_u |e_0> = |u>``.

    We extend ``u`` to an orthonormal basis by QR on ``[u | I]`` so that
    ``V[:, 0] = u``, then exponentiate the matrix logarithm of ``V`` via
    ``orbital_rotation_unitary``.
    """
    u = np.asarray(u, dtype=complex).ravel()
    n = u.shape[0]
    if not np.isclose(np.linalg.norm(u), 1.0, atol=1e-10):
        raise ValueError("u must be unit norm")
    aug = np.column_stack([u.reshape(-1, 1), np.eye(n, dtype=complex)])
    # Gram-Schmidt: keep u as column 0, orthogonalize standard basis against it.
    V = np.zeros((n, n), dtype=complex)
    V[:, 0] = u
    for j in range(1, n):
        col = aug[:, j]  # e_{j-1}
        for k in range(j):
            col = col - np.vdot(V[:, k], col) * V[:, k]
        norm = np.linalg.norm(col)
        if norm < 1e-10:
            # pick an alternative standard basis vector orthogonal to span
            for e_idx in range(n):
                cand = np.zeros(n, dtype=complex)
                cand[e_idx] = 1.0
                for k in range(j):
                    cand = cand - np.vdot(V[:, k], cand) * V[:, k]
                if np.linalg.norm(cand) > 1e-6:
                    col = cand / np.linalg.norm(cand)
                    break
        else:
            col = col / norm
        V[:, j] = col
    assert np.allclose(V.conj().T @ V, np.eye(n), atol=1e-8)
    return orbital_rotation_unitary(V)


def _cnot_matrix_lsb_first(qubits: tuple[int, int], control: int, target: int) -> np.ndarray:
    """4x4 CNOT matrix written in the LSB-first basis of ``qubits``.

    ``qubits`` is the tuple passed to the Gate (LSB first — qubits[0] is
    the least-significant internal bit). ``control``/``target`` are
    system qubit indices (must both appear in ``qubits``).
    """
    if set(qubits) != {control, target}:
        raise ValueError("qubits must be exactly (control, target)")
    # internal bit of control / target:
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


def cnot_gate(control: int, target: int) -> Gate:
    """CNOT gate wrapper using the LSB-first Gate convention."""
    if control == target:
        raise ValueError("CNOT control and target must differ")
    lo, hi = (control, target) if control < target else (target, control)
    qubits = (lo, hi)
    mat = _cnot_matrix_lsb_first(qubits, control, target)
    return Gate(
        name=f"CNOT({control}->{target})",
        qubits=qubits,
        matrix=mat,
        kind=f"CNOT({control}->{target})",
    )


def _x_gate(qubit: int) -> Gate:
    return Gate(
        name="X",
        qubits=(qubit,),
        matrix=np.array([[0, 1], [1, 0]], dtype=complex),
        kind="X",
    )


class BilinearBlockEncoding:
    """Container for the Lemma 1 adaptor output."""

    def __init__(self, circuit: Circuit, alpha: float, n_system: int, ancilla: int) -> None:
        self.circuit = circuit
        self.alpha = alpha
        self.n_system = n_system
        self.ancilla = ancilla

    def top_left_block(self) -> np.ndarray:
        """Return the ``2**n x 2**n`` top-left (ancilla = |0>) block of ``W``."""
        W = circuit_unitary(self.circuit)
        dim_sys = 2**self.n_system
        # ancilla = qubit n_system (MSB). Basis index = anc * dim_sys + sys.
        return W[:dim_sys, :dim_sys]


def build_bilinear_block_encoding(L: BilinearRankOne) -> BilinearBlockEncoding:
    """Build the Lemma 1 circuit for a ``BilinearRankOne`` operator.

    ASSUMPTION #4: ``alpha = coeff`` (smallest admissible). Phases are
    already absorbed into ``u``, ``v`` by the ``BilinearRankOne``
    constructor convention (``coeff`` is real non-negative).
    """
    u = L.u
    v = L.v
    alpha = float(L.coeff)
    n = L.n_orbitals
    ancilla = n  # single ancilla at MSB

    # Build number-conserving rotations U_u, U_v with U_u|e_0> = |u>.
    U_u = orbital_rotation_first_column(u)
    U_v = orbital_rotation_first_column(v)

    U_u_gate = Gate(
        name="U_u",
        qubits=tuple(range(n)),
        matrix=U_u,
        kind="U_u_rotation",
    )
    U_v_dag_gate = Gate(
        name="U_v^dag",
        qubits=tuple(range(n)),
        matrix=U_v.conj().T,
        kind="U_v_rotation_dag",
    )

    circuit = Circuit(num_qubits=n + 1)
    circuit.append(U_v_dag_gate)
    circuit.append(cnot_gate(control=0, target=ancilla))
    circuit.append(_x_gate(ancilla))
    circuit.append(U_u_gate)

    return BilinearBlockEncoding(circuit=circuit, alpha=alpha, n_system=n, ancilla=ancilla)
