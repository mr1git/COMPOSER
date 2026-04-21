"""Dense statevector / unitary simulator for small systems.

This is the *only* verification engine in the project: every
Lemma / Theorem test builds the Circuit, asks this module to turn it
into a dense 2**n x 2**n matrix (or applies it to a statevector), and
then compares top-left blocks or explicit amplitudes against hand-built
second-quantized references.

The construction path elsewhere in the repo may now retain hierarchical
subcircuit calls and branch-select objects. This simulator remains the
verification backend: when such composite operations are encountered,
their dense matrices are synthesized lazily only here.

Conventions
-----------
* **Qubit ordering.** Qubit 0 is the least-significant tensor factor:
  basis state ``|x_{n-1} ... x_1 x_0>`` has integer index
  ``sum_p x_p * 2**p``. Matches ``utils/fermion.py``.
* **Gate matrix basis.** For a gate with ``qubits = (q_0, q_1, ...,
  q_{k-1})`` and a ``2**k x 2**k`` matrix ``M``, the matrix is written
  in the basis ``|b_{k-1} ... b_1 b_0>`` *of the gate's internal
  qubits*, with internal qubit ``i`` as the bit at position ``2**i`` of
  the matrix's row/column index. Internal qubit ``i`` corresponds to
  system qubit ``qubits[i]``. In particular, a two-qubit gate with
  ``qubits=(0, 1)`` is written in the basis ``|q_1 q_0>`` and a bit at
  position ``2**0`` of the matrix acts on system qubit 0.

Internally we reshape the state vector and gate matrix so that
**tensor axis ``a`` corresponds to qubit ``a``** (LSB-first axes),
then use ``tensordot`` to contract. One final reshape returns a dense
state vector.
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit
from .gate import CircuitOp

__all__ = [
    "apply_gate_to_statevector",
    "apply_gate_to_unitary",
    "statevector",
    "unitary",
    "embed_gate",
]


def _state_to_tensor(state: np.ndarray, n: int) -> np.ndarray:
    """Reshape flat state so axis a = qubit a (LSB-first)."""
    return state.reshape((2,) * n).transpose(tuple(range(n - 1, -1, -1)))


def _tensor_to_state(tensor: np.ndarray, n: int) -> np.ndarray:
    """Inverse of _state_to_tensor."""
    return tensor.transpose(tuple(range(n - 1, -1, -1))).reshape(2**n)


def _gate_to_tensor(matrix: np.ndarray, k: int) -> np.ndarray:
    """Reshape (2**k, 2**k) gate matrix to tensor with axes

        (out_0, out_1, ..., out_{k-1}, in_0, in_1, ..., in_{k-1})

    where internal qubit i sits at axis i (LSB-first on each side).
    """
    # Default reshape: axis 0 = bit (k-1) of the row index (MSB-first).
    # We reverse output and input axis blocks independently.
    t = matrix.reshape((2,) * (2 * k))
    perm = tuple(range(k - 1, -1, -1)) + tuple(range(2 * k - 1, k - 1, -1))
    return t.transpose(perm)


def apply_gate_to_statevector(state: np.ndarray, gate: CircuitOp, n_qubits: int) -> np.ndarray:
    """Apply a gate to a 2**n statevector; returns the new statevector."""
    if state.shape != (2**n_qubits,):
        raise ValueError(f"statevector shape {state.shape} != (2**{n_qubits},)")
    k = gate.num_qubits
    G = _gate_to_tensor(gate.matrix, k)
    psi = _state_to_tensor(state, n_qubits)
    # G input axes (k..2k-1) pair with psi axes (gate.qubits in the same order):
    sys_axes = list(gate.qubits)
    in_axes = list(range(k, 2 * k))
    contracted = np.tensordot(G, psi, axes=(in_axes, sys_axes))
    # After tensordot, the first k axes of contracted are G's output axes
    # (internal qubits 0..k-1 = system qubits gate.qubits[0..k-1]),
    # followed by psi axes in their original order excluding sys_axes.
    remaining = [q for q in range(n_qubits) if q not in gate.qubits]
    axis_to_sys = list(gate.qubits) + remaining
    perm = [axis_to_sys.index(q) for q in range(n_qubits)]
    result_tensor = contracted.transpose(perm)
    return _tensor_to_state(result_tensor, n_qubits)


def apply_gate_to_unitary(u: np.ndarray, gate: CircuitOp, n_qubits: int) -> np.ndarray:
    """Left-multiply u by the embedded gate: return G @ u."""
    dim = 2**n_qubits
    if u.shape != (dim, dim):
        raise ValueError(f"unitary shape {u.shape} != ({dim}, {dim})")
    out = np.empty_like(u)
    for col in range(dim):
        out[:, col] = apply_gate_to_statevector(u[:, col], gate, n_qubits)
    return out


def statevector(circuit: Circuit, init: np.ndarray | None = None) -> np.ndarray:
    """Simulate the circuit on |0...0> (or an explicit init state)."""
    n = circuit.num_qubits
    if init is None:
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0
    else:
        if init.shape != (2**n,):
            raise ValueError(f"init shape {init.shape} != (2**{n},)")
        psi = init.astype(complex, copy=True)
    for g in circuit.gates:
        psi = apply_gate_to_statevector(psi, g, n)
    return psi


def unitary(circuit: Circuit) -> np.ndarray:
    """Dense 2**n x 2**n unitary realized by the circuit."""
    n = circuit.num_qubits
    dim = 2**n
    u = np.eye(dim, dtype=complex)
    for g in circuit.gates:
        u = apply_gate_to_unitary(u, g, n)
    return u


def embed_gate(gate: Gate, n_qubits: int) -> np.ndarray:
    """Return the 2**n x 2**n embedding of ``gate``, identity elsewhere."""
    c = Circuit(num_qubits=n_qubits)
    c.append(gate)
    return unitary(c)
