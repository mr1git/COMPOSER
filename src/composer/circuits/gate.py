"""Lightweight Gate dataclass.

A gate is an (ordered) list of target qubits plus the dense unitary
matrix that acts on them. We do not try to abstract over a gate set;
callers build the matrix directly (for primitives) or compose existing
gates via ``circuit.Circuit``. The simulator works with these dense
matrices via tensor contraction.

The qubit order convention matches ``utils/fermion.py``:

* ``qubits = [q0, q1, ..., q_{k-1}]`` means the gate's matrix acts on
  the tensor factor indexed by ``q0`` (LSB), then ``q1``, etc.
* Concretely, if ``qubits = [p, q]`` with ``p < q``, the gate matrix
  ``G`` (4x4) has the basis order ``|q_q q_p> = |00>, |01>, |10>, |11>``
  where ``q_p`` is the least-significant among the two. This matches
  the LSB-first convention used elsewhere.

Two-qubit gates may be called ``two_qubit_topology_id`` for the
compile-once hash check: this returns a canonical tuple identifying
the gate's qubit pair regardless of rotation angles, so that two
circuits that differ only in rotation parameters have the same
topology hash (see ``circuit.Circuit.two_qubit_topology_hash``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from .circuit import Circuit

__all__ = ["Gate", "CircuitCall", "SelectGate", "CircuitOp"]


@dataclass
class Gate:
    name: str
    qubits: tuple[int, ...]
    matrix: np.ndarray
    # Optional opaque payload identifying the gate's *kind* for the
    # compile-once topology hash. If None, ``name`` is used.
    kind: str | None = field(default=None)

    def __post_init__(self) -> None:
        if not isinstance(self.qubits, tuple):
            self.qubits = tuple(self.qubits)
        for q in self.qubits:
            if q < 0:
                raise ValueError(f"negative qubit index: {q}")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"repeated qubit in {self.qubits}")
        expected = 2 ** len(self.qubits)
        if self.matrix.shape != (expected, expected):
            raise ValueError(
                f"matrix shape {self.matrix.shape} does not match "
                f"{len(self.qubits)} qubits (need {expected} x {expected})"
            )

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    def topology_id(self) -> tuple[str, tuple[int, ...]]:
        """Identifier for compile-once topology hashing.

        Two gates with the same ``(kind, qubits)`` share a topology id
        regardless of their rotation angles; in particular, a mask
        update that only changes single-qubit Z/RY angles keeps the
        hash invariant.
        """
        return (self.kind or self.name, self.qubits)

    def matrix_shape(self) -> tuple[int, int]:
        return self.matrix.shape

    def implementation_tag(self) -> str:
        return "dense"

    def compiled_payload(self) -> tuple[str, ...]:
        return ()

    def dagger(self) -> "Gate":
        return Gate(
            name=self.name + "^dag",
            qubits=self.qubits,
            matrix=self.matrix.conj().T,
            kind=self.kind,
        )


@dataclass
class CircuitCall:
    """Hierarchical call to a compiled subcircuit.

    This keeps the construction path structural: higher-level oracles
    can refer to a compiled child circuit directly rather than
    materializing its full unitary during assembly.
    """

    name: str
    qubits: tuple[int, ...]
    subcircuit: "Circuit"
    kind: str | None = field(default=None)
    _matrix_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qubits, tuple):
            self.qubits = tuple(self.qubits)
        for q in self.qubits:
            if q < 0:
                raise ValueError(f"negative qubit index: {q}")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"repeated qubit in {self.qubits}")
        if self.subcircuit.num_qubits != len(self.qubits):
            raise ValueError(
                f"subcircuit width {self.subcircuit.num_qubits} does not match "
                f"call support {len(self.qubits)}"
            )

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix_cache is None:
            from .simulator import unitary

            self._matrix_cache = unitary(self.subcircuit)
        return self._matrix_cache

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    def topology_id(self) -> tuple[str, tuple[int, ...]]:
        return (self.kind or self.name, self.qubits)

    def matrix_shape(self) -> tuple[int, int]:
        dim = 2 ** self.num_qubits
        return (dim, dim)

    def implementation_tag(self) -> str:
        return "subcircuit"

    def compiled_payload(self) -> tuple[str, ...]:
        return (self.subcircuit.compiled_signature_hash(),)

    def dagger(self) -> "CircuitCall":
        return CircuitCall(
            name=self.name + "^dag",
            qubits=self.qubits,
            subcircuit=self.subcircuit.inverse(),
            kind=self.kind,
        )


@dataclass
class SelectGate:
    """Two-branch multiplexor over compiled child circuits.

    ``qubits`` must end with the selector/control qubit. The zero and
    one branches act on the remaining leading qubits, in that order.
    """

    name: str
    qubits: tuple[int, ...]
    zero_circuit: "Circuit"
    one_circuit: "Circuit"
    kind: str | None = field(default=None)
    _matrix_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qubits, tuple):
            self.qubits = tuple(self.qubits)
        for q in self.qubits:
            if q < 0:
                raise ValueError(f"negative qubit index: {q}")
        if len(set(self.qubits)) != len(self.qubits):
            raise ValueError(f"repeated qubit in {self.qubits}")
        target_width = len(self.qubits) - 1
        if self.zero_circuit.num_qubits != target_width:
            raise ValueError(
                f"zero branch width {self.zero_circuit.num_qubits} does not match "
                f"target width {target_width}"
            )
        if self.one_circuit.num_qubits != target_width:
            raise ValueError(
                f"one branch width {self.one_circuit.num_qubits} does not match "
                f"target width {target_width}"
            )

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix_cache is None:
            from .simulator import unitary

            zero_u = unitary(self.zero_circuit)
            one_u = unitary(self.one_circuit)
            self._matrix_cache = np.block(
                [
                    [zero_u, np.zeros_like(zero_u)],
                    [np.zeros_like(one_u), one_u],
                ]
            )
        return self._matrix_cache

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def control_qubit(self) -> int:
        return self.qubits[-1]

    @property
    def target_qubits(self) -> tuple[int, ...]:
        return self.qubits[:-1]

    def topology_id(self) -> tuple[str, tuple[int, ...]]:
        return (self.kind or self.name, self.qubits)

    def matrix_shape(self) -> tuple[int, int]:
        dim = 2 ** self.num_qubits
        return (dim, dim)

    def implementation_tag(self) -> str:
        return "select"

    def compiled_payload(self) -> tuple[str, ...]:
        return (
            self.zero_circuit.compiled_signature_hash(),
            self.one_circuit.compiled_signature_hash(),
        )

    def dagger(self) -> "SelectGate":
        return SelectGate(
            name=self.name + "^dag",
            qubits=self.qubits,
            zero_circuit=self.zero_circuit.inverse(),
            one_circuit=self.one_circuit.inverse(),
            kind=self.kind,
        )


CircuitOp = Union[Gate, CircuitCall, SelectGate]


def _default_repr(qubits: Sequence[int]) -> str:
    return "(" + ", ".join(str(q) for q in qubits) + ")"
