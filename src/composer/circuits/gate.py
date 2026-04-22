"""Circuit-operation dataclasses.

The verification backend remains dense, but the main circuit
representation now distinguishes between:

* raw dense gates,
* hierarchical compiled subcircuit calls,
* two-branch and multi-branch selector multiplexors,
* synthesized state-preparation primitives, and
* synthesized ancilla-zero reflections.

Composite operations expose a ``matrix`` property so the dense
simulator can still verify them numerically, but their primary
representation is structural rather than one opaque dense blob.

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

__all__ = [
    "Gate",
    "CircuitCall",
    "SelectGate",
    "StatePreparationGate",
    "MultiplexedGate",
    "AncillaZeroReflectionGate",
    "CircuitOp",
]


def _validate_qubits(qubits: tuple[int, ...]) -> None:
    for q in qubits:
        if q < 0:
            raise ValueError(f"negative qubit index: {q}")
    if len(set(qubits)) != len(qubits):
        raise ValueError(f"repeated qubit in {qubits}")


def _complete_state_preparation_unitary(amplitudes: np.ndarray) -> np.ndarray:
    """Extend a normalized state vector to a unitary with that first column."""
    amps = np.asarray(amplitudes, dtype=complex).reshape(-1)
    dim = amps.shape[0]
    U = np.zeros((dim, dim), dtype=complex)
    U[:, 0] = amps
    for j in range(1, dim):
        col = np.zeros(dim, dtype=complex)
        col[j] = 1.0
        for k in range(j):
            col = col - np.vdot(U[:, k], col) * U[:, k]
        norm = np.linalg.norm(col)
        if norm < 1e-12:
            for alt in range(dim):
                cand = np.zeros(dim, dtype=complex)
                cand[alt] = 1.0
                for k in range(j):
                    cand = cand - np.vdot(U[:, k], cand) * U[:, k]
                cand_norm = np.linalg.norm(cand)
                if cand_norm > 1e-6:
                    col = cand / cand_norm
                    break
        else:
            col = col / norm
        U[:, j] = col
    return U


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
        _validate_qubits(self.qubits)
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
        _validate_qubits(self.qubits)
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
        _validate_qubits(self.qubits)
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


@dataclass
class StatePreparationGate:
    """Synthesized state preparation with a fixed prepared state."""

    name: str
    qubits: tuple[int, ...]
    amplitudes: np.ndarray
    kind: str | None = field(default=None)
    adjoint: bool = field(default=False)
    _matrix_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qubits, tuple):
            self.qubits = tuple(self.qubits)
        _validate_qubits(self.qubits)
        self.amplitudes = np.asarray(self.amplitudes, dtype=complex).reshape(-1)
        expected = 2 ** len(self.qubits)
        if self.amplitudes.shape != (expected,):
            raise ValueError(
                f"amplitudes shape {self.amplitudes.shape} does not match "
                f"{len(self.qubits)} qubits (need length {expected})"
            )
        norm = np.linalg.norm(self.amplitudes)
        if not np.isclose(norm, 1.0, atol=1e-10):
            raise ValueError(f"state-preparation amplitudes must be normalized, got norm {norm}")

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix_cache is None:
            base = _complete_state_preparation_unitary(self.amplitudes)
            self._matrix_cache = base.conj().T if self.adjoint else base
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
        return "state_prep"

    def compiled_payload(self) -> tuple[str, ...]:
        return ("gram_schmidt",)

    def dagger(self) -> "StatePreparationGate":
        return StatePreparationGate(
            name=self.name + "^dag",
            qubits=self.qubits,
            amplitudes=self.amplitudes,
            kind=self.kind,
            adjoint=not self.adjoint,
        )


@dataclass
class MultiplexedGate:
    """Binary-selector multiplexor over compiled child circuits.

    ``qubits`` is ordered as ``target/workspace`` first and selector
    qubits last. Selector value ``s`` applies branch ``s`` when it is
    present; otherwise ``default_circuit`` is used, or identity when no
    default is provided.
    """

    name: str
    qubits: tuple[int, ...]
    selector_width: int
    branch_circuits: tuple["Circuit", ...]
    default_circuit: "Circuit | None" = field(default=None)
    branch_phases: tuple[complex, ...] = field(default_factory=tuple)
    kind: str | None = field(default=None)
    _matrix_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qubits, tuple):
            self.qubits = tuple(self.qubits)
        _validate_qubits(self.qubits)
        if self.selector_width < 0:
            raise ValueError("selector_width must be non-negative")
        if self.selector_width > len(self.qubits):
            raise ValueError("selector_width cannot exceed the gate width")
        self.branch_circuits = tuple(self.branch_circuits)
        target_width = len(self.qubits) - self.selector_width
        for branch in self.branch_circuits:
            if branch.num_qubits != target_width:
                raise ValueError(
                    f"branch width {branch.num_qubits} does not match target width {target_width}"
                )
        if self.default_circuit is not None and self.default_circuit.num_qubits != target_width:
            raise ValueError(
                "default branch width "
                f"{self.default_circuit.num_qubits} does not match target width {target_width}"
            )
        if not self.branch_phases:
            self.branch_phases = tuple(1.0 + 0.0j for _ in self.branch_circuits)
        elif len(self.branch_phases) != len(self.branch_circuits):
            raise ValueError("branch_phases must match the number of branches")
        else:
            self.branch_phases = tuple(complex(phase) for phase in self.branch_phases)
            for phase in self.branch_phases:
                if not np.isclose(abs(phase), 1.0, atol=1e-10):
                    raise ValueError(f"branch phase must have unit modulus, got {phase}")

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix_cache is None:
            from .simulator import unitary

            target_width = len(self.qubits) - self.selector_width
            if self.selector_width == 0:
                branch = self.branch_circuits[0] if self.branch_circuits else self.default_circuit
                if branch is None:
                    self._matrix_cache = np.eye(2**target_width, dtype=complex)
                else:
                    phase = self.branch_phases[0] if self.branch_circuits else 1.0 + 0.0j
                    self._matrix_cache = phase * unitary(branch)
                return self._matrix_cache

            sel_dim = 2**self.selector_width
            sub_dim = 2**target_width
            full = np.zeros((sel_dim * sub_dim, sel_dim * sub_dim), dtype=complex)
            identity = np.eye(sub_dim, dtype=complex)
            default_u = identity if self.default_circuit is None else unitary(self.default_circuit)
            for s in range(sel_dim):
                if s < len(self.branch_circuits):
                    block = self.branch_phases[s] * unitary(self.branch_circuits[s])
                else:
                    block = default_u
                start = s * sub_dim
                stop = start + sub_dim
                full[start:stop, start:stop] = block
            self._matrix_cache = full
        return self._matrix_cache

    @property
    def num_qubits(self) -> int:
        return len(self.qubits)

    @property
    def target_qubits(self) -> tuple[int, ...]:
        return self.qubits[: self.num_qubits - self.selector_width]

    @property
    def selector_qubits(self) -> tuple[int, ...]:
        return self.qubits[self.num_qubits - self.selector_width :]

    def topology_id(self) -> tuple[str, tuple[int, ...]]:
        return (self.kind or self.name, self.qubits)

    def matrix_shape(self) -> tuple[int, int]:
        dim = 2 ** self.num_qubits
        return (dim, dim)

    def implementation_tag(self) -> str:
        return "multiplexed"

    def compiled_payload(self) -> tuple[str, ...]:
        payload = [
            f"selector_width={self.selector_width}",
            f"branch_count={len(self.branch_circuits)}",
        ]
        payload.extend(
            f"branch[{idx}]={branch.compiled_signature_hash()}"
            for idx, branch in enumerate(self.branch_circuits)
        )
        if self.default_circuit is not None:
            payload.append(f"default={self.default_circuit.compiled_signature_hash()}")
        return tuple(payload)

    def dagger(self) -> "MultiplexedGate":
        return MultiplexedGate(
            name=self.name + "^dag",
            qubits=self.qubits,
            selector_width=self.selector_width,
            branch_circuits=tuple(branch.inverse() for branch in self.branch_circuits),
            default_circuit=None if self.default_circuit is None else self.default_circuit.inverse(),
            branch_phases=tuple(np.conjugate(phase) for phase in self.branch_phases),
            kind=self.kind,
        )


@dataclass
class AncillaZeroReflectionGate:
    """Reflection ``2|0_anc><0_anc| - I`` over a fixed ancilla partition."""

    name: str
    qubits: tuple[int, ...]
    system_width: int
    kind: str | None = field(default=None)
    _matrix_cache: np.ndarray | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.qubits, tuple):
            self.qubits = tuple(self.qubits)
        _validate_qubits(self.qubits)
        if self.system_width < 0 or self.system_width > len(self.qubits):
            raise ValueError("system_width must lie within the gate width")
        if self.system_width == len(self.qubits):
            raise ValueError("ancilla-zero reflection requires at least one ancilla qubit")

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix_cache is None:
            dim_sys = 2**self.system_width
            dim_total = 2 ** self.num_qubits
            diag = -np.ones(dim_total, dtype=complex)
            diag[:dim_sys] = 1.0
            self._matrix_cache = np.diag(diag)
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
        return "reflection"

    def compiled_payload(self) -> tuple[str, ...]:
        return (f"system_width={self.system_width}",)

    def dagger(self) -> "AncillaZeroReflectionGate":
        return AncillaZeroReflectionGate(
            name=self.name + "^dag",
            qubits=self.qubits,
            system_width=self.system_width,
            kind=self.kind,
        )


CircuitOp = Union[
    Gate,
    CircuitCall,
    SelectGate,
    StatePreparationGate,
    MultiplexedGate,
    AncillaZeroReflectionGate,
]


def _default_repr(qubits: Sequence[int]) -> str:
    return "(" + ", ".join(str(q) for q in qubits) + ")"
