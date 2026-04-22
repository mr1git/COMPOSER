"""Ordered circuit of Gate instances, with composition, inverse, and
compile-once structural fingerprints used to verify ASSUMPTION #13.

The circuit is just a thin sequence wrapper. Time order is **first
gate applied first** (i.e., gates[0] acts on the initial state first).
This matches the order they read in source code.
"""
from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from .gate import CircuitOp

__all__ = ["CircuitResourceSummary", "Circuit"]


@dataclass(frozen=True)
class CircuitResourceSummary:
    """Resource/accounting summary derived from a compiled ``Circuit``."""

    num_qubits: int
    gate_count: int
    multi_qubit_gate_count: int
    full_width_gate_count: int
    composite_gate_count: int
    subcircuit_call_count: int
    select_gate_count: int
    state_preparation_gate_count: int
    multiplexed_gate_count: int
    reflection_gate_count: int
    max_gate_arity: int
    gate_count_by_kind: dict[str, int]
    compiled_signature_hash: str


@dataclass
class Circuit:
    num_qubits: int
    gates: list[CircuitOp] = field(default_factory=list)

    def append(self, gate: CircuitOp) -> "Circuit":
        for q in gate.qubits:
            if q >= self.num_qubits:
                raise ValueError(
                    f"gate on qubit {q} but circuit has only {self.num_qubits} qubits"
                )
        self.gates.append(gate)
        return self

    def extend(self, gates: Iterable[CircuitOp]) -> "Circuit":
        for g in gates:
            self.append(g)
        return self

    def compose(self, other: "Circuit") -> "Circuit":
        """Return a new circuit = self then other (other applied after)."""
        if other.num_qubits > self.num_qubits:
            raise ValueError("other has more qubits than self")
        new = Circuit(num_qubits=self.num_qubits, gates=list(self.gates))
        new.gates.extend(other.gates)
        return new

    def inverse(self) -> "Circuit":
        """Return the reverse-order, dagger-of-each-gate circuit."""
        return Circuit(
            num_qubits=self.num_qubits,
            gates=[g.dagger() for g in reversed(self.gates)],
        )

    # -- compile-once topology hash -----------------------------------

    def two_qubit_topology(self) -> list[tuple[str, tuple[int, ...]]]:
        """List of topology ids for all *multi-qubit* gates, in order.

        Single-qubit gates are excluded. In the current dense
        verification path, mask updates can change gate parameters and
        some dense matrices, while the compile-once audit tracks only
        ordered multi-qubit topology ids.
        """
        return [g.topology_id() for g in self.gates if g.num_qubits >= 2]

    def two_qubit_topology_hash(self) -> str:
        """Hex digest of the ordered multi-qubit topology.

        Used by ``tests/test_similarity_sandwich.py`` to verify that
        different masks produce *identical* ordered multi-qubit topology
        ids (ASSUMPTION #13). This is a structural compile-once proxy,
        not a full equality check on every dense gate matrix.
        """
        h = hashlib.sha256()
        for kind, qubits in self.two_qubit_topology():
            h.update(kind.encode("utf-8"))
            h.update(b"|")
            for q in qubits:
                h.update(str(q).encode("utf-8"))
            h.update(b",")
            h.update(b";")
        return h.hexdigest()

    def compiled_signature(self) -> list[tuple[str, tuple[int, ...], tuple[int, int], str, tuple[str, ...]]]:
        """Ordered full-gate signature for compile-once checks.

        Unlike ``two_qubit_topology()``, this includes every gate, its
        ordered qubit support, matrix shape, and implementation mode.
        Parameter values and dense matrix entries are intentionally
        excluded so tests can distinguish fixed compiled structure from
        allowed mask-dependent retuning of PREP amplitudes.
        """
        return [
            (
                (g.kind or g.name),
                g.qubits,
                g.matrix_shape(),
                g.implementation_tag(),
                g.compiled_payload(),
            )
            for g in self.gates
        ]

    def compiled_signature_hash(self) -> str:
        """Hex digest of the ordered full-gate compiled signature."""
        h = hashlib.sha256()
        for kind, qubits, shape, implementation, payload in self.compiled_signature():
            h.update(kind.encode("utf-8"))
            h.update(b"|")
            for q in qubits:
                h.update(str(q).encode("utf-8"))
                h.update(b",")
            h.update(b"|")
            h.update(str(shape[0]).encode("utf-8"))
            h.update(b"x")
            h.update(str(shape[1]).encode("utf-8"))
            h.update(b"|")
            h.update(implementation.encode("utf-8"))
            h.update(b"|")
            for item in payload:
                h.update(item.encode("utf-8"))
                h.update(b",")
            h.update(b";")
        return h.hexdigest()

    def resource_summary(self) -> CircuitResourceSummary:
        """Return accounting data derived from the compiled gate list."""
        kind_counts = Counter((g.kind or g.name) for g in self.gates)
        max_arity = max((g.num_qubits for g in self.gates), default=0)
        return CircuitResourceSummary(
            num_qubits=self.num_qubits,
            gate_count=len(self.gates),
            multi_qubit_gate_count=sum(g.num_qubits >= 2 for g in self.gates),
            full_width_gate_count=sum(g.num_qubits == self.num_qubits for g in self.gates),
            composite_gate_count=sum(g.implementation_tag() != "dense" for g in self.gates),
            subcircuit_call_count=sum(g.implementation_tag() == "subcircuit" for g in self.gates),
            select_gate_count=sum(g.implementation_tag() in {"select", "multiplexed"} for g in self.gates),
            state_preparation_gate_count=sum(
                g.implementation_tag() == "state_prep" for g in self.gates
            ),
            multiplexed_gate_count=sum(g.implementation_tag() == "multiplexed" for g in self.gates),
            reflection_gate_count=sum(g.implementation_tag() == "reflection" for g in self.gates),
            max_gate_arity=max_arity,
            gate_count_by_kind=dict(sorted(kind_counts.items())),
            compiled_signature_hash=self.compiled_signature_hash(),
        )

    def resource_report(self, **kwargs):
        """Return compiled and optional backend resource-estimation views."""
        from .resources import resource_report

        return resource_report(self, **kwargs)

    def __len__(self) -> int:
        return len(self.gates)
