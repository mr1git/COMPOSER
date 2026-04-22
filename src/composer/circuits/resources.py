"""Resource-estimation views for compiled and optionally exported circuits.

This keeps three layers distinct:

* ``Circuit.resource_summary()`` remains the existing logical summary of
  the top-level compiled object.
* ``resource_report(...)`` adds a structural compiled-circuit view that
  recursively reports ancilla usage and selector/control overhead.
* optional backend export adds a separate SDK-side view (currently
  Qiskit) for instruction/depth and two-qubit counts when those are
  representable after export/transpilation.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

from .circuit import Circuit, CircuitResourceSummary
from .export import export_circuit
from .gate import CircuitCall, Gate, MultiplexedGate, SelectGate

__all__ = [
    "SelectorControlSummary",
    "CompiledCircuitResourceEstimate",
    "BackendCircuitResourceEstimate",
    "CircuitResourceReport",
    "resource_report",
]


@dataclass(frozen=True)
class SelectorControlSummary:
    """Selector/control overhead visible in the compiled circuit structure."""

    select_gate_count: int
    multiplexed_gate_count: int
    compiled_selector_state_count: int
    explicit_branch_count: int
    default_routed_state_count: int
    max_selector_width: int
    max_control_width: int
    selector_width_histogram: dict[int, int]


@dataclass(frozen=True)
class CompiledCircuitResourceEstimate:
    """Structural resource view derived from the compiled COMPOSER circuit."""

    num_qubits: int
    system_qubits: int | None
    ancilla_qubits: int | None
    logical_summary: CircuitResourceSummary
    expanded_gate_count: int
    expanded_gate_count_by_kind: dict[str, int]
    dense_leaf_gate_count: int
    structural_gate_count: int
    max_expanded_gate_arity: int
    selector_control: SelectorControlSummary


@dataclass(frozen=True)
class BackendCircuitResourceEstimate:
    """Optional SDK/export resource view for one exported circuit backend."""

    backend: str
    preserve_hierarchy: bool
    optimization_level: int | None
    basis_gates: tuple[str, ...] | None
    exported_instruction_count: int
    exported_max_instruction_arity: int
    exported_gate_family_counts: dict[str, int]
    transpiled_instruction_count: int | None
    transpiled_depth: int | None
    transpiled_max_instruction_arity: int | None
    transpiled_gate_family_counts: dict[str, int] | None
    two_qubit_count: int | None
    two_qubit_depth: int | None


@dataclass(frozen=True)
class CircuitResourceReport:
    """Combined resource report for a compiled circuit."""

    logical: CircuitResourceSummary
    compiled: CompiledCircuitResourceEstimate
    backend: BackendCircuitResourceEstimate | None = None


@dataclass
class _CompiledAccumulator:
    gate_count_by_kind: Counter[str]
    dense_leaf_gate_count: int = 0
    structural_gate_count: int = 0
    max_gate_arity: int = 0
    select_gate_count: int = 0
    multiplexed_gate_count: int = 0
    compiled_selector_state_count: int = 0
    explicit_branch_count: int = 0
    default_routed_state_count: int = 0
    max_selector_width: int = 0
    max_control_width: int = 0
    selector_width_histogram: Counter[int] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.selector_width_histogram is None:
            self.selector_width_histogram = Counter()


def _merge_accumulator(target: _CompiledAccumulator, source: _CompiledAccumulator) -> None:
    target.gate_count_by_kind.update(source.gate_count_by_kind)
    target.dense_leaf_gate_count += source.dense_leaf_gate_count
    target.structural_gate_count += source.structural_gate_count
    target.max_gate_arity = max(target.max_gate_arity, source.max_gate_arity)
    target.select_gate_count += source.select_gate_count
    target.multiplexed_gate_count += source.multiplexed_gate_count
    target.compiled_selector_state_count += source.compiled_selector_state_count
    target.explicit_branch_count += source.explicit_branch_count
    target.default_routed_state_count += source.default_routed_state_count
    target.max_selector_width = max(target.max_selector_width, source.max_selector_width)
    target.max_control_width = max(target.max_control_width, source.max_control_width)
    target.selector_width_histogram.update(source.selector_width_histogram)


def _validate_system_width(circuit: Circuit, system_width: int | None) -> int | None:
    if system_width is None:
        return None
    if system_width < 0 or system_width > circuit.num_qubits:
        raise ValueError(
            f"system_width must lie in [0, {circuit.num_qubits}], got {system_width}"
        )
    return system_width


def _summarize_compiled_subtree(
    circuit: Circuit,
    *,
    memo: dict[int, _CompiledAccumulator],
    recursion_stack: set[int],
) -> _CompiledAccumulator:
    cached = memo.get(id(circuit))
    if cached is not None:
        return cached

    circuit_id = id(circuit)
    if circuit_id in recursion_stack:
        raise ValueError("recursive circuit definitions are not supported in resource_report()")
    recursion_stack.add(circuit_id)
    accumulator = _CompiledAccumulator(gate_count_by_kind=Counter())
    try:
        for op in circuit.gates:
            kind = op.kind or op.name
            accumulator.gate_count_by_kind[kind] += 1
            accumulator.max_gate_arity = max(accumulator.max_gate_arity, op.num_qubits)
            if isinstance(op, Gate):
                accumulator.dense_leaf_gate_count += 1
                continue

            accumulator.structural_gate_count += 1

            if isinstance(op, CircuitCall):
                _merge_accumulator(
                    accumulator,
                    _summarize_compiled_subtree(
                        op.subcircuit,
                        memo=memo,
                        recursion_stack=recursion_stack,
                    ),
                )
                continue

            if isinstance(op, SelectGate):
                accumulator.select_gate_count += 1
                accumulator.compiled_selector_state_count += 2
                accumulator.explicit_branch_count += 2
                accumulator.max_selector_width = max(accumulator.max_selector_width, 1)
                accumulator.max_control_width = max(accumulator.max_control_width, 1)
                accumulator.selector_width_histogram[1] += 1
                _merge_accumulator(
                    accumulator,
                    _summarize_compiled_subtree(
                        op.zero_circuit,
                        memo=memo,
                        recursion_stack=recursion_stack,
                    ),
                )
                _merge_accumulator(
                    accumulator,
                    _summarize_compiled_subtree(
                        op.one_circuit,
                        memo=memo,
                        recursion_stack=recursion_stack,
                    ),
                )
                continue

            if isinstance(op, MultiplexedGate):
                selector_states = (1 << op.selector_width) if op.selector_width > 0 else 1
                accumulator.multiplexed_gate_count += 1
                accumulator.compiled_selector_state_count += selector_states
                accumulator.explicit_branch_count += len(op.branch_circuits)
                accumulator.max_selector_width = max(
                    accumulator.max_selector_width,
                    op.selector_width,
                )
                accumulator.max_control_width = max(
                    accumulator.max_control_width,
                    op.selector_width,
                )
                accumulator.selector_width_histogram[op.selector_width] += 1
                if op.default_circuit is not None:
                    accumulator.default_routed_state_count += max(
                        selector_states - len(op.branch_circuits),
                        0,
                    )
                for branch in op.branch_circuits:
                    _merge_accumulator(
                        accumulator,
                        _summarize_compiled_subtree(
                            branch,
                            memo=memo,
                            recursion_stack=recursion_stack,
                        ),
                    )
                if op.default_circuit is not None:
                    _merge_accumulator(
                        accumulator,
                        _summarize_compiled_subtree(
                            op.default_circuit,
                            memo=memo,
                            recursion_stack=recursion_stack,
                        ),
                    )
    finally:
        recursion_stack.remove(circuit_id)
    memo[circuit_id] = accumulator
    return accumulator


def _compiled_resource_estimate(
    circuit: Circuit,
    *,
    system_width: int | None,
) -> CompiledCircuitResourceEstimate:
    logical = circuit.resource_summary()
    accumulator = _summarize_compiled_subtree(circuit, memo={}, recursion_stack=set())
    return CompiledCircuitResourceEstimate(
        num_qubits=circuit.num_qubits,
        system_qubits=system_width,
        ancilla_qubits=None if system_width is None else circuit.num_qubits - system_width,
        logical_summary=logical,
        expanded_gate_count=sum(accumulator.gate_count_by_kind.values()),
        expanded_gate_count_by_kind=dict(sorted(accumulator.gate_count_by_kind.items())),
        dense_leaf_gate_count=accumulator.dense_leaf_gate_count,
        structural_gate_count=accumulator.structural_gate_count,
        max_expanded_gate_arity=accumulator.max_gate_arity,
        selector_control=SelectorControlSummary(
            select_gate_count=accumulator.select_gate_count,
            multiplexed_gate_count=accumulator.multiplexed_gate_count,
            compiled_selector_state_count=accumulator.compiled_selector_state_count,
            explicit_branch_count=accumulator.explicit_branch_count,
            default_routed_state_count=accumulator.default_routed_state_count,
            max_selector_width=accumulator.max_selector_width,
            max_control_width=accumulator.max_control_width,
            selector_width_histogram=dict(sorted(accumulator.selector_width_histogram.items())),
        ),
    )


def _instruction_arity(quantum_circuit) -> int:
    return max((len(instruction.qubits) for instruction in quantum_circuit.data), default=0)


def _gate_family_counts(quantum_circuit) -> dict[str, int]:
    return dict(sorted(quantum_circuit.count_ops().items()))


def _transpiled_two_qubit_depth(quantum_circuit) -> int:
    qubit_depths = [0] * quantum_circuit.num_qubits
    for instruction in quantum_circuit.data:
        qubit_indices = [quantum_circuit.find_bit(q).index for q in instruction.qubits]
        if len(qubit_indices) != 2:
            continue
        layer = max(qubit_depths[index] for index in qubit_indices) + 1
        for index in qubit_indices:
            qubit_depths[index] = layer
    return max(qubit_depths, default=0)


def _backend_resource_estimate(
    circuit: Circuit,
    *,
    backend: str,
    preserve_hierarchy: bool,
    allow_unitary_fallback: bool,
    basis_gates: Sequence[str] | None,
    optimization_level: int,
) -> BackendCircuitResourceEstimate:
    exported = export_circuit(
        circuit,
        backend=backend,
        preserve_hierarchy=preserve_hierarchy,
        allow_unitary_fallback=allow_unitary_fallback,
    )

    transpiled = None
    transpiled_instruction_count = None
    transpiled_depth = None
    transpiled_max_instruction_arity = None
    transpiled_gate_family_counts = None
    two_qubit_count = None
    two_qubit_depth = None
    basis_tuple = None if basis_gates is None else tuple(basis_gates)

    if backend == "qiskit" and basis_tuple is not None:
        from qiskit import transpile

        transpiled = transpile(
            exported,
            basis_gates=list(basis_tuple),
            optimization_level=optimization_level,
        )
        transpiled_instruction_count = len(transpiled.data)
        transpiled_depth = transpiled.depth()
        transpiled_max_instruction_arity = _instruction_arity(transpiled)
        transpiled_gate_family_counts = _gate_family_counts(transpiled)
        if transpiled_max_instruction_arity <= 2:
            two_qubit_count = sum(len(instruction.qubits) == 2 for instruction in transpiled.data)
            two_qubit_depth = _transpiled_two_qubit_depth(transpiled)

    return BackendCircuitResourceEstimate(
        backend=backend,
        preserve_hierarchy=preserve_hierarchy,
        optimization_level=optimization_level if basis_tuple is not None else None,
        basis_gates=basis_tuple,
        exported_instruction_count=len(exported.data),
        exported_max_instruction_arity=_instruction_arity(exported),
        exported_gate_family_counts=_gate_family_counts(exported),
        transpiled_instruction_count=transpiled_instruction_count,
        transpiled_depth=transpiled_depth,
        transpiled_max_instruction_arity=transpiled_max_instruction_arity,
        transpiled_gate_family_counts=transpiled_gate_family_counts,
        two_qubit_count=two_qubit_count,
        two_qubit_depth=two_qubit_depth,
    )


def resource_report(
    circuit: Circuit,
    *,
    system_width: int | None = None,
    backend: str | None = None,
    preserve_hierarchy: bool = True,
    allow_unitary_fallback: bool = True,
    basis_gates: Sequence[str] | None = None,
    optimization_level: int = 0,
) -> CircuitResourceReport:
    """Return compiled and optional backend resource views for ``circuit``.

    Parameters
    ----------
    circuit:
        The compiled COMPOSER circuit to analyze.
    system_width:
        Optional width of the logical system register. When supplied,
        the report also exposes ancilla count as
        ``circuit.num_qubits - system_width``.
    backend:
        Optional export backend name. The current repo scope supports
        ``"qiskit"``.
    preserve_hierarchy:
        Forwarded to the backend exporter when ``backend`` is requested.
    allow_unitary_fallback:
        Forwarded to the backend exporter when ``backend`` is requested.
    basis_gates:
        Optional backend basis to transpile into. When omitted, the
        backend report still returns exported instruction families but
        leaves backend depth / two-qubit counts unset.
    optimization_level:
        Backend transpilation optimization level used when
        ``basis_gates`` is provided.
    """
    system_width = _validate_system_width(circuit, system_width)
    compiled = _compiled_resource_estimate(circuit, system_width=system_width)
    logical = circuit.resource_summary()
    backend_report = None
    if backend is not None:
        backend_report = _backend_resource_estimate(
            circuit,
            backend=backend,
            preserve_hierarchy=preserve_hierarchy,
            allow_unitary_fallback=allow_unitary_fallback,
            basis_gates=basis_gates,
            optimization_level=optimization_level,
        )
    return CircuitResourceReport(
        logical=logical,
        compiled=compiled,
        backend=backend_report,
    )
