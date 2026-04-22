"""Qiskit export adapter for compiled COMPOSER circuits.

This is intentionally an adapter layer: the reference semantics stay in
COMPOSER's own ``Circuit`` object model, while this module lowers those
objects into Qiskit circuits on demand. Structural COMPOSER operations
(``CircuitCall``, ``SelectGate``, ``MultiplexedGate``, and
``AncillaZeroReflectionGate``) are preserved structurally where Qiskit
can express them directly. ``StatePreparationGate`` is exported through
its exact COMPOSER verification unitary because COMPOSER fixes the full
unitary action, not only the prepared ``|0...0>`` column. Generic dense
leaf gates are exported as native SDK primitives when recognized, and
otherwise fall back to ``UnitaryGate``.
"""
from __future__ import annotations

import importlib
from cmath import phase

import numpy as np

from ..circuit import Circuit
from ..gate import (
    AncillaZeroReflectionGate,
    CircuitCall,
    Gate,
    MultiplexedGate,
    SelectGate,
    StatePreparationGate,
)

__all__ = ["qiskit_available", "to_qiskit"]


def qiskit_available() -> bool:
    """Return whether Qiskit can be imported in the active interpreter."""
    return importlib.util.find_spec("qiskit") is not None


def _require_qiskit():
    if not qiskit_available():
        raise ModuleNotFoundError(
            "Qiskit is not installed. Install the optional backend with "
            "`python -m pip install -e '.[qiskit]'`."
        )
    qiskit = importlib.import_module("qiskit")
    library = importlib.import_module("qiskit.circuit.library")
    return qiskit, library


class _QiskitExporter:
    def __init__(self, *, preserve_hierarchy: bool, allow_unitary_fallback: bool) -> None:
        self.preserve_hierarchy = preserve_hierarchy
        self.allow_unitary_fallback = allow_unitary_fallback
        self._circuit_cache: dict[tuple[int, str], object] = {}
        self._gate_cache: dict[tuple[int, str], object] = {}
        self.qiskit, self.library = _require_qiskit()

    def export_circuit(self, circuit: Circuit, *, name: str | None = None):
        cache_key = (id(circuit), name or "")
        cached = self._circuit_cache.get(cache_key)
        if cached is not None:
            return cached.copy()

        quantum_circuit = self.qiskit.QuantumCircuit(circuit.num_qubits, name=name or "composer")
        quantum_circuit.metadata = {
            "composer_num_qubits": circuit.num_qubits,
            "composer_compiled_signature_hash": circuit.compiled_signature_hash(),
            "composer_gate_count": len(circuit.gates),
        }
        for op in circuit.gates:
            self._append_op(quantum_circuit, op)

        self._circuit_cache[cache_key] = quantum_circuit
        return quantum_circuit.copy()

    def _append_op(self, quantum_circuit, op) -> None:
        if isinstance(op, Gate):
            quantum_circuit.append(self._primitive_instruction(op), list(op.qubits))
            return

        if isinstance(op, CircuitCall):
            if self.preserve_hierarchy:
                quantum_circuit.append(
                    self._circuit_as_gate(op.subcircuit, label=op.name),
                    list(op.qubits),
                )
            else:
                quantum_circuit.compose(
                    self.export_circuit(op.subcircuit, name=op.name),
                    qubits=list(op.qubits),
                    inplace=True,
                )
            return

        if isinstance(op, StatePreparationGate):
            if not self.allow_unitary_fallback:
                raise ValueError(
                    "exact Qiskit export of StatePreparationGate currently requires "
                    "unitary fallback because COMPOSER fixes the full verification "
                    "unitary, not only the prepared first column"
                )
            quantum_circuit.append(
                self.library.UnitaryGate(op.matrix, label=op.name),
                list(op.qubits),
            )
            return

        if isinstance(op, SelectGate):
            self._append_select_gate(quantum_circuit, op)
            return

        if isinstance(op, MultiplexedGate):
            self._append_multiplexed_gate(quantum_circuit, op)
            return

        if isinstance(op, AncillaZeroReflectionGate):
            self._append_ancilla_zero_reflection(quantum_circuit, op)
            return

        raise TypeError(f"unsupported COMPOSER circuit op: {type(op).__name__}")

    def _primitive_instruction(self, gate: Gate):
        matrix = np.asarray(gate.matrix, dtype=complex)
        if gate.num_qubits == 1:
            if np.allclose(matrix, np.array([[0, 1], [1, 0]], dtype=complex)):
                return self.library.XGate()
            if np.allclose(
                matrix,
                np.array([[1, 0], [0, -1]], dtype=complex),
            ):
                return self.library.ZGate()
            if np.allclose(
                matrix,
                np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0),
            ):
                return self.library.HGate()
            if np.allclose(
                matrix,
                np.array([[1, 0], [0, 1j]], dtype=complex),
            ):
                return self.library.SGate()
            if np.allclose(
                matrix,
                np.array([[1, 0], [0, -1j]], dtype=complex),
            ):
                return self.library.SdgGate()
        if gate.num_qubits == 2:
            cnot_lsb = np.array(
                [[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]],
                dtype=complex,
            )
            if np.allclose(matrix, cnot_lsb):
                return self.library.CXGate()
        if not self.allow_unitary_fallback:
            raise ValueError(
                f"cannot export dense primitive gate {gate.name!r} without unitary fallback"
            )
        return self.library.UnitaryGate(matrix, label=gate.name)

    def _circuit_as_gate(self, circuit: Circuit, *, label: str):
        cache_key = (id(circuit), label)
        cached = self._gate_cache.get(cache_key)
        if cached is not None:
            return cached
        gate = self.export_circuit(circuit, name=label).to_gate(label=label)
        self._gate_cache[cache_key] = gate
        return gate

    def _append_controlled_branch(
        self,
        quantum_circuit,
        *,
        branch_circuit: Circuit,
        target_qubits: tuple[int, ...],
        control_qubits: tuple[int, ...],
        selector_state: int,
        label: str,
        branch_phase: complex = 1.0 + 0.0j,
    ) -> None:
        if np.isclose(branch_phase, 1.0, atol=1e-12) and self._is_identity_circuit(branch_circuit):
            return
        if len(control_qubits) == 0:
            if not np.isclose(branch_phase, 1.0, atol=1e-12):
                quantum_circuit.global_phase += phase(branch_phase)
            if self.preserve_hierarchy:
                quantum_circuit.append(
                    self._circuit_as_gate(branch_circuit, label=label),
                    list(target_qubits),
                )
            else:
                quantum_circuit.compose(
                    self.export_circuit(branch_circuit, name=label),
                    qubits=list(target_qubits),
                    inplace=True,
                )
            return

        conditioned_controls = []
        for bit_index, control_qubit in enumerate(control_qubits):
            if ((selector_state >> bit_index) & 1) == 0:
                quantum_circuit.x(control_qubit)
                conditioned_controls.append(control_qubit)

        base_circuit = self.export_circuit(branch_circuit, name=label)
        if not np.isclose(branch_phase, 1.0, atol=1e-12):
            base_circuit.global_phase += phase(branch_phase)
        branch_gate = base_circuit.to_gate(label=label)
        controlled_gate = branch_gate.control(len(control_qubits))
        quantum_circuit.append(controlled_gate, list(control_qubits) + list(target_qubits))

        for control_qubit in reversed(conditioned_controls):
            quantum_circuit.x(control_qubit)

    @staticmethod
    def _is_identity_circuit(circuit: Circuit) -> bool:
        if not circuit.gates:
            return True
        for op in circuit.gates:
            if not isinstance(op, Gate):
                return False
            dim = 2 ** op.num_qubits
            if op.qubits != tuple(range(circuit.num_qubits)):
                return False
            if not np.allclose(op.matrix, np.eye(dim, dtype=complex)):
                return False
        return True

    def _append_select_gate(self, quantum_circuit, gate: SelectGate) -> None:
        target_qubits = gate.target_qubits
        control_qubit = gate.control_qubit
        self._append_controlled_branch(
            quantum_circuit,
            branch_circuit=gate.zero_circuit,
            target_qubits=target_qubits,
            control_qubits=(control_qubit,),
            selector_state=0,
            label=f"{gate.name}_zero",
        )
        self._append_controlled_branch(
            quantum_circuit,
            branch_circuit=gate.one_circuit,
            target_qubits=target_qubits,
            control_qubits=(control_qubit,),
            selector_state=1,
            label=f"{gate.name}_one",
        )

    def _append_multiplexed_gate(self, quantum_circuit, gate: MultiplexedGate) -> None:
        target_qubits = gate.target_qubits
        selector_qubits = gate.selector_qubits
        if gate.selector_width == 0:
            branch = gate.branch_circuits[0] if gate.branch_circuits else gate.default_circuit
            branch_phase = gate.branch_phases[0] if gate.branch_circuits else 1.0 + 0.0j
            if branch is None:
                return
            self._append_controlled_branch(
                quantum_circuit,
                branch_circuit=branch,
                target_qubits=target_qubits,
                control_qubits=(),
                selector_state=0,
                label=gate.name,
                branch_phase=branch_phase,
            )
            return

        for selector_state, branch_circuit in enumerate(gate.branch_circuits):
            self._append_controlled_branch(
                quantum_circuit,
                branch_circuit=branch_circuit,
                target_qubits=target_qubits,
                control_qubits=selector_qubits,
                selector_state=selector_state,
                label=f"{gate.name}_{selector_state}",
                branch_phase=gate.branch_phases[selector_state],
            )

        if gate.default_circuit is None:
            return

        for selector_state in range(len(gate.branch_circuits), 2 ** gate.selector_width):
            self._append_controlled_branch(
                quantum_circuit,
                branch_circuit=gate.default_circuit,
                target_qubits=target_qubits,
                control_qubits=selector_qubits,
                selector_state=selector_state,
                label=f"{gate.name}_default_{selector_state}",
            )

    def _append_ancilla_zero_reflection(self, quantum_circuit, gate: AncillaZeroReflectionGate) -> None:
        ancilla_qubits = gate.qubits[gate.system_width :]
        if len(ancilla_qubits) == 1:
            quantum_circuit.z(ancilla_qubits[0])
            return

        quantum_circuit.global_phase += np.pi
        for qubit in ancilla_qubits:
            quantum_circuit.x(qubit)
        last = ancilla_qubits[-1]
        controls = list(ancilla_qubits[:-1])
        quantum_circuit.h(last)
        quantum_circuit.mcx(controls, last)
        quantum_circuit.h(last)
        for qubit in reversed(ancilla_qubits):
            quantum_circuit.x(qubit)


def to_qiskit(
    circuit: Circuit,
    *,
    preserve_hierarchy: bool = True,
    allow_unitary_fallback: bool = True,
):
    """Export a compiled COMPOSER ``Circuit`` into a Qiskit ``QuantumCircuit``.

    Structural COMPOSER operations are lowered recursively; the
    resulting Qiskit circuit therefore preserves the compiled hierarchy
    and selector structure instead of collapsing the whole COMPOSER
    circuit into one dense top-level matrix.
    """
    exporter = _QiskitExporter(
        preserve_hierarchy=preserve_hierarchy,
        allow_unitary_fallback=allow_unitary_fallback,
    )
    return exporter.export_circuit(circuit)
