"""Optional Qiskit export checks for compiled COMPOSER circuits."""
from __future__ import annotations

import numpy as np

from composer.circuits.circuit import Circuit
from composer.circuits.export import qiskit_available, to_qiskit
from composer.circuits.gate import (
    AncillaZeroReflectionGate,
    CircuitCall,
    Gate,
    MultiplexedGate,
    SelectGate,
    StatePreparationGate,
)
from composer.circuits.simulator import unitary

if qiskit_available():
    from qiskit.quantum_info import Operator

    def _x_gate() -> np.ndarray:
        return np.array([[0, 1], [1, 0]], dtype=complex)


    def _h_gate() -> np.ndarray:
        return np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2.0)


    def _identity_circuit(width: int) -> Circuit:
        circuit = Circuit(num_qubits=width)
        circuit.append(Gate("I", tuple(range(width)), np.eye(2**width, dtype=complex), kind="I"))
        return circuit


    def test_qiskit_export_matches_dense_unitary_for_primitives_and_calls():
        child = Circuit(num_qubits=1)
        child.append(Gate("X", (0,), _x_gate(), kind="X"))

        circuit = Circuit(num_qubits=2)
        circuit.append(Gate("H", (0,), _h_gate(), kind="H"))
        circuit.append(CircuitCall(name="child_x", qubits=(1,), subcircuit=child, kind="child_x"))

        exported = to_qiskit(circuit)

        assert exported.num_qubits == circuit.num_qubits
        assert len(exported.data) >= len(circuit.gates)
        assert np.allclose(Operator(exported).data, unitary(circuit))


    def test_qiskit_export_lowers_select_gate_without_whole_circuit_unitary_wrap():
        zero_branch = _identity_circuit(1)
        one_branch = Circuit(num_qubits=1)
        one_branch.append(Gate("X", (0,), _x_gate(), kind="X"))

        circuit = Circuit(num_qubits=2)
        circuit.append(
            SelectGate(
                name="select_x",
                qubits=(0, 1),
                zero_circuit=zero_branch,
                one_circuit=one_branch,
                kind="select_x",
            )
        )

        exported = to_qiskit(circuit)

        assert len(exported.data) >= 1
        assert not (
            len(exported.data) == 1
            and exported.data[0].operation.name == "unitary"
            and len(exported.data[0].qubits) == circuit.num_qubits
        )
        assert np.allclose(Operator(exported).data, unitary(circuit))


    def test_qiskit_export_lowers_multiplexor_state_prep_and_reflection():
        amplitudes = np.array([1.0, 2.0], dtype=complex)
        amplitudes = amplitudes / np.linalg.norm(amplitudes)

        branch_zero = _identity_circuit(2)
        branch_one = Circuit(num_qubits=2)
        branch_one.append(Gate("X", (0,), _x_gate(), kind="X"))

        circuit = Circuit(num_qubits=4)
        circuit.append(
            StatePreparationGate(
                name="prep_sel",
                qubits=(2, 3),
                amplitudes=np.array([1.0, 1.0, 0.0, 0.0], dtype=complex) / np.sqrt(2.0),
                kind="prep_sel",
            )
        )
        circuit.append(
            MultiplexedGate(
                name="mux",
                qubits=(0, 1, 2, 3),
                selector_width=2,
                branch_circuits=(branch_zero, branch_one),
                default_circuit=_identity_circuit(2),
                kind="mux",
            )
        )
        circuit.append(
            StatePreparationGate(
                name="prep_sys",
                qubits=(0,),
                amplitudes=amplitudes,
                kind="prep_sys",
            )
        )
        circuit.append(
            AncillaZeroReflectionGate(
                name="reflect",
                qubits=(0, 1, 2, 3),
                system_width=1,
                kind="reflect",
            )
        )

        exported = to_qiskit(circuit)

        assert len(exported.data) >= len(circuit.gates)
        assert not (
            len(exported.data) == 1
            and exported.data[0].operation.name == "unitary"
            and len(exported.data[0].qubits) == circuit.num_qubits
        )
        assert np.allclose(Operator(exported).data, unitary(circuit))


    def test_qiskit_resource_report_adds_transpiled_depth_and_two_qubit_counts():
        branch_zero = _identity_circuit(1)
        branch_one = Circuit(num_qubits=1)
        branch_one.append(Gate("X", (0,), _x_gate(), kind="X"))

        circuit = Circuit(num_qubits=3)
        circuit.append(
            SelectGate(
                name="select_x",
                qubits=(0, 1),
                zero_circuit=branch_zero,
                one_circuit=branch_one,
                kind="select_x",
            )
        )
        circuit.append(
            MultiplexedGate(
                name="mux_x",
                qubits=(0, 1, 2),
                selector_width=2,
                branch_circuits=(branch_zero, branch_one),
                default_circuit=branch_zero,
                kind="mux_x",
            )
        )

        report = circuit.resource_report(
            system_width=1,
            backend="qiskit",
            preserve_hierarchy=False,
            basis_gates=("u", "cx"),
            optimization_level=0,
        )

        assert report.compiled.ancilla_qubits == 2
        assert report.compiled.selector_control.max_selector_width == 2
        assert report.backend is not None
        assert report.backend.backend == "qiskit"
        assert report.backend.exported_instruction_count >= 2
        assert report.backend.transpiled_instruction_count is not None
        assert report.backend.transpiled_depth is not None
        assert report.backend.transpiled_max_instruction_arity == 2
        assert report.backend.two_qubit_count is not None
        assert report.backend.two_qubit_depth is not None
        assert report.backend.two_qubit_count > 0
        assert report.backend.transpiled_gate_family_counts is not None
        assert "cx" in report.backend.transpiled_gate_family_counts
