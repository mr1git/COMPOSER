"""Optional Qiskit export of the shipped H2 Hamiltonian block encoding.

This example keeps COMPOSER's own circuit/object model as the semantic
source of truth and exports the compiled Theorem-1 oracle into Qiskit as
an adapter layer for SDK-side inspection.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from composer.block_encoding.lcu import build_hamiltonian_block_encoding
from composer.circuits import qiskit_available, resource_report, to_qiskit
from composer.circuits.simulator import statevector as composer_statevector
from composer.operators.hamiltonian import build_pool_from_integrals

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "h2_sto3g_integrals.npz")


def main() -> None:
    if not qiskit_available():
        print("Qiskit is not installed. Install the optional backend with:")
        print("python -m pip install -e '.[qiskit]'")
        return

    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    data = np.load(DATA_PATH)
    pool = build_pool_from_integrals(data["h"], data["eri"])
    block_encoding = build_hamiltonian_block_encoding(pool)
    report = resource_report(
        block_encoding.circuit,
        system_width=block_encoding.n_system,
        backend="qiskit",
        preserve_hierarchy=False,
        basis_gates=("u", "cx"),
        optimization_level=0,
    )

    exported = to_qiskit(block_encoding.circuit)
    zero_state = Statevector.from_label("0" * exported.num_qubits)
    evolved = zero_state.evolve(exported)
    composer_evolved = composer_statevector(block_encoding.circuit)
    residual = float(np.max(np.abs(evolved.data - composer_evolved)))
    transpiled = transpile(exported, optimization_level=0)

    print("Qiskit export of the H2 Theorem-1 Hamiltonian oracle")
    print(f"Qubits: {exported.num_qubits}")
    print(
        "Compiled synthesis view: "
        f"ancilla={report.compiled.ancilla_qubits}, "
        f"selector states={report.compiled.selector_control.compiled_selector_state_count}, "
        f"dense leaves={report.compiled.dense_leaf_gate_count}"
    )
    print(f"Top-level instruction count: {len(exported.data)}")
    print(f"Top-level op histogram: {dict(exported.count_ops())}")
    print(f"Transpiled instruction count: {len(transpiled.data)}")
    print(f"Transpiled depth: {transpiled.depth()}")
    if report.backend is not None:
        print(
            "Backend resource view (flattened u/cx transpilation): "
            f"two-qubit count={report.backend.two_qubit_count}, "
            f"two-qubit depth={report.backend.two_qubit_depth}, "
            f"transpiled families={report.backend.transpiled_gate_family_counts}"
        )
    print(f"SDK/composer |0...0> evolution residual: {residual:.2e}")
    print(f"Statevector norm after evolution: {np.linalg.norm(evolved.data):.6f}")
    print(
        "Compiled signature hash carried in metadata: "
        f"{exported.metadata.get('composer_compiled_signature_hash')}"
    )


if __name__ == "__main__":
    main()
