"""End-to-end Hamiltonian block encoding on the shipped H2 / STO-3G data.

Loads the repository's canned spin-orbital integrals, builds the
rank-one Hamiltonian pool, compiles the Theorem-1 PREP-SELECT-PREP^dag
block encoding, and prints the ground-state energy recovered from the
dense reconstruction.
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
from composer.circuits import resource_report
from composer.operators.hamiltonian import build_pool_from_integrals

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "h2_sto3g_integrals.npz")


def main() -> None:
    d = np.load(DATA_PATH)
    h, eri, E_nuc = d["h"], d["eri"], float(d["E_nuc"])
    NO, NV = int(d["NO"]), int(d["NV"])
    print(f"H2 / STO-3G: NO={NO} occupied spin orbitals, NV={NV} virtual spin orbitals")
    print(f"Total spin-orbital count: {h.shape[0]}")
    print(f"Nuclear repulsion E_nuc = {E_nuc:.6f} Ha")

    pool = build_pool_from_integrals(h, eri)
    print(f"Hamiltonian pool: h~ shape {pool.h_tilde.shape}, "
          f"Cholesky rank {pool.cholesky_factors.shape[0]}")

    be = build_hamiltonian_block_encoding(pool)
    compiled_report = resource_report(be.circuit, system_width=be.n_system)
    print(f"Theorem-1 block encoding: alpha = {be.alpha:.6f}, "
          f"selector width = {be.n_ancilla} qubits, "
          f"{be.W.shape[0]} x {be.W.shape[1]} unitary")
    print(
        "Compiled LCU resources: "
        f"branches={be.resources.active_branch_count} "
        f"(one-body={be.resources.one_body_branch_count}, "
        f"Cholesky={be.resources.cholesky_branch_count}), "
        f"compiled selector branches={be.resources.compiled_branch_count}, "
        f"gate inventory={be.resources.circuit.gate_count_by_kind}"
    )
    print(
        "Compiled synthesis view: "
        f"system={compiled_report.compiled.system_qubits}, "
        f"ancilla={compiled_report.compiled.ancilla_qubits}, "
        f"dense leaves={compiled_report.compiled.dense_leaf_gate_count}, "
        f"selector states={compiled_report.compiled.selector_control.compiled_selector_state_count}, "
        f"max selector width={compiled_report.compiled.selector_control.max_selector_width}"
    )

    # Verify the block encoding reproduces the dense Hamiltonian.
    block = be.top_left_block()
    reconstructed = block * be.alpha
    H_dense = pool.dense_matrix()
    residual = float(np.max(np.abs(reconstructed - H_dense)))
    print(f"||alpha * top_left_block - H_dense||_max = {residual:.2e}")

    # Ground state in the N=2 subspace.
    n = pool.n_orbitals
    occs = np.array([bin(k).count("1") for k in range(2**n)])
    idx = np.where(occs == 2)[0]
    H_N2 = H_dense[np.ix_(idx, idx)]
    eigs = np.linalg.eigvalsh(H_N2)
    ground_elec = float(eigs.min())
    print(f"Electronic ground state (N=2): {ground_elec:.6f} Ha")
    print(f"Total ground state energy: {ground_elec + E_nuc:.6f} Ha  "
          f"(published FCI ~ -1.137 Ha)")


if __name__ == "__main__":
    main()
