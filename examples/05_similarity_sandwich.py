"""App-E workflow: MP2 diagnostics, selector-mask choice, and sandwich reuse.

The shipped H2 / STO-3G dataset has only one nonzero MP2 pair channel,
so it is too small to meaningfully demonstrate mask screening or wAUC.
This example instead uses a small synthetic occupied/virtual split that
stays within the repo's supported dense-verification scale while making
the diagnostics non-vacuous.
"""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np

REPO_SRC = Path(__file__).resolve().parents[1] / "src"
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

from composer.block_encoding.similarity_sandwich import (
    ModelSpaceProjector,
    build_similarity_sandwich,
)
from composer.circuits import resource_report
from composer.diagnostics.mask_selection import channel_weights_mp2, cumulative_coverage_mask
from composer.diagnostics.mp2 import mp2_doubles_amplitudes
from composer.diagnostics.subspace import wauc
from composer.operators.generator import build_cluster_generator
from composer.operators.hamiltonian import build_pool_from_integrals
from composer.operators.mask import uniform_mask


def _random_real_symmetric_eri(n: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    eri_chem = np.zeros((n, n, n, n))
    for _ in range(rank):
        A = rng.normal(size=(n, n))
        L = 0.5 * (A + A.T)
        eri_chem += np.einsum("pq,rs->pqrs", L, L)
    return eri_chem.transpose(0, 2, 1, 3)


def _n_electron_determinants(n_orbitals: int, n_electrons: int) -> tuple[int, ...]:
    return tuple(idx for idx in range(2**n_orbitals) if bin(idx).count("1") == n_electrons)


def main() -> None:
    rng = np.random.default_rng(0)
    NO, NV = 3, 3
    n = NO + NV
    eps_occ = np.array([-1.6, -1.2, -0.9])
    eps_vir = np.array([0.3, 0.7, 1.1])
    h = np.diag(np.concatenate([eps_occ, eps_vir]))
    eri = 0.03 * _random_real_symmetric_eri(n, rank=3, rng=rng)

    pool = build_pool_from_integrals(h, eri)
    t2_mp2 = mp2_doubles_amplitudes(eri, eps_occ, eps_vir)
    gen = build_cluster_generator(NO, NV, t2=t2_mp2)
    channels = gen.pair_rank_one_pool()
    weights = channel_weights_mp2(channels)
    mask_selected = cumulative_coverage_mask(channels, coverage=0.9)
    mask_full = uniform_mask(len(channels))
    selected_channels = [ch for ch, keep in zip(channels, mask_selected.weights) if keep > 0.5]

    proj = ModelSpaceProjector(determinants=_n_electron_determinants(n, NO))
    sw_compiled = build_similarity_sandwich(pool, gen, mask_full, proj, exp_eps=8e-2, qsp_max_iter=400)
    sw_selected = sw_compiled.redial_mask(mask_selected)
    compiled_report = resource_report(sw_compiled.circuit, system_width=sw_compiled.resources.n_system)
    P = proj.dense_matrix(n)
    nested_projected_compiled = P @ (sw_compiled.alpha * sw_compiled.encoded_system_block_dense) @ P
    nested_projected_selected = P @ (sw_selected.alpha * sw_selected.encoded_system_block_dense) @ P

    print("Synthetic 3-occ / 3-vir MP2 screening workflow")
    print(f"Pair-SVD channel count: {len(channels)}")
    print(f"MP2 ladder weights: {np.array2string(weights, precision=6)}")
    print(f"Selected labels @ 90% coverage: {np.flatnonzero(mask_selected.weights).tolist()}")
    print(f"wAUC(full, selected) = {wauc(channels, selected_channels):.6f}")
    print(
        f"||H_eff(full) - H_eff(selected)||_F = "
        f"{np.linalg.norm(sw_compiled.H_eff_dense - sw_selected.H_eff_dense):.4e}"
    )
    print(
        "Nested circuit vs paper target, ||P[alpha<0|W_eff|0>]P - H_eff||_F: "
        f"{np.linalg.norm(nested_projected_compiled - sw_compiled.H_eff_dense):.4e} "
        f"(full), {np.linalg.norm(nested_projected_selected - sw_selected.H_eff_dense):.4e} (selected)"
    )
    print(
        "Compiled Hamiltonian resources: "
        f"selector_width={sw_compiled.hamiltonian_oracle.resources.selector_width}, "
        f"branches={sw_compiled.hamiltonian_oracle.resources.active_branch_count}, "
        f"gate inventory={sw_compiled.hamiltonian_oracle.resources.circuit.gate_count_by_kind}"
    )
    print(
        "Compiled sigma resources: "
        f"selector_width={sw_compiled.generator_exp_oracle.sigma_oracle.resources.selector_width}, "
        f"compiled branches={sw_compiled.generator_exp_oracle.sigma_oracle.resources.compiled_branch_count}, "
        f"gate inventory={sw_compiled.generator_exp_oracle.sigma_oracle.resources.circuit.gate_count_by_kind}"
    )
    print(
        "Compiled generator-exp resources: "
        f"cos queries={sw_compiled.generator_exp_oracle.resources.cos_qsp_query_count}, "
        f"sin queries={sw_compiled.generator_exp_oracle.resources.sin_qsp_query_count}, "
        f"outer gate inventory={sw_compiled.generator_exp_oracle.resources.circuit.gate_count_by_kind}"
    )
    print(
        "Outer sandwich resources: "
        f"U_sigma calls={sw_compiled.resources.u_sigma_call_count}, "
        f"projector rank={sw_compiled.resources.projector_rank}, "
        f"gate inventory={sw_compiled.resources.circuit.gate_count_by_kind}"
    )
    print(
        "Nested compiled synthesis view: "
        f"ancilla={compiled_report.compiled.ancilla_qubits}, "
        f"expanded gate kinds={compiled_report.compiled.expanded_gate_count_by_kind}, "
        f"dense leaf kinds={compiled_report.compiled.dense_leaf_gate_count_by_kind}, "
        f"selector states={compiled_report.compiled.selector_control.compiled_selector_state_count}, "
        f"max selector width={compiled_report.compiled.selector_control.max_selector_width}"
    )
    print(
        "Outer circuit uses nested subcircuits only: "
        f"{all(g.implementation_tag() == 'subcircuit' for g in sw_compiled.circuit.gates)}"
    )
    print(f"topology hash (full)    : {sw_compiled.topology_hash}")
    print(f"topology hash (selected): {sw_selected.topology_hash}")
    print(f"compiled signature (full)    : {sw_compiled.compiled_signature_hash}")
    print(f"compiled signature (selected): {sw_selected.compiled_signature_hash}")
    print(
        "Compile-once structural check: "
        f"{sw_compiled.compiled_signature_hash == sw_selected.compiled_signature_hash}"
    )
    print(
        "Fixed compiled structure under mask re-dial: "
        f"{sw_compiled.compiled_signature_hash == sw_selected.compiled_signature_hash}"
    )
    print(
        "Outer nested oracle payload reused: "
        f"{sw_compiled.circuit.gates[0].subcircuit.compiled_signature_hash() == sw_selected.circuit.gates[0].subcircuit.compiled_signature_hash()}"
    )
    print(
        "QSP phases reused: "
        f"{np.allclose(sw_compiled.generator_exp_oracle.cos_phases, sw_selected.generator_exp_oracle.cos_phases)}"
    )
    print(
        "Sigma SELECT fixed / PREP re-dialed: "
        f"{np.allclose(sw_compiled.generator_exp_oracle.sigma_oracle.circuit.gates[1].matrix, sw_selected.generator_exp_oracle.sigma_oracle.circuit.gates[1].matrix)}"
        " / "
        f"{not np.allclose(sw_compiled.generator_exp_oracle.sigma_oracle.circuit.gates[0].matrix, sw_selected.generator_exp_oracle.sigma_oracle.circuit.gates[0].matrix)}"
    )


if __name__ == "__main__":
    main()
