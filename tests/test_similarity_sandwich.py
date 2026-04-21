"""Similarity sandwich tests (Sec. IV.C, Eq. 47-53)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from composer.block_encoding.similarity_sandwich import (
    ModelSpaceProjector,
    build_similarity_sandwich,
    effective_hamiltonian_dense,
)
from composer.block_encoding.generator_exp import dense_masked_generator_sigma
from composer.circuits.simulator import unitary as circuit_unitary
from composer.operators.generator import build_cluster_generator
from composer.operators.hamiltonian import build_pool_from_integrals
from composer.operators.mask import ChannelMask, top_k_mask, uniform_mask


def _random_real_symmetric_eri(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    eri_chem = np.zeros((n, n, n, n))
    for _ in range(K):
        A = rng.normal(size=(n, n))
        L = 0.5 * (A + A.T)
        eri_chem += np.einsum("pq,rs->pqrs", L, L)
    return eri_chem.transpose(0, 2, 1, 3)


def _random_antisymmetric_t2(NV: int, NO: int, rng: np.random.Generator) -> np.ndarray:
    t2 = rng.normal(size=(NV, NV, NO, NO)) + 1j * rng.normal(size=(NV, NV, NO, NO))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    return t2


def _build_small_pool_and_generator(seed: int = 0, *, include_singles: bool = False):
    rng = np.random.default_rng(seed)
    NO, NV = 2, 2
    n = NO + NV
    h = 0.05 * rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = 0.01 * _random_real_symmetric_eri(n, K=2, rng=rng)
    pool = build_pool_from_integrals(h, eri)

    t1 = 0.02 * rng.normal(size=(NV, NO)) if include_singles else None
    t2 = 0.02 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=t1, t2=t2)
    return pool, gen


def _build_low_ancilla_pool_and_generator(seed: int = 0):
    rng = np.random.default_rng(seed)
    NO = NV = 2
    n = NO + NV
    h = np.diag([0.0, 0.0, 0.0, 0.07])
    L = np.diag([0.0, 0.0, 0.0, 0.02])
    eri = np.einsum("pq,rs->pqrs", L, L).transpose(0, 2, 1, 3)
    pool = build_pool_from_integrals(h, eri)
    t2 = 0.04 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    return pool, gen


def _unprojected_similarity_target(pool, gen, mask):
    sigma = dense_masked_generator_sigma(gen, mask)
    return expm(-sigma) @ pool.dense_matrix() @ expm(sigma)


def test_effective_hamiltonian_dense_equals_expm_reference_with_singles():
    pool, gen = _build_small_pool_and_generator(seed=3, include_singles=True)
    channels = gen.generator_channels()
    mask = uniform_mask(len(channels))
    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))
    H_eff = effective_hamiltonian_dense(pool, gen, mask, proj)

    sigma = gen.dense_sigma()
    H = pool.dense_matrix()
    P = proj.dense_matrix(pool.n_orbitals)
    H_ref = P @ expm(-sigma) @ H @ expm(sigma) @ P
    assert np.allclose(H_eff, H_ref, atol=1e-8)


def test_zero_mask_gives_projected_hamiltonian():
    pool, gen = _build_small_pool_and_generator(seed=4)
    channels = gen.pair_rank_one_pool()
    mask = ChannelMask(weights=np.zeros(len(channels)))
    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))

    H_eff = effective_hamiltonian_dense(pool, gen, mask, proj)
    P = proj.dense_matrix(pool.n_orbitals)
    H = pool.dense_matrix()
    assert np.allclose(H_eff, P @ H @ P, atol=1e-10)


def test_similarity_sandwich_accepts_nonzero_singles_for_oracle_path():
    pool, gen = _build_small_pool_and_generator(seed=5, include_singles=True)
    mask = uniform_mask(len(gen.generator_channels()))
    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))

    sw = build_similarity_sandwich(pool, gen, mask, proj, exp_eps=2e-2)
    assert np.allclose(sw.H_eff_dense, effective_hamiltonian_dense(pool, gen, mask, proj), atol=1e-10)
    assert sw.circuit.gates[0].kind == "U_sigma_oracle"
    assert sw.circuit.gates[1].kind == "W_H_oracle"
    assert sw.circuit.gates[2].kind == "U_sigma_oracle"


def test_compile_once_redial_keeps_fixed_compiled_structure_and_allows_only_parameter_changes():
    pool, gen = _build_small_pool_and_generator(seed=6)
    channels = gen.pair_rank_one_pool()
    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))
    sigma_singvals = np.array([ch.sigma for ch in channels])

    sw_full = build_similarity_sandwich(pool, gen, uniform_mask(len(channels)), proj, exp_eps=2e-2)
    sw_sparse = sw_full.redial_mask(top_k_mask(sigma_singvals, k=max(0, len(channels) // 2)))
    sw_frac = sw_full.redial_mask(ChannelMask(weights=0.23 * np.ones(len(channels))))

    assert sw_full.topology_hash == sw_sparse.topology_hash == sw_frac.topology_hash
    assert sw_full.compiled_signature_hash == sw_sparse.compiled_signature_hash == sw_frac.compiled_signature_hash
    assert sw_full.circuit.compiled_signature() == sw_sparse.circuit.compiled_signature()
    assert sw_sparse.circuit.compiled_signature() == sw_frac.circuit.compiled_signature()

    assert sw_full.circuit.gates[0].kind == "U_sigma_oracle"
    assert sw_full.circuit.gates[1].kind == "W_H_oracle"
    assert sw_full.circuit.gates[2].kind == "U_sigma_oracle"
    assert all(g.implementation_tag() == "subcircuit" for g in sw_full.circuit.gates)
    assert sw_full.circuit.resource_summary().subcircuit_call_count == 3
    assert (
        sw_full.circuit.gates[0].subcircuit.compiled_signature_hash()
        == sw_full.generator_exp_oracle.unitary_circuit.compiled_signature_hash()
    )
    assert sw_full.circuit.gates[1].subcircuit.compiled_signature_hash() == sw_full.hamiltonian_oracle.circuit.compiled_signature_hash()
    assert (
        sw_full.circuit.gates[2].subcircuit.compiled_signature_hash()
        == sw_full.generator_exp_oracle.unitary_circuit.inverse().compiled_signature_hash()
    )
    assert sw_full.circuit.gates[1].subcircuit.compiled_signature_hash() == sw_sparse.circuit.gates[1].subcircuit.compiled_signature_hash()
    assert sw_full.circuit.gates[0].subcircuit.compiled_signature_hash() == sw_sparse.circuit.gates[0].subcircuit.compiled_signature_hash()
    assert sw_full.circuit.gates[2].subcircuit.compiled_signature_hash() == sw_sparse.circuit.gates[2].subcircuit.compiled_signature_hash()

    exp_full = sw_full.generator_exp_oracle
    exp_sparse = sw_sparse.generator_exp_oracle
    exp_frac = sw_frac.generator_exp_oracle
    assert exp_full.circuit.compiled_signature_hash() == exp_sparse.circuit.compiled_signature_hash()
    assert exp_sparse.circuit.compiled_signature_hash() == exp_frac.circuit.compiled_signature_hash()
    assert exp_full.unitary_circuit.compiled_signature_hash() == exp_sparse.unitary_circuit.compiled_signature_hash()
    assert exp_sparse.unitary_circuit.compiled_signature_hash() == exp_frac.unitary_circuit.compiled_signature_hash()
    assert exp_full.cos_qsp_circuit.compiled_signature_hash() == exp_sparse.cos_qsp_circuit.compiled_signature_hash()
    assert exp_full.sin_qsp_circuit.compiled_signature_hash() == exp_sparse.sin_qsp_circuit.compiled_signature_hash()
    assert exp_full.sigma_oracle.circuit.compiled_signature_hash() == exp_sparse.sigma_oracle.circuit.compiled_signature_hash()
    assert np.allclose(exp_full.cos_phases, exp_sparse.cos_phases, atol=1e-12)
    assert np.allclose(exp_full.sin_phases, exp_sparse.sin_phases, atol=1e-12)
    assert np.isclose(sw_full.compiled_alpha_bar, sw_sparse.compiled_alpha_bar)
    assert np.allclose(
        exp_full.sigma_oracle.circuit.gates[1].matrix,
        exp_sparse.sigma_oracle.circuit.gates[1].matrix,
        atol=1e-12,
    )
    assert not np.allclose(
        exp_full.sigma_oracle.circuit.gates[0].matrix,
        exp_sparse.sigma_oracle.circuit.gates[0].matrix,
        atol=1e-8,
    )
    assert not np.allclose(sw_full.H_eff_dense, sw_sparse.H_eff_dense, atol=1e-6)


def test_similarity_sandwich_ancilla_zero_block_matches_reported_block_across_masks():
    pool, gen = _build_low_ancilla_pool_and_generator(seed=7)
    channels = gen.pair_rank_one_pool()
    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))
    sigma_singvals = np.array([ch.sigma for ch in channels])

    sw_full = build_similarity_sandwich(pool, gen, uniform_mask(len(channels)), proj, exp_eps=2e-2)
    masks = [
        sw_full.mask,
        top_k_mask(sigma_singvals, k=max(0, len(channels) - 1)),
        ChannelMask(weights=0.35 * np.ones(len(channels))),
    ]

    for mask in masks:
        sw = sw_full if np.allclose(mask.weights, sw_full.mask.weights) else sw_full.redial_mask(mask)
        U = circuit_unitary(sw.circuit)
        dim_sys = 2 ** pool.n_orbitals
        ancilla_zero_block = U[:dim_sys, :dim_sys]
        exact_unprojected = _unprojected_similarity_target(pool, gen, mask)
        exact_ref = effective_hamiltonian_dense(pool, gen, mask, proj)
        P = proj.dense_matrix(pool.n_orbitals)

        assert np.allclose(ancilla_zero_block, sw.encoded_system_block_dense, atol=1e-10)
        assert np.allclose(
            sw.generator_exp_oracle.unitary_zero_block_dense,
            expm(dense_masked_generator_sigma(gen, mask)),
            atol=5e-3,
        )
        assert np.allclose(sw.alpha * ancilla_zero_block, exact_unprojected, atol=5e-3)
        assert np.allclose(sw.H_eff_dense, exact_ref, atol=1e-10)
        assert np.allclose(P @ (sw.alpha * ancilla_zero_block) @ P, sw.H_eff_dense, atol=5e-3)


def test_top_k_mask_matches_effective_hamiltonian():
    pool, gen = _build_small_pool_and_generator(seed=8, include_singles=True)
    channels = gen.generator_channels()
    magnitudes = np.array([np.linalg.norm(ch.dense_sigma()) for ch in channels])
    k = max(0, len(channels) - 1)
    mask = top_k_mask(magnitudes, k=k)

    order = np.argsort(magnitudes)[::-1]
    sigma_ref = np.zeros_like(gen.dense_sigma())
    for mu in order[:k]:
        sigma_ref += channels[mu].dense_sigma()

    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))
    H_eff = effective_hamiltonian_dense(pool, gen, mask, proj)
    H = pool.dense_matrix()
    P = proj.dense_matrix(pool.n_orbitals)
    H_ref = P @ expm(-sigma_ref) @ H @ expm(sigma_ref) @ P
    assert np.allclose(H_eff, H_ref, atol=1e-8)


def test_similarity_sandwich_alpha_tracks_composed_block_encodings():
    pool, gen = _build_small_pool_and_generator(seed=9)
    mask = uniform_mask(len(gen.pair_rank_one_pool()))
    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))
    sw = build_similarity_sandwich(pool, gen, mask, proj, exp_eps=2e-2)

    expected = sw.hamiltonian_oracle.alpha
    assert np.isclose(sw.alpha, expected)


def test_similarity_sandwich_resources_track_real_compiled_oracles():
    pool, gen = _build_small_pool_and_generator(seed=11)
    mask = uniform_mask(len(gen.pair_rank_one_pool()))
    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))
    sw = build_similarity_sandwich(pool, gen, mask, proj, exp_eps=2e-2)

    resources = sw.resources
    assert resources.alpha == sw.alpha
    assert resources.n_system == pool.n_orbitals
    assert resources.n_ancilla == max(sw.hamiltonian_oracle.n_ancilla, sw.generator_exp_oracle.n_ancilla)
    assert resources.compiled_alpha_bar == sw.compiled_alpha_bar
    assert resources.projector_rank == len(proj.determinants)
    assert resources.u_sigma_call_count == 2
    assert resources.hamiltonian_oracle == sw.hamiltonian_oracle.resources
    assert resources.generator_exp_oracle == sw.generator_exp_oracle.resources
    assert resources.u_sigma_circuit == sw.generator_exp_oracle.unitary_circuit.resource_summary()
    assert resources.circuit == sw.circuit.resource_summary()
    assert resources.circuit.subcircuit_call_count == 3
    assert resources.circuit.gate_count_by_kind == {"U_sigma_oracle": 2, "W_H_oracle": 1}


def test_channel_mask_rejects_negative_weights():
    with pytest.raises(ValueError):
        ChannelMask(weights=np.array([1.0, -0.5]))


def test_channel_mask_with_alpha_bar_absorbs_residual():
    mask = ChannelMask(weights=np.array([0.3, 0.2]))
    out = mask.with_alpha_bar(1.0)
    assert np.isclose(out.total, 1.0)
    assert np.isclose(out.null_weight, 0.5)


def test_channel_mask_with_compiled_alpha_bar_absorbs_branch_scaled_residual():
    mask = ChannelMask(weights=np.array([1.0, 0.5]))
    out = mask.with_compiled_alpha_bar(np.array([2.0, 3.0]), alpha_bar=5.0)
    assert np.isclose(out.null_weight, 1.5)


def test_channel_mask_with_alpha_bar_errors_when_too_small():
    mask = ChannelMask(weights=np.array([0.6, 0.5]))
    with pytest.raises(ValueError):
        mask.with_alpha_bar(0.5)


def test_mask_length_must_match_compiled_generator_pool():
    pool, gen = _build_small_pool_and_generator(seed=10)
    proj = ModelSpaceProjector(determinants=(3, 5, 6, 9, 10, 12))
    channels = gen.generator_channels()
    bad_mask = ChannelMask(weights=np.ones(len(channels) - 1))

    with pytest.raises(ValueError, match="compiled generator pool"):
        effective_hamiltonian_dense(pool, gen, bad_mask, proj)
    with pytest.raises(ValueError, match="compiled generator pool"):
        build_similarity_sandwich(pool, gen, bad_mask, proj)


def test_uniform_mask_shape():
    mask = uniform_mask(4)
    assert mask.weights.shape == (4,)
    assert np.allclose(mask.weights, 1.0)


def test_top_k_mask_preserves_top_singular_values():
    sigma = np.array([0.3, 2.0, 0.1, 1.0])
    mask = top_k_mask(sigma, k=2)
    expected = np.array([0.0, 1.0, 0.0, 1.0])
    assert np.allclose(mask.weights, expected)


def test_model_space_projector_is_idempotent():
    proj = ModelSpaceProjector(determinants=(0, 3, 5))
    P = proj.dense_matrix(3)
    assert np.allclose(P @ P, P, atol=1e-12)
    assert np.allclose(P, P.conj().T, atol=1e-12)
    assert np.isclose(np.trace(P).real, 3.0)
