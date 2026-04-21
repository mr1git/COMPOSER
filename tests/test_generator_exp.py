"""Tests for the pair-SVD factorization, cluster generator, and the
generator-exp oracle/QSP path.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from composer.block_encoding.generator_exp import (
    build_generator_exp_oracle,
    build_sigma_pool_oracle,
    dense_generator_exp_reference,
    dense_masked_generator_sigma,
    dense_masked_doubles_sigma,
    generator_exp_top_left_block,
    hermitian_fock_block_encoding,
    matrix_chebyshev_eval,
)
from composer.circuits.gate import CircuitCall, SelectGate
from composer.circuits.simulator import unitary as circuit_unitary
from composer.factorization.pair_svd import pair_svd_decompose, reconstruct_t2
from composer.operators.generator import build_cluster_generator
from composer.operators.mask import ChannelMask, uniform_mask
from composer.qsp.chebyshev import jacobi_anger_coefficients, recommended_degree
from composer.utils import fermion as jw
from composer.utils.antisymmetric import pair_matrix_from_vector


def _random_antisymmetric_t2(NV: int, NO: int, rng: np.random.Generator) -> np.ndarray:
    t2 = rng.normal(size=(NV, NV, NO, NO)) + 1j * rng.normal(size=(NV, NV, NO, NO))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    return t2


def _direct_doubles_excitation(t2: np.ndarray, NO: int) -> np.ndarray:
    NV = t2.shape[0]
    n = NO + NV
    dim = 2**n
    adag = [jw.jw_a_dagger(p, n) for p in range(n)]
    a_ = [jw.jw_a(p, n) for p in range(n)]
    T2 = np.zeros((dim, dim), dtype=complex)
    for a in range(NV):
        p = NO + a
        for b in range(NV):
            q = NO + b
            for i in range(NO):
                for j in range(NO):
                    c = t2[a, b, i, j]
                    if c != 0:
                        T2 += 0.25 * c * (adag[p] @ adag[q] @ a_[j] @ a_[i])
    return T2


def test_pair_svd_reconstructs_t2():
    rng = np.random.default_rng(0)
    NV, NO = 3, 2
    t2 = _random_antisymmetric_t2(NV, NO, rng)
    channels = pair_svd_decompose(t2)
    rebuilt = reconstruct_t2(channels, NV, NO)
    assert np.allclose(rebuilt, t2, atol=1e-8)


def test_pair_svd_recovers_known_singular_channels():
    NV = NO = 3
    u1 = pair_matrix_from_vector(np.array([1.0, 0.0, 0.0]), NV)
    u2 = pair_matrix_from_vector(np.array([0.0, 1.0, 0.0]), NV)
    v1 = pair_matrix_from_vector(np.array([0.0, 0.0, 1.0]), NO)
    v2 = pair_matrix_from_vector(np.array([0.0, 1.0, 0.0]), NO)
    t2 = 3.0 * np.einsum("ab,ij->abij", u1, v1) + 0.5 * np.einsum("ab,ij->abij", u2, v2)

    channels = pair_svd_decompose(t2, tol=1e-14)

    assert len(channels) == 2
    assert np.allclose([ch.sigma for ch in channels], [3.0, 0.5], atol=1e-12)
    assert np.isclose(abs(np.vdot(channels[0].U, u1)) / 2.0, 1.0, atol=1e-12)
    assert np.isclose(abs(np.vdot(channels[0].V, v1)) / 2.0, 1.0, atol=1e-12)
    assert np.isclose(abs(np.vdot(channels[1].U, u2)) / 2.0, 1.0, atol=1e-12)
    assert np.isclose(abs(np.vdot(channels[1].V, v2)) / 2.0, 1.0, atol=1e-12)


def test_pair_svd_channel_pair_space_matrix_matches_unfolded_tensor():
    rng = np.random.default_rng(7)
    NV, NO = 3, 3
    t2 = _random_antisymmetric_t2(NV, NO, rng)
    channels = pair_svd_decompose(t2)

    direct = np.zeros((NV * (NV - 1) // 2, NO * (NO - 1) // 2), dtype=complex)
    row = 0
    for a in range(NV):
        for b in range(a + 1, NV):
            col = 0
            for i in range(NO):
                for j in range(i + 1, NO):
                    direct[row, col] = t2[a, b, i, j]
                    col += 1
            row += 1

    rebuilt = sum((ch.pair_space_matrix() for ch in channels), start=np.zeros_like(direct))
    assert np.allclose(rebuilt, direct, atol=1e-8)


def test_pair_svd_channel_normalization():
    rng = np.random.default_rng(1)
    NV, NO = 3, 2
    t2 = _random_antisymmetric_t2(NV, NO, rng)
    channels = pair_svd_decompose(t2)
    for ch in channels:
        u_norm = np.linalg.norm(ch.U[np.triu_indices(NV, k=1)])
        v_norm = np.linalg.norm(ch.V[np.triu_indices(NO, k=1)])
        assert np.isclose(u_norm, 1.0, atol=1e-8)
        assert np.isclose(v_norm, 1.0, atol=1e-8)


def test_cluster_generator_doubles_excitation_matches_direct_operator():
    rng = np.random.default_rng(11)
    NO, NV = 2, 3
    t2 = 0.1 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)

    direct = _direct_doubles_excitation(gen.t2, NO)
    from_channels = gen.dense_doubles_excitation()
    assert np.allclose(from_channels, direct, atol=1e-10)


def test_cluster_generator_exposes_explicit_embedded_doubles_channels():
    rng = np.random.default_rng(111)
    NO, NV = 2, 3
    t2 = 0.1 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)

    channels = gen.doubles_channels()
    assert all(ch.creation_orbitals == (2, 3, 4) for ch in channels)
    assert all(ch.annihilation_orbitals == (0, 1) for ch in channels)

    t2_from_channels = np.zeros_like(gen.t2)
    dense_from_channels = np.zeros_like(gen.dense_doubles_excitation())
    for ch in channels:
        term = ch.as_pair_rank_one()
        assert np.allclose(term.coefficient_tensor(), ch.coefficient_tensor(), atol=1e-12)
        t2_from_channels += ch.coefficient_tensor()
        dense_from_channels += ch.dense_excitation()

    assert np.allclose(t2_from_channels, gen.t2, atol=1e-8)
    assert np.allclose(dense_from_channels, gen.dense_doubles_excitation(), atol=1e-10)


def test_cluster_generator_doubles_sigma_matches_direct_sigma():
    rng = np.random.default_rng(12)
    NO, NV = 2, 2
    t2 = 0.05 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)

    assert np.allclose(gen.dense_doubles_sigma(), gen.dense_sigma(), atol=1e-10)


def test_cluster_generator_channel_sigmas_sum_to_dense_sigma():
    rng = np.random.default_rng(13)
    NO, NV = 2, 2
    t2 = 0.05 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)

    sigma_from_channels = np.zeros_like(gen.dense_doubles_sigma())
    for ch in gen.doubles_channels():
        sigma_from_channels += ch.dense_sigma()

    assert np.allclose(sigma_from_channels, gen.dense_doubles_sigma(), atol=1e-10)


def test_cluster_generator_exposes_explicit_singles_channels():
    NO = NV = 2
    t1 = np.array(
        [
            [0.12 + 0.03j, 0.0],
            [0.0, -0.07j],
        ],
        dtype=complex,
    )
    gen = build_cluster_generator(NO, NV, t1=t1, t2=None)

    channels = gen.singles_channels()
    assert len(channels) == 2
    assert [(ch.creation_orbital, ch.annihilation_orbital) for ch in channels] == [(2, 0), (3, 1)]
    assert np.allclose([ch.coeff for ch in channels], [t1[0, 0], t1[1, 1]])

    sigma_from_channels = sum((ch.dense_sigma() for ch in channels), start=np.zeros_like(gen.dense_sigma()))
    assert np.allclose(sigma_from_channels, gen.dense_sigma(), atol=1e-10)


def test_cluster_generator_sigma_is_anti_hermitian():
    rng = np.random.default_rng(2)
    NO, NV = 2, 2
    t1 = rng.normal(size=(NV, NO)) * 0.1
    t2 = _random_antisymmetric_t2(NV, NO, rng) * 0.05
    gen = build_cluster_generator(NO, NV, t1=t1, t2=t2)
    sigma = gen.dense_sigma()
    assert np.allclose(sigma, -sigma.conj().T, atol=1e-10)


def test_sigma_pool_oracle_top_left_block_matches_masked_doubles_generator():
    rng = np.random.default_rng(21)
    NO = NV = 3
    t2 = 0.02 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    channels = gen.pair_rank_one_pool()
    mask = ChannelMask(weights=np.linspace(1.0, 0.2, len(channels)))

    oracle = build_sigma_pool_oracle(gen, mask)
    sigma_masked = dense_masked_doubles_sigma(gen, mask)

    assert np.allclose(
        oracle.ancilla_zero_block_dense,
        (-1j * sigma_masked) / oracle.alpha,
        atol=1e-10,
    )
    assert np.allclose(
        oracle.sigma_zero_block_dense,
        sigma_masked / oracle.alpha,
        atol=1e-10,
    )
    assert np.allclose(circuit_unitary(oracle.circuit), oracle.W, atol=1e-10)


def test_sigma_pool_oracle_top_left_block_matches_masked_full_generator_with_singles():
    rng = np.random.default_rng(211)
    NO = NV = 2
    t1 = np.array(
        [
            [0.03 + 0.01j, -0.02j],
            [0.0, -0.015 + 0.005j],
        ],
        dtype=complex,
    )
    t2 = 0.02 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=t1, t2=t2)
    channels = gen.generator_channels()
    mask = ChannelMask(weights=np.linspace(1.0, 0.4, len(channels)))

    oracle = build_sigma_pool_oracle(gen, mask)
    sigma_masked = dense_masked_generator_sigma(gen, mask)

    assert np.allclose(
        oracle.ancilla_zero_block_dense,
        (-1j * sigma_masked) / oracle.alpha,
        atol=1e-10,
    )
    assert np.allclose(
        oracle.sigma_zero_block_dense,
        sigma_masked / oracle.alpha,
        atol=1e-10,
    )


def test_sigma_pool_oracle_changes_only_prep_under_mask_update():
    rng = np.random.default_rng(22)
    NO = NV = 3
    t2 = 0.02 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    channels = gen.pair_rank_one_pool()

    oracle_a = build_sigma_pool_oracle(gen, uniform_mask(len(channels)))
    oracle_b = build_sigma_pool_oracle(gen, ChannelMask(weights=0.37 * np.ones(len(channels))))

    assert [g.kind for g in oracle_a.circuit.gates] == ["PREP_sigma", "SELECT_sigma", "PREP_sigma"]
    assert oracle_a.circuit.gates[0].num_qubits == oracle_a.selector_width
    assert oracle_a.circuit.gates[1].num_qubits > gen.n_orbitals
    assert np.allclose(oracle_a.circuit.gates[1].matrix, oracle_b.circuit.gates[1].matrix, atol=1e-12)
    assert not np.allclose(oracle_a.circuit.gates[0].matrix, oracle_b.circuit.gates[0].matrix, atol=1e-8)
    assert np.allclose(oracle_a.channel_norms, oracle_b.channel_norms, atol=1e-12)


def test_sigma_pool_oracle_resources_track_compiled_selector_construction():
    rng = np.random.default_rng(220)
    NO = NV = 3
    t2 = 0.02 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    channels = gen.pair_rank_one_pool()
    oracle = build_sigma_pool_oracle(gen, uniform_mask(len(channels)))

    resources = oracle.resources
    assert resources.alpha == oracle.alpha
    assert resources.n_system == gen.n_orbitals
    assert resources.n_ancilla == oracle.n_ancilla
    assert resources.selector_width == oracle.selector_width
    assert resources.active_branch_count == len(channels)
    assert resources.compiled_branch_count == len(channels) + 1
    assert resources.null_branch_index == len(channels)
    assert resources.circuit == oracle.circuit.resource_summary()
    assert resources.circuit.gate_count_by_kind == {"PREP_sigma": 2, "SELECT_sigma": 1}


def test_sigma_pool_oracle_null_branch_is_compiled_behavior():
    rng = np.random.default_rng(23)
    NO = NV = 3
    t2 = 0.02 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    channels = gen.pair_rank_one_pool()

    base = ChannelMask(weights=0.41 * np.ones(len(channels)))
    oracle_base = build_sigma_pool_oracle(gen, base)
    padded_alpha = oracle_base.alpha + 2.0
    padded = base.with_compiled_alpha_bar(oracle_base.channel_norms, alpha_bar=padded_alpha)
    oracle_padded = build_sigma_pool_oracle(gen, padded, alpha_bar=padded_alpha)

    expected_null = padded_alpha - base.compiled_weight_sum(oracle_base.channel_norms)
    assert np.isclose(padded.null_weight, expected_null)
    assert np.isclose(
        oracle_padded.prep_branch_weights[oracle_padded.null_branch_index],
        padded.null_weight,
    )
    assert oracle_padded.alpha > oracle_base.alpha
    assert not np.allclose(
        oracle_base.ancilla_zero_block_dense,
        oracle_padded.ancilla_zero_block_dense,
        atol=1e-10,
    )


def test_hermitian_fock_block_encoding_is_unitary():
    rng = np.random.default_rng(3)
    n = 3
    M = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
    H = 0.5 * (M + M.conj().T)
    W, alpha = hermitian_fock_block_encoding(H)
    assert np.allclose(W @ W.conj().T, np.eye(W.shape[0]), atol=1e-9)
    assert np.allclose(W[: 2**n, : 2**n] * alpha, H, atol=1e-9)


def test_matrix_chebyshev_eval_matches_polyval():
    # Compare with numpy.polynomial.chebyshev on a diagonal matrix.
    coeffs = np.array([0.3, -0.2, 0.5, 0.1])
    xs = np.array([-0.7, 0.1, 0.4])
    A = np.diag(xs).astype(complex)
    from numpy.polynomial.chebyshev import chebval

    expected = np.diag(chebval(xs, coeffs))
    got = matrix_chebyshev_eval(coeffs, A)
    assert np.allclose(got, expected, atol=1e-12)


def test_generator_exp_dense_reference_approximates_expm():
    rng = np.random.default_rng(4)
    n = 3
    M = rng.normal(size=(2**n, 2**n)) * 0.2 + 1j * rng.normal(size=(2**n, 2**n)) * 0.2
    sigma = M - M.conj().T  # anti-Hermitian
    approx = dense_generator_exp_reference(sigma, eps=1e-10)
    exact = expm(sigma)
    assert np.max(np.abs(approx - exact)) < 1e-8


def test_generator_exp_dense_reference_small_amplitude():
    # Tiny sigma should give nearly identity.
    n = 2
    sigma = np.zeros((2**n, 2**n), dtype=complex)
    sigma[0, 1] = 0.01
    sigma[1, 0] = -0.01
    approx = dense_generator_exp_reference(sigma, eps=1e-12)
    exact = expm(sigma)
    assert np.allclose(approx, exact, atol=1e-10)


def test_generator_exp_dense_reference_matches_direct_truncated_chebyshev_series():
    kappa = 2.0
    eps = 0.8
    H = np.diag(np.array([-2.0, -0.5, 0.0, 1.0, 2.0], dtype=float))
    sigma = 1j * H

    approx = dense_generator_exp_reference(sigma, eps=eps)
    degree = recommended_degree(kappa, eps)
    coeffs = jacobi_anger_coefficients(-kappa, degree)
    direct_poly = matrix_chebyshev_eval(coeffs, H / kappa)
    exact = expm(sigma)

    assert np.allclose(approx, direct_poly, atol=1e-12)
    err = np.max(np.abs(approx - exact))
    assert 1e-8 < err < 1e-4


def test_generator_exp_oracle_matches_expm_small_system():
    rng = np.random.default_rng(24)
    NO = NV = 2
    t2 = 0.08 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    mask = uniform_mask(len(gen.pair_rank_one_pool()))

    sigma_oracle = build_sigma_pool_oracle(gen, mask)
    exp_oracle = build_generator_exp_oracle(sigma_oracle, eps=2e-2, qsp_max_iter=600)
    sigma = dense_masked_doubles_sigma(gen, mask)
    exact = expm(sigma)

    assert np.max(np.abs(exp_oracle.exp_zero_block_dense - exact)) < 5e-3
    assert np.allclose(generator_exp_top_left_block(sigma_oracle, eps=2e-2, qsp_max_iter=600), exact, atol=5e-3)


def test_generator_exp_oracle_matches_expm_with_singles_and_doubles():
    rng = np.random.default_rng(241)
    NO = NV = 2
    t1 = np.array(
        [
            [0.025 + 0.01j, -0.015j],
            [0.0, -0.02 + 0.005j],
        ],
        dtype=complex,
    )
    t2 = 0.04 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=t1, t2=t2)
    mask = ChannelMask(weights=np.linspace(1.0, 0.35, len(gen.generator_channels())))

    sigma_oracle = build_sigma_pool_oracle(gen, mask)
    exp_oracle = build_generator_exp_oracle(sigma_oracle, eps=2e-2, qsp_max_iter=600)
    sigma = dense_masked_generator_sigma(gen, mask)
    exact = expm(sigma)

    assert np.max(np.abs(exp_oracle.exp_zero_block_dense - exact)) < 5e-3


def test_generator_exp_oracle_resolves_cos_and_sin_blocks_before_final_lcu():
    rng = np.random.default_rng(25)
    NO = NV = 2
    t2 = 0.06 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    mask = uniform_mask(len(gen.pair_rank_one_pool()))

    sigma_oracle = build_sigma_pool_oracle(gen, mask)
    exp_oracle = build_generator_exp_oracle(sigma_oracle, eps=2e-2, qsp_max_iter=600)
    sigma = dense_masked_doubles_sigma(gen, mask)
    exact_pos = expm(sigma)
    exact_neg = expm(-sigma)
    cos_exact = 0.5 * (exact_pos + exact_neg)
    sin_exact = (exact_pos - exact_neg) / (2j)

    assert np.max(np.abs(exp_oracle.cos_zero_block_dense - cos_exact)) < 5e-3
    assert np.max(np.abs(exp_oracle.sin_zero_block_dense - sin_exact)) < 5e-3


def test_generator_exp_oracle_is_built_from_reusable_compiled_subcircuits():
    rng = np.random.default_rng(251)
    NO = NV = 2
    t2 = 0.06 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    mask = uniform_mask(len(gen.pair_rank_one_pool()))

    sigma_oracle = build_sigma_pool_oracle(gen, mask)
    exp_oracle = build_generator_exp_oracle(sigma_oracle, eps=2e-2, qsp_max_iter=600)

    assert isinstance(exp_oracle.wx_oracle_circuit.gates[1], CircuitCall)
    assert exp_oracle.wx_oracle_circuit.gates[1].subcircuit is sigma_oracle.circuit

    cos_calls = [g for g in exp_oracle.cos_qsp_circuit.gates if g.kind == "QSP_sigma_cos"]
    sin_calls = [g for g in exp_oracle.sin_qsp_circuit.gates if g.kind == "QSP_sigma_sin"]
    assert cos_calls and sin_calls
    assert all(isinstance(g, CircuitCall) for g in cos_calls)
    assert all(isinstance(g, CircuitCall) for g in sin_calls)
    assert all(g.subcircuit is exp_oracle.wx_oracle_circuit for g in cos_calls)
    assert all(g.subcircuit is exp_oracle.wx_oracle_circuit for g in sin_calls)

    assert isinstance(exp_oracle.cos_oracle_circuit.gates[1], SelectGate)
    assert exp_oracle.cos_oracle_circuit.gates[1].zero_circuit is exp_oracle.cos_qsp_circuit
    assert exp_oracle.cos_oracle_circuit.gates[1].one_circuit.compiled_signature_hash() == exp_oracle.cos_qsp_circuit.inverse().compiled_signature_hash()

    assert isinstance(exp_oracle.sin_oracle_circuit.gates[1], SelectGate)
    assert exp_oracle.sin_oracle_circuit.gates[1].zero_circuit is exp_oracle.sin_qsp_circuit
    assert exp_oracle.sin_oracle_circuit.gates[1].one_circuit.compiled_signature_hash() == exp_oracle.sin_qsp_circuit.inverse().compiled_signature_hash()

    assert [g.kind for g in exp_oracle.circuit.gates] == ["PREP_exp", "PHASE_exp", "SELECT_exp", "PREP_exp"]
    assert isinstance(exp_oracle.circuit.gates[2], SelectGate)
    assert exp_oracle.circuit.gates[2].zero_circuit is exp_oracle.cos_oracle_circuit
    assert exp_oracle.circuit.gates[2].one_circuit is exp_oracle.sin_oracle_circuit


def test_generator_exp_oracle_reuses_qsp_phase_lists_across_mask_updates():
    rng = np.random.default_rng(26)
    NO = NV = 2
    t2 = 0.03 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    channels = gen.pair_rank_one_pool()

    base = uniform_mask(len(channels))
    sigma_oracle = build_sigma_pool_oracle(gen, base)
    padded_alpha = sigma_oracle.alpha + 0.5
    mask_a = ChannelMask(weights=0.8 * np.ones(len(channels))).with_compiled_alpha_bar(
        sigma_oracle.channel_norms,
        alpha_bar=padded_alpha,
    )
    mask_b = ChannelMask(weights=0.3 * np.ones(len(channels))).with_compiled_alpha_bar(
        sigma_oracle.channel_norms,
        alpha_bar=padded_alpha,
    )

    exp_a = build_generator_exp_oracle(
        build_sigma_pool_oracle(gen, mask_a, alpha_bar=padded_alpha),
        eps=5e-2,
        qsp_max_iter=500,
    )
    exp_b = build_generator_exp_oracle(
        build_sigma_pool_oracle(gen, mask_b, alpha_bar=padded_alpha),
        eps=5e-2,
        qsp_max_iter=500,
    )

    assert exp_a.cos_degree == exp_b.cos_degree
    assert exp_a.sin_degree == exp_b.sin_degree
    assert np.allclose(exp_a.cos_phases, exp_b.cos_phases, atol=1e-12)
    assert np.allclose(exp_a.sin_phases, exp_b.sin_phases, atol=1e-12)


def test_generator_exp_resources_count_actual_qsp_queries():
    rng = np.random.default_rng(260)
    NO = NV = 2
    t2 = 0.03 * _random_antisymmetric_t2(NV, NO, rng)
    gen = build_cluster_generator(NO, NV, t1=None, t2=t2)
    sigma_oracle = build_sigma_pool_oracle(gen, uniform_mask(len(gen.pair_rank_one_pool())))
    exp_oracle = build_generator_exp_oracle(sigma_oracle, eps=5e-2, qsp_max_iter=500)

    resources = exp_oracle.resources
    assert resources.alpha == exp_oracle.alpha
    assert resources.n_system == gen.n_orbitals
    assert resources.n_ancilla == exp_oracle.n_ancilla
    assert resources.exp_sign == exp_oracle.exp_sign
    assert resources.cos_degree == exp_oracle.cos_degree
    assert resources.sin_degree == exp_oracle.sin_degree
    assert resources.cos_phase_count == len(exp_oracle.cos_phases)
    assert resources.sin_phase_count == len(exp_oracle.sin_phases)
    assert resources.cos_qsp_query_count == len(exp_oracle.cos_phases) - 1
    assert resources.sin_qsp_query_count == len(exp_oracle.sin_phases) - 1
    assert resources.sigma_oracle == sigma_oracle.resources
    assert resources.cos_qsp_circuit == exp_oracle.cos_qsp_circuit.resource_summary()
    assert resources.sin_qsp_circuit == exp_oracle.sin_qsp_circuit.resource_summary()
    assert resources.circuit == exp_oracle.circuit.resource_summary()
    assert resources.cos_qsp_circuit.subcircuit_call_count == resources.cos_qsp_query_count
    assert resources.sin_qsp_circuit.subcircuit_call_count == resources.sin_qsp_query_count
    assert resources.circuit.select_gate_count == 1
    assert resources.circuit.composite_gate_count == 1
