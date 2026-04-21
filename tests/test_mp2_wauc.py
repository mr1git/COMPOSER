"""Tests for MP2 amplitudes, the wAUC subspace diagnostic, and the
App E.3 one-shot cumulative-coverage mask builder.
"""
from __future__ import annotations

import numpy as np
import pytest

from composer.diagnostics.mask_selection import (
    channel_weights_mp2,
    cumulative_coverage_mask,
)
from composer.diagnostics.mp2 import mp2_doubles_amplitudes, mp2_energy
from composer.diagnostics.subspace import channel_overlap_matrix, rdm1_drift, wauc
from composer.factorization.pair_svd import PairChannel, pair_svd_decompose
from composer.utils.antisymmetric import pair_matrix_from_vector


def _make_random_eri(n: int, rng: np.random.Generator) -> np.ndarray:
    """Random physicist-order ERI with 8-fold real symmetry, built as
    ``<pq|rs> = sum_mu L_mu_{pq} L_mu_{rs}`` with real symmetric
    ``L_mu`` (chemist convention), transposed to physicist order.
    """
    eri_chem = np.zeros((n, n, n, n))
    for _ in range(2):
        A = rng.normal(size=(n, n))
        L = 0.5 * (A + A.T)
        eri_chem += np.einsum("pq,rs->pqrs", L, L)
    return eri_chem.transpose(0, 2, 1, 3) * 0.1


def test_mp2_amplitudes_are_antisymmetric():
    rng = np.random.default_rng(0)
    NO, NV = 2, 3
    n = NO + NV
    eri = _make_random_eri(n, rng)
    eps_occ = np.array([-1.0, -0.8])
    eps_vir = np.array([0.5, 0.7, 0.9])
    t2 = mp2_doubles_amplitudes(eri, eps_occ, eps_vir)
    assert t2.shape == (NV, NV, NO, NO)
    # antisymmetry in (a, b)
    assert np.allclose(t2, -t2.transpose(1, 0, 2, 3), atol=1e-12)
    # antisymmetry in (i, j)
    assert np.allclose(t2, -t2.transpose(0, 1, 3, 2), atol=1e-12)


def test_mp2_energy_matches_direct_sum():
    rng = np.random.default_rng(1)
    NO, NV = 2, 2
    n = NO + NV
    eri = _make_random_eri(n, rng)
    eps_occ = np.array([-1.2, -0.9])
    eps_vir = np.array([0.6, 0.8])
    E = mp2_energy(eri, eps_occ, eps_vir)

    # Independent reference: direct double sum.
    ref = 0.0
    for a in range(NV):
        pa = NO + a
        for b in range(NV):
            pb = NO + b
            for i in range(NO):
                for j in range(NO):
                    antisym = eri[i, j, pa, pb] - eri[i, j, pb, pa]
                    denom = eps_occ[i] + eps_occ[j] - eps_vir[a] - eps_vir[b]
                    ref += 0.25 * (antisym.conjugate() * antisym / denom).real
    assert np.isclose(E, ref, atol=1e-10)


def test_mp2_zero_eri_gives_zero_amplitudes():
    NO, NV = 2, 2
    n = NO + NV
    eri = np.zeros((n, n, n, n))
    eps_occ = np.array([-1.0, -0.8])
    eps_vir = np.array([0.5, 0.7])
    t2 = mp2_doubles_amplitudes(eri, eps_occ, eps_vir)
    assert np.allclose(t2, 0.0)
    assert mp2_energy(eri, eps_occ, eps_vir) == 0.0


def test_mp2_rejects_near_zero_denominator():
    NO, NV = 2, 2
    n = NO + NV
    eri = np.zeros((n, n, n, n))
    eri[0, 1, 2, 3] = 1.0
    eri[0, 1, 3, 2] = -1.0
    eps_occ = np.array([-1.0, -0.5])
    eps_vir = np.array([-1.5, 0.0])  # denom for (a,b,i,j)=(0,1,0,1) is zero
    with pytest.raises(ValueError, match="near-zero MP2 denominator"):
        mp2_doubles_amplitudes(eri, eps_occ, eps_vir)


def test_wauc_self_overlap_is_one():
    rng = np.random.default_rng(2)
    NV, NO = 3, 2
    # Build some t2 and its pair-SVD channels.
    t2 = rng.normal(size=(NV, NV, NO, NO))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    channels = pair_svd_decompose(t2)
    # A against itself: every channel has a perfect match in B = A.
    assert np.isclose(wauc(channels, channels), 1.0, atol=1e-10)


def test_wauc_matches_eq_e7_e8_rank_weighting():
    # Pair-space basis vectors for NV=NO=3, so the pair basis has dimension 3.
    e0 = np.array([1.0, 0.0, 0.0])
    e1 = np.array([0.0, 1.0, 0.0])
    e2 = np.array([0.0, 0.0, 1.0])

    reference = [
        PairChannel(sigma=3.0, U=pair_matrix_from_vector(e0, 3), V=pair_matrix_from_vector(e0, 3)),
        PairChannel(sigma=1.0, U=pair_matrix_from_vector(e1, 3), V=pair_matrix_from_vector(e1, 3)),
    ]
    mixed = (e1 + e2) / np.sqrt(2.0)
    truncated = [
        PairChannel(sigma=3.0, U=pair_matrix_from_vector(e0, 3), V=pair_matrix_from_vector(e0, 3)),
        PairChannel(sigma=1.0, U=pair_matrix_from_vector(mixed, 3), V=pair_matrix_from_vector(mixed, 3)),
    ]

    # Eq. (E7): ov(1) = 1, ov(2) = (1 + 1/4) / 2 = 5/8.
    # Eq. (E8): weights are sigma^2-normalized => [9/10, 1/10].
    expected = (9.0 / 10.0) * 1.0 + (1.0 / 10.0) * (5.0 / 8.0)
    assert np.isclose(wauc(reference, truncated), expected, atol=1e-12)


def test_wauc_empty_truncated_is_zero():
    rng = np.random.default_rng(3)
    NV, NO = 3, 2
    t2 = rng.normal(size=(NV, NV, NO, NO))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    channels = pair_svd_decompose(t2)
    assert wauc(channels, []) == 0.0
    assert wauc([], channels) == 0.0


def test_wauc_monotone_in_inclusion():
    rng = np.random.default_rng(4)
    NV, NO = 3, 2
    t2 = rng.normal(size=(NV, NV, NO, NO))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    channels = pair_svd_decompose(t2)
    # wAUC(A, B_k) with growing B_k should be non-decreasing
    vals = [wauc(channels, channels[:k]) for k in range(1, len(channels) + 1)]
    for a, b in zip(vals, vals[1:]):
        assert b + 1e-12 >= a


def test_channel_overlap_matrix_is_bounded():
    rng = np.random.default_rng(5)
    NV, NO = 3, 2
    t2 = rng.normal(size=(NV, NV, NO, NO))
    t2 = t2 - t2.transpose(1, 0, 2, 3)
    t2 = t2 - t2.transpose(0, 1, 3, 2)
    ch = pair_svd_decompose(t2)
    O = channel_overlap_matrix(ch, ch)
    assert O.shape == (len(ch), len(ch))
    assert np.all(O >= -1e-12)
    assert np.all(O <= 1.0 + 1e-12)
    # Self-overlap: diagonal is 1 (left/right singular vectors orthonormal
    # within the pair-SVD).
    assert np.allclose(np.diag(O), 1.0, atol=1e-8)


def test_channel_weights_mp2_respects_sigma_sq_and_pair_norms():
    base_u = pair_matrix_from_vector(np.array([1.0, 0.0, 0.0]), 3)
    base_v = pair_matrix_from_vector(np.array([0.0, 1.0, 0.0]), 3)
    channels = [
        PairChannel(sigma=2.0, U=base_u, V=base_v),
        PairChannel(sigma=1.5, U=2.0 * base_u, V=base_v),
    ]
    weights = channel_weights_mp2(channels)
    assert np.allclose(weights, np.array([16.0, 36.0]))


def test_cumulative_coverage_mask_selects_by_mp2_weight():
    base_u = pair_matrix_from_vector(np.array([1.0, 0.0, 0.0]), 3)
    base_v = pair_matrix_from_vector(np.array([0.0, 1.0, 0.0]), 3)
    channels = [
        PairChannel(sigma=2.0, U=base_u, V=base_v),
        PairChannel(sigma=1.5, U=2.0 * base_u, V=base_v),
        PairChannel(sigma=0.25, U=base_u, V=base_v),
    ]
    # MP2 ladder weights are [16, 36, 0.25]. A 60% coverage target must
    # therefore keep only channel 1, even though channel 0 has the larger sigma.
    mask = cumulative_coverage_mask(channels, coverage=0.6)
    assert np.allclose(mask.weights, np.array([0.0, 1.0, 0.0]))


def test_cumulative_coverage_mask_selects_binary_prefix():
    # Build synthetic channels with known sigmas; coverage is driven by
    # sigma^2 because the pair norms are equal across channels.
    U_dummy = pair_matrix_from_vector(np.array([1.0, 0.0, 0.0]), 3)
    V_dummy = pair_matrix_from_vector(np.array([0.0, 1.0, 0.0]), 3)
    channels = [
        PairChannel(sigma=2.0, U=U_dummy, V=V_dummy),
        PairChannel(sigma=1.0, U=U_dummy, V=V_dummy),
        PairChannel(sigma=0.5, U=U_dummy, V=V_dummy),
        PairChannel(sigma=0.1, U=U_dummy, V=V_dummy),
    ]
    # coverage = 0.85 -> target weight 0.85 * (4 + 1 + 0.25 + 0.01) = 4.4625.
    # Top-1 gives 4.0, top-2 gives 5.0, so the selector keeps the top two.
    mask = cumulative_coverage_mask(channels, coverage=0.85)
    assert np.allclose(mask.weights, np.array([1.0, 1.0, 0.0, 0.0]))


def test_cumulative_coverage_mask_full_coverage_selects_all():
    NV, NO = 3, 2
    # canonical pair vectors (don't actually matter here)
    U = np.zeros((NV, NV))
    U[0, 1] = 1 / np.sqrt(2)
    U[1, 0] = -1 / np.sqrt(2)
    V = np.zeros((NO, NO))
    V[0, 1] = 1 / np.sqrt(2)
    V[1, 0] = -1 / np.sqrt(2)
    channels = [PairChannel(sigma=s, U=U, V=V) for s in [1.0, 0.5, 0.25]]
    mask = cumulative_coverage_mask(channels, coverage=1.0)
    assert np.allclose(mask.weights, 1.0)


def test_cumulative_coverage_mask_rejects_bad_coverage():
    with pytest.raises(ValueError):
        cumulative_coverage_mask([], coverage=1.5)
    with pytest.raises(ValueError):
        cumulative_coverage_mask([], coverage=0.0)


def test_rdm1_drift_norm():
    A = np.eye(3)
    B = np.eye(3)
    B[0, 0] = 1.5
    assert np.isclose(rdm1_drift(A, B), 0.5)


def test_rdm1_drift_shape_mismatch():
    with pytest.raises(ValueError):
        rdm1_drift(np.zeros((3, 3)), np.zeros((2, 3)))
