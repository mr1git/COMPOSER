"""Tests for pivoted Cholesky and the mean-field shift.

Level-1 primitive checks: factor reconstructs ERIs; shift identity
converts H (Eq 10) into the rank-one pool form (Eq 13-17).
"""
from __future__ import annotations

import numpy as np
import pytest

from composer.factorization.cholesky import cholesky_eri, reconstruct_eri
from composer.factorization.mean_field_shift import mean_field_shifted_h
from composer.operators.hamiltonian import HamiltonianPool, build_pool_from_integrals
from composer.utils import fermion as jw


def _random_symmetric_psd_eri(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random 4-index physicist-order ERI with the full
    8-fold real symmetry: ``<pq|rs> = <qp|sr>``, ``<pq|rs> = <rs|pq>``,
    and (for real orbitals) chemist ``(pq|rs) = (qp|rs)``.

    Strategy: build chemist ERIs from
    ``(pq|rs) = sum_mu Lmu[p,q] Lmu[r,s]`` for random real *symmetric*
    ``Lmu``; this automatically satisfies the 8-fold chemist symmetry.
    Then convert to physicist by
    ``<pq|rs> = (pr|qs)``, i.e., ``physicist = chemist.transpose(0, 2, 1, 3)``.
    """
    K = n + 2
    chemist = np.zeros((n, n, n, n))
    for _ in range(K):
        Lmu = rng.normal(size=(n, n))
        Lmu = 0.5 * (Lmu + Lmu.T)
        chemist += np.einsum("pq,rs->pqrs", Lmu, Lmu)
    # Convert chemist (pq|rs) to physicist <pq|rs> = (pr|qs)
    return chemist.transpose(0, 2, 1, 3)


def _random_complex_psd_eri(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a generic complex Hermitian-PSD physicist-order ERI.

    This respects only the Hermitian PSD structure of the matricized
    ``M[(p, r), (q, s)]`` tensor, not the additional real-orbital
    symmetries used by the Hamiltonian pool / LCU path.
    """
    A = rng.normal(size=(n * n, n * n)) + 1j * rng.normal(size=(n * n, n * n))
    M = A @ A.conj().T
    M = 0.5 * (M + M.conj().T)
    return M.reshape(n, n, n, n).transpose(0, 2, 1, 3)


def test_physicist_helper_is_not_vacuously_chemist_order():
    rng = np.random.default_rng(11)
    eri_phys = _random_symmetric_psd_eri(4, rng)
    eri_chem = eri_phys.transpose(0, 2, 1, 3)
    assert not np.allclose(eri_phys, eri_chem, atol=1e-12)


def test_pivoted_cholesky_reconstructs_eri():
    rng = np.random.default_rng(0)
    n = 4
    eri = _random_symmetric_psd_eri(n, rng)
    L = cholesky_eri(eri, threshold=1e-12)
    eri_rec = reconstruct_eri(L)
    # Tolerance tied to threshold
    assert np.allclose(eri_rec, eri, atol=1e-8), np.abs(eri_rec - eri).max()


def test_real_electronic_cholesky_factors_are_symmetric():
    rng = np.random.default_rng(12)
    eri = _random_symmetric_psd_eri(4, rng)
    L = cholesky_eri(eri, threshold=1e-12)
    assert np.allclose(L.imag, 0.0, atol=1e-12)
    assert np.allclose(L, L.transpose(0, 2, 1), atol=1e-10)


def test_cholesky_factor_shape_scales_with_n():
    rng = np.random.default_rng(1)
    ranks = []
    for n in [3, 4, 5]:
        eri = _random_symmetric_psd_eri(n, rng)
        L = cholesky_eri(eri, threshold=1e-10)
        ranks.append(L.shape[0])
    # Factor count bounded by n^2 and typically grows with n.
    for n, K in zip([3, 4, 5], ranks):
        assert K <= n * n


def test_mean_field_shift_identity():
    """Verify that the explicit rearrangement of H (Eq 10) into
    h_tilde one-body + (1/2) sum_mu O_mu^2 holds as a second-quantized
    operator equality.
    """
    rng = np.random.default_rng(2)
    n = 3
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = _random_symmetric_psd_eri(n, rng)
    # Direct: H = sum h_pq a^dag_p a_q + (1/2) sum <pq|rs> a^dag_p a^dag_q a_s a_r
    H_direct = jw.one_body_matrix(h) + jw.two_body_matrix(eri)
    # Rank-one pool form
    pool = build_pool_from_integrals(h, eri, cholesky_threshold=1e-13)
    H_pool = pool.dense_matrix()
    assert np.allclose(H_direct, H_pool, atol=1e-8), np.abs(H_direct - H_pool).max()


def test_h_tilde_is_hermitian_when_h_is():
    rng = np.random.default_rng(3)
    n = 4
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = _random_symmetric_psd_eri(n, rng)
    h_t = mean_field_shifted_h(h, eri)
    assert np.allclose(h_t, h_t.conj().T, atol=1e-12)


def test_one_body_eigendecomposition_reconstructs_h_tilde():
    rng = np.random.default_rng(13)
    n = 4
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = _random_symmetric_psd_eri(n, rng)
    pool = build_pool_from_integrals(h, eri, cholesky_threshold=1e-12)
    channels = pool.one_body_eigendecomposition()
    h_rec = np.zeros_like(pool.h_tilde, dtype=complex)
    for ch in channels:
        assert np.isclose(np.linalg.norm(ch.phi), 1.0, atol=1e-12)
        h_rec += ch.coeff * np.outer(ch.phi, ch.phi.conj())
    assert np.allclose(h_rec, pool.h_tilde, atol=1e-10)


def test_build_pool_rejects_complex_general_case():
    rng = np.random.default_rng(14)
    n = 3
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)
    eri = _random_complex_psd_eri(n, rng)
    L = cholesky_eri(eri, threshold=1e-12)
    assert np.allclose(reconstruct_eri(L), eri, atol=1e-8)
    with pytest.raises(NotImplementedError, match="real-valued"):
        build_pool_from_integrals(h, eri, cholesky_threshold=1e-12)


def test_dense_matrix_reconstructs_manual_complex_pool_exactly():
    rng = np.random.default_rng(15)
    n = 3
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h = 0.5 * (h + h.conj().T)
    eri = _random_complex_psd_eri(n, rng)
    L = cholesky_eri(eri, threshold=1e-12)
    pool = HamiltonianPool(
        n_orbitals=n,
        h_tilde=mean_field_shifted_h(h, eri),
        cholesky_factors=L,
    )
    H_direct = jw.one_body_matrix(h) + jw.two_body_matrix(eri)
    assert np.allclose(pool.dense_matrix(), H_direct, atol=1e-8)
