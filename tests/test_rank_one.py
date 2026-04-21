"""Tests for the rank-one operator primitives (Def 1/2/3).

We verify that each dataclass' dense_matrix reproduces the
second-quantized operator built directly from creation/annihilation
operators and user-supplied coefficients.
"""
from __future__ import annotations

import numpy as np
import pytest

from composer.operators.rank_one import (
    BilinearRankOne,
    PairRankOne,
    ProjectedQuadraticRankOne,
)
from composer.utils import fermion as jw


def _random_unit_vector(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    return v / np.linalg.norm(v)


def _random_antisymmetric(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    a = a - a.T
    # Normalize on strict upper triangle
    triu_vals = a[np.triu_indices(n, k=1)]
    norm = np.linalg.norm(triu_vals)
    return a / norm


# ------------------------------------------------------------------- Def 1


def test_bilinear_rank_one_matches_direct():
    rng = np.random.default_rng(0)
    n = 3
    u = _random_unit_vector(n, rng)
    v = _random_unit_vector(n, rng)
    lam = 0.37
    op = BilinearRankOne(u=u, v=v, coeff=lam)
    M = op.dense_matrix()
    adag = [jw.jw_a_dagger(p, n) for p in range(n)]
    a = [jw.jw_a(p, n) for p in range(n)]
    expected = np.zeros_like(M)
    for p in range(n):
        for q in range(n):
            expected += lam * u[p] * v[q].conj() * (adag[p] @ a[q])
    assert np.allclose(M, expected, atol=1e-12)


def test_bilinear_is_not_in_general_hermitian():
    rng = np.random.default_rng(1)
    n = 3
    u = _random_unit_vector(n, rng)
    v = _random_unit_vector(n, rng)
    op = BilinearRankOne(u=u, v=v, coeff=0.5)
    M = op.dense_matrix()
    # For generic u != v, L is not Hermitian
    assert not np.allclose(M, M.conj().T)


def test_bilinear_from_outer_roundtrip():
    rng = np.random.default_rng(2)
    n = 4
    u = _random_unit_vector(n, rng)
    v = _random_unit_vector(n, rng)
    lam = 1.2
    h = lam * np.outer(u, v.conj())
    op = BilinearRankOne.from_outer(h)
    # Reconstruct h up to a joint phase (u, v can each pick up a phase
    # so long as the product is consistent).
    h_rebuilt = op.coeff * np.outer(op.u, op.v.conj())
    phase = np.vdot(h.ravel(), h_rebuilt.ravel())
    phase /= abs(phase)
    assert np.allclose(h_rebuilt * phase.conjugate(), h, atol=1e-10)


# ------------------------------------------------------------------- Def 2


def test_pair_rank_one_matches_direct():
    rng = np.random.default_rng(3)
    n = 4
    U = _random_antisymmetric(n, rng)
    V = _random_antisymmetric(n, rng)
    coeff = 0.21 + 0.13j
    op = PairRankOne(U=U, V=V, coeff=coeff)
    M = op.dense_matrix()
    adag = [jw.jw_a_dagger(p, n) for p in range(n)]
    a = [jw.jw_a(p, n) for p in range(n)]
    dim = 2**n
    Bdag = np.zeros((dim, dim), dtype=complex)
    B = np.zeros((dim, dim), dtype=complex)
    for p in range(n):
        for q in range(p + 1, n):
            Bdag += U[p, q] * (adag[p] @ adag[q])
            B += V[p, q].conj() * (a[q] @ a[p])
    expected = coeff * (Bdag @ B)
    assert np.allclose(M, expected, atol=1e-12)


def test_pair_rank_one_supports_distinct_creation_and_annihilation_subspaces():
    n = 4
    U = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    V = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    op = PairRankOne(
        U=U,
        V=V,
        coeff=0.75,
        creation_orbitals=(2, 3),
        annihilation_orbitals=(0, 1),
    )
    M = op.dense_matrix(n)
    adag = [jw.jw_a_dagger(p, n) for p in range(n)]
    a = [jw.jw_a(p, n) for p in range(n)]
    expected = 0.75 * (adag[2] @ adag[3] @ a[1] @ a[0])
    assert np.allclose(M, expected, atol=1e-12)


def test_pair_rank_one_coefficient_tensor_uses_operator_conjugation():
    U = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    V = np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=complex)
    op = PairRankOne(U=U, V=V, coeff=0.5 - 0.25j)
    expected = op.coeff * np.einsum("ab,ij->abij", U, V.conj())
    assert np.allclose(op.coefficient_tensor(), expected, atol=1e-12)


def test_pair_rank_one_adjoint_swaps_subspaces_and_matches_dense_adjoint():
    U = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    V = np.array([[0.0, 1.0j], [-1.0j, 0.0]], dtype=complex)
    op = PairRankOne(
        U=U,
        V=V,
        coeff=0.2 + 0.3j,
        creation_orbitals=(2, 3),
        annihilation_orbitals=(0, 1),
    )
    adj = op.adjoint()
    assert adj.creation_orbitals == (0, 1)
    assert adj.annihilation_orbitals == (2, 3)
    assert np.allclose(adj.dense_matrix(4), op.dense_matrix(4).conj().T, atol=1e-12)


def test_pair_rank_one_number_conserving():
    # B^dag[U] B[V] preserves particle number (adds 2, removes 2).
    rng = np.random.default_rng(4)
    n = 4
    U = _random_antisymmetric(n, rng)
    V = _random_antisymmetric(n, rng)
    op = PairRankOne(U=U, V=V, coeff=1.0)
    M = op.dense_matrix()
    # Number operator sum_p n_p
    N = sum(jw.jw_number(p, n) for p in range(n))
    # [N, M] should vanish
    assert np.allclose(N @ M - M @ N, 0, atol=1e-10)


def test_pair_rank_one_dense_sigma_term_matches_l_minus_l_dag():
    rng = np.random.default_rng(44)
    n = 4
    U = _random_antisymmetric(2, rng)
    V = _random_antisymmetric(2, rng)
    op = PairRankOne(
        U=U,
        V=V,
        coeff=0.4 - 0.2j,
        creation_orbitals=(2, 3),
        annihilation_orbitals=(0, 1),
    )
    L = op.dense_matrix(n)
    assert np.allclose(op.dense_sigma_term(n), L - L.conj().T, atol=1e-12)


# ------------------------------------------------------------------- Def 3


def test_projected_quadratic_is_psd():
    rng = np.random.default_rng(5)
    n = 3
    orbitals = np.vstack([_random_unit_vector(n, rng) for _ in range(3)])
    weights = rng.normal(size=3) + 1j * rng.normal(size=3)
    op = ProjectedQuadraticRankOne(orbitals=orbitals, weights=weights, coeff=0.8)
    M = op.dense_matrix()
    # M is Hermitian
    assert np.allclose(M, M.conj().T, atol=1e-12)
    # Spectrum non-negative
    w = np.linalg.eigvalsh(0.5 * (M + M.conj().T))
    assert np.all(w > -1e-10)


def test_projected_quadratic_is_number_conserving():
    rng = np.random.default_rng(6)
    n = 4
    orbitals = np.vstack([_random_unit_vector(n, rng) for _ in range(2)])
    weights = rng.normal(size=2) + 1j * rng.normal(size=2)
    op = ProjectedQuadraticRankOne(orbitals=orbitals, weights=weights)
    M = op.dense_matrix()
    N = sum(jw.jw_number(p, n) for p in range(n))
    assert np.allclose(N @ M - M @ N, 0, atol=1e-12)


def test_projected_quadratic_matches_direct():
    rng = np.random.default_rng(9)
    n = 3
    orbitals = np.vstack([_random_unit_vector(n, rng) for _ in range(3)])
    weights = rng.normal(size=3) + 1j * rng.normal(size=3)
    gamma = 1.7 - 0.2j
    op = ProjectedQuadraticRankOne(orbitals=orbitals, weights=weights, coeff=gamma)
    M = op.dense_matrix()
    dim = 2**n
    O = np.zeros((dim, dim), dtype=complex)
    for c_r, u_r in zip(weights, orbitals, strict=True):
        O += c_r * jw.jw_mode_number(u_r)
    expected = gamma * (O @ O.conj().T)
    assert np.allclose(M, expected, atol=1e-12)


def test_projected_quadratic_single_mode_reduces_to_number_operator():
    n = 4
    orbital = np.array([0.0, 1.0, 0.0, 0.0], dtype=complex)
    op = ProjectedQuadraticRankOne(orbitals=orbital, weights=np.array([1.0]), coeff=2.3)
    assert np.allclose(op.dense_matrix(), 2.3 * jw.jw_number(1, n), atol=1e-12)


# ------------------------------------------------------------------- validation

def test_bilinear_rejects_nonunit_vectors():
    with pytest.raises(ValueError):
        BilinearRankOne(u=np.array([1.0, 1.0]), v=np.array([1.0, 0.0]), coeff=1.0)


def test_pair_rejects_nonantisymmetric():
    U = np.array([[1.0, 2.0], [2.0, 1.0]])  # symmetric
    V = np.array([[0.0, 1.0], [-1.0, 0.0]])
    # V must also be normalized; but U should fail first
    with pytest.raises(ValueError):
        PairRankOne(U=U, V=V, coeff=1.0)


def test_pair_rejects_ambiguous_embedding_for_mismatched_dimensions():
    U = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    V = np.zeros((3, 3), dtype=complex)
    V[0, 1] = 1 / np.sqrt(2)
    V[1, 0] = -1 / np.sqrt(2)
    V[0, 2] = 1 / np.sqrt(2)
    V[2, 0] = -1 / np.sqrt(2)
    with pytest.raises(ValueError):
        PairRankOne(U=U, V=V, coeff=1.0)
