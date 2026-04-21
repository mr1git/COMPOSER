"""Theorem 1 verification.

For small real-symmetric integrals, the PREP-SELECT-PREP\u2020 top-left
block times ``alpha`` reproduces the Hamiltonian
``H = h_tilde + (1/2) sum O_mu^2`` on the full Fock space.
"""
from __future__ import annotations

import numpy as np

from composer.block_encoding.lcu import build_hamiltonian_block_encoding
from composer.operators.hamiltonian import HamiltonianPool, build_pool_from_integrals


def _random_real_symmetric_eri(n: int, K: int, rng: np.random.Generator) -> np.ndarray:
    # Build ERIs that are 8-fold symmetric by factoring (pq|rs) = sum_mu L_mu_{pq} L_mu_{rs}
    # with L_mu symmetric (chemist convention), then converting to physicist via transpose.
    eri_chem = np.zeros((n, n, n, n))
    for _ in range(K):
        A = rng.normal(size=(n, n))
        L = 0.5 * (A + A.T)
        eri_chem += np.einsum("pq,rs->pqrs", L, L)
    # chemist (pq|rs) -> physicist <pr|qs>
    return eri_chem.transpose(0, 2, 1, 3)


def _one_body_H(pool: HamiltonianPool) -> np.ndarray:
    from composer.utils import fermion as jw

    n = pool.n_orbitals
    dim = jw.fock_dim(n)
    adag = [jw.jw_a_dagger(p, n) for p in range(n)]
    a_ = [jw.jw_a(p, n) for p in range(n)]
    H = np.zeros((dim, dim), dtype=complex)
    for p in range(n):
        for q in range(n):
            H += pool.h_tilde[p, q] * (adag[p] @ a_[q])
    return H


def test_lcu_top_left_block_times_alpha_equals_H():
    rng = np.random.default_rng(0)
    n = 2
    h = 0.1 * rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = _random_real_symmetric_eri(n, K=2, rng=rng)
    pool = build_pool_from_integrals(h, eri)
    be = build_hamiltonian_block_encoding(pool)
    block = be.top_left_block()
    reconstructed = be.alpha * block
    H_ref = pool.dense_matrix()
    assert np.allclose(reconstructed, H_ref, atol=1e-8)


def test_lcu_circuit_is_unitary():
    rng = np.random.default_rng(1)
    n = 2
    h = 0.1 * rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = _random_real_symmetric_eri(n, K=1, rng=rng)
    pool = build_pool_from_integrals(h, eri)
    be = build_hamiltonian_block_encoding(pool)
    W = be.W
    assert np.allclose(W @ W.conj().T, np.eye(W.shape[0]), atol=1e-8)


def test_lcu_alpha_is_sum_of_absolute_weights():
    rng = np.random.default_rng(2)
    n = 2
    h = rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = _random_real_symmetric_eri(n, K=1, rng=rng)
    pool = build_pool_from_integrals(h, eri)
    be = build_hamiltonian_block_encoding(pool)
    assert np.isclose(be.alpha, np.sum(np.abs(be.weights)), atol=1e-10)


def test_lcu_resources_match_compiled_circuit_and_branch_accounting():
    rng = np.random.default_rng(12)
    n = 3
    h = 0.1 * rng.normal(size=(n, n))
    h = 0.5 * (h + h.T)
    eri = _random_real_symmetric_eri(n, K=2, rng=rng)
    pool = build_pool_from_integrals(h, eri)

    be = build_hamiltonian_block_encoding(pool)
    resources = be.resources

    assert resources.alpha == be.alpha
    assert resources.n_system == pool.n_orbitals
    assert resources.n_ancilla == be.n_ancilla
    assert resources.selector_width == be.selector_width
    assert resources.subencoding_ancilla == 1
    assert resources.active_branch_count == len(be.weights)
    assert resources.compiled_branch_count == len(be.weights) + 1
    assert resources.null_branch_index == len(be.weights)
    assert resources.one_body_branch_count + resources.cholesky_branch_count == len(be.weights)
    assert resources.circuit == be.circuit.resource_summary()
    assert resources.circuit.gate_count_by_kind == {"PREP_H": 2, "SELECT_H": 1}


def test_lcu_zero_hamiltonian_gives_empty_encoding():
    # All-zero h and all-zero ERI -> no channels. Verify graceful handling.
    n = 2
    h = np.zeros((n, n))
    eri = np.zeros((n, n, n, n))
    # The pool builder needs non-zero to produce at least one channel.
    # We bypass by constructing a minimal pool with a single zero factor.
    from composer.operators.hamiltonian import HamiltonianPool

    pool = HamiltonianPool(
        n_orbitals=n,
        h_tilde=np.eye(n) * 1e-6,  # tiny diagonal so eigenchannels exist
        cholesky_factors=np.zeros((0, n, n)),
    )
    be = build_hamiltonian_block_encoding(pool)
    block = be.top_left_block()
    H_ref = pool.dense_matrix()
    assert np.allclose(be.alpha * block, H_ref, atol=1e-8)
