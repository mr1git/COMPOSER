"""Verify Lemma 1 (bilinear block encoding) on the N=1 subspace.

Top-left ``2**n x 2**n`` block of the Lemma 1 circuit (ancilla=|0>),
restricted to the single-excitation subspace ``H_{N=1}``, must equal

    L / alpha  =  |u> <v|     (ASSUMPTION #4: alpha = |lambda|)

where the ``|e_p> = a_p^dag |vac>`` basis is used for ``H_{N=1}``.
"""
from __future__ import annotations

import numpy as np

from composer.block_encoding.bilinear import (
    build_bilinear_block_encoding,
    orbital_rotation_first_column,
)
from composer.operators.rank_one import BilinearRankOne
from composer.utils.fermion import single_excitation_basis_indices


def _random_complex_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    return v / np.linalg.norm(v)


def _restrict_to_single_excitation(M: np.ndarray, n: int) -> np.ndarray:
    """Return the n x n block of M in the |e_p> basis (p = 0..n-1)."""
    idx = single_excitation_basis_indices(n)
    return M[np.ix_(idx, idx)]


def _projector_on_qubit_zero_occupied(n: int) -> np.ndarray:
    dim = 2**n
    P = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        if idx & 1:
            P[idx, idx] = 1.0
    return P


def test_bilinear_be_top_left_block_matches_projector_sandwich_on_full_fock():
    rng = np.random.default_rng(7)
    n = 4
    u = _random_complex_unit(n, rng)
    v = _random_complex_unit(n, rng)
    be = build_bilinear_block_encoding(BilinearRankOne(u=u, v=v, coeff=1.0))
    block = be.top_left_block()

    U_u = orbital_rotation_first_column(u)
    U_v = orbital_rotation_first_column(v)
    expected = U_u @ _projector_on_qubit_zero_occupied(n) @ U_v.conj().T
    assert np.allclose(block, expected, atol=1e-9)


def test_bilinear_be_top_left_block_is_rank_one_dyad():
    rng = np.random.default_rng(0)
    n = 4
    for _ in range(5):
        u = _random_complex_unit(n, rng)
        v = _random_complex_unit(n, rng)
        lam = float(rng.uniform(0.1, 2.0))
        L = BilinearRankOne(u=u, v=v, coeff=lam)
        be = build_bilinear_block_encoding(L)
        block = be.top_left_block()
        # Restrict to N=1 subspace: n x n matrix in |e_p> basis.
        reduced = _restrict_to_single_excitation(block, n)
        # The reconstructed dyad equals |u><v| = outer(u, v.conj()).
        expected = np.outer(u, v.conj())
        # alpha = lam: block*alpha should equal lam * |u><v| = L's coefficient matrix.
        # But be.top_left_block is W[:dim, :dim] which equals L/alpha on H_{N=1}.
        assert np.allclose(reduced, expected, atol=1e-9)


def test_bilinear_be_top_left_times_alpha_equals_L():
    rng = np.random.default_rng(1)
    n = 4
    u = _random_complex_unit(n, rng)
    v = _random_complex_unit(n, rng)
    lam = 1.3
    L = BilinearRankOne(u=u, v=v, coeff=lam)
    be = build_bilinear_block_encoding(L)
    block = be.top_left_block()
    reduced = _restrict_to_single_excitation(block, n)
    L_dense = L.dense_matrix()
    L_reduced = _restrict_to_single_excitation(L_dense, n)
    assert np.allclose(be.alpha * reduced, L_reduced, atol=1e-9)


def test_bilinear_be_is_not_a_full_fock_block_encoding_of_one_body_operator():
    rng = np.random.default_rng(11)
    n = 4
    u = _random_complex_unit(n, rng)
    v = _random_complex_unit(n, rng)
    be = build_bilinear_block_encoding(BilinearRankOne(u=u, v=v, coeff=1.0))
    block = be.top_left_block()
    L_dense = BilinearRankOne(u=u, v=v, coeff=1.0).dense_matrix()
    assert not np.allclose(block, L_dense, atol=1e-6)


def test_bilinear_be_circuit_is_unitary():
    rng = np.random.default_rng(2)
    n = 3
    u = _random_complex_unit(n, rng)
    v = _random_complex_unit(n, rng)
    L = BilinearRankOne(u=u, v=v, coeff=0.77)
    be = build_bilinear_block_encoding(L)
    from composer.circuits.simulator import unitary as circuit_unitary

    W = circuit_unitary(be.circuit)
    dim = 2 ** (n + 1)
    assert W.shape == (dim, dim)
    assert np.allclose(W @ W.conj().T, np.eye(dim), atol=1e-9)


def test_bilinear_be_alpha_equals_coeff():
    u = np.array([1.0, 0, 0, 0], dtype=complex)
    v = np.array([0, 1.0, 0, 0], dtype=complex)
    L = BilinearRankOne(u=u, v=v, coeff=2.5)
    be = build_bilinear_block_encoding(L)
    assert np.isclose(be.alpha, 2.5)


def test_bilinear_be_one_ancilla():
    u = np.array([1.0, 0, 0, 0], dtype=complex)
    v = np.array([0, 1.0, 0, 0], dtype=complex)
    L = BilinearRankOne(u=u, v=v, coeff=1.0)
    be = build_bilinear_block_encoding(L)
    assert be.circuit.num_qubits == 5  # n + 1
    assert be.ancilla == 4
