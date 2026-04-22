"""Verify the explicit Appendix B.2 Lemma-2 construction on small channels."""
from __future__ import annotations

import numpy as np

from composer.block_encoding.cholesky_channel import (
    apply_x_squared_qsvt,
    build_hermitian_one_body_block_encoding,
    cholesky_channel_block_encoding,
    hermitian_one_body_block_encoding,
    x_squared_qsvt_unitary,
)
from composer.circuits.simulator import unitary as circuit_unitary
from composer.circuits.gate import CircuitCall, MultiplexedGate, SelectGate, StatePreparationGate
from composer.utils import fermion as jw


def _one_body_dense(L: np.ndarray, n: int) -> np.ndarray:
    n_orb = L.shape[0]
    adag = [jw.jw_a_dagger(p, n) for p in range(n_orb)]
    a_ = [jw.jw_a(p, n) for p in range(n_orb)]
    dim = jw.fock_dim(n)
    O = np.zeros((dim, dim), dtype=complex)
    for p in range(n_orb):
        for q in range(n_orb):
            coef = L[p, q]
            if abs(coef) < 1e-15:
                continue
            O += coef * (adag[p] @ a_[q])
    return 0.5 * (O + O.conj().T)


def _random_hermitian(n: int, rng: np.random.Generator) -> np.ndarray:
    M = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    return 0.5 * (M + M.conj().T)


def test_rotated_mode_branches_encode_occupation_of_the_retained_modes():
    L = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    be = cholesky_channel_block_encoding(L)

    assert be.n_index == 1
    assert be.n_flag == 1
    assert be.n_signal == 1
    assert len(be.branch_unitaries) == 2

    for idx, orbital in enumerate(be.orbitals):
        expected = jw.jw_mode_number(orbital, 2)
        actual = be.branch_top_left_block(idx)
        assert np.allclose(actual, expected, atol=1e-10)

        signed = be.branch_top_left_block(idx, include_phase=True)
        phase = np.exp(1j * np.angle(be.eigenvalues[idx]))
        assert np.allclose(signed, phase * expected, atol=1e-10)


def test_prep_select_prep_block_matches_O_over_gamma():
    rng = np.random.default_rng(1)
    L = _random_hermitian(3, rng)
    be = cholesky_channel_block_encoding(L)
    O = _one_body_dense(L, 3)

    with np.errstate(all="ignore"):
        gram = be.one_body_unitary @ be.one_body_unitary.conj().T
    assert np.allclose(gram, np.eye(be.one_body_unitary.shape[0]), atol=1e-9)
    assert np.allclose(be.one_body_top_left_block() * be.alpha, O, atol=1e-9)
    assert np.allclose(be.one_body_unitary, build_hermitian_one_body_block_encoding(L).unitary, atol=1e-10)


def test_structural_one_body_and_degree_two_circuits_match_stored_matrices():
    rng = np.random.default_rng(11)
    L = _random_hermitian(3, rng)
    be = cholesky_channel_block_encoding(L)

    assert np.allclose(circuit_unitary(be.one_body_circuit), be.one_body_unitary, atol=1e-10)
    assert np.allclose(circuit_unitary(be.circuit), be.unitary, atol=1e-10)
    assert any(isinstance(op, CircuitCall) for op in be.one_body_circuit.gates)
    assert any(isinstance(op, StatePreparationGate) for op in be.one_body_circuit.gates)
    assert any(isinstance(op, CircuitCall) and op.kind == "SELECT_O" for op in be.one_body_circuit.gates)
    assert any(isinstance(op, MultiplexedGate) for op in be.select_circuit.gates)
    assert any(isinstance(op, SelectGate) for op in be.circuit.gates)


def test_degree_two_transform_returns_full_signal_index_flag_unitary():
    rng = np.random.default_rng(2)
    L = _random_hermitian(3, rng)
    be = cholesky_channel_block_encoding(L)
    O = _one_body_dense(L, 3)
    expected = (O @ O) / (be.alpha * be.alpha)

    with np.errstate(all="ignore"):
        gram = be.unitary @ be.unitary.conj().T
    assert np.allclose(gram, np.eye(be.unitary.shape[0]), atol=1e-9)
    assert np.allclose(be.top_left_block(), expected, atol=1e-9)


def test_diagonal_channel_uses_binary_index_width_for_retained_rank():
    L = np.diag([2.0, -0.5, 0.0])
    be = cholesky_channel_block_encoding(L)

    assert np.isclose(be.alpha, 2.5, atol=1e-12)
    assert be.n_index == 1
    assert len(be.branch_unitaries) == 2

    for idx, orbital in enumerate(be.orbitals):
        assert np.allclose(
            be.branch_top_left_block(idx),
            jw.jw_mode_number(orbital, 3),
            atol=1e-10,
        )


def test_compatibility_wrappers_still_match_exact_small_system_identity():
    rng = np.random.default_rng(3)
    L = _random_hermitian(3, rng)
    W, alpha = hermitian_one_body_block_encoding(L)
    U2 = x_squared_qsvt_unitary(W)
    O = _one_body_dense(L, 3)
    expected = (O @ O) / (alpha * alpha)
    dim = W.shape[0] // 2

    assert np.allclose(apply_x_squared_qsvt(W), expected, atol=1e-9)
    assert np.allclose(U2[:dim, :dim], expected, atol=1e-9)
