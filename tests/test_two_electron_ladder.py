"""Verification of two-electron preparation and pair-Givens primitives."""
from __future__ import annotations

import numpy as np

from composer.circuits.simulator import statevector, unitary
from composer.ladders.phased_pair_givens import phased_pair_givens_matrix
from composer.ladders.two_electron import (
    build_ladder,
    build_number_conserving_ladder,
    build_rank2_ladder,
    orbital_rotation_unitary,
    rank2_decomposition,
    solve_angles,
)
from composer.utils import fermion as jw


def _rank2_antisymmetric(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    A = np.outer(u, v) - np.outer(v, u)
    A /= np.linalg.norm(A[np.triu_indices(A.shape[0], k=1)])
    return A


def _random_antisymmetric_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = A - A.T
    A /= np.linalg.norm(A[np.triu_indices(n, k=1)])
    return A


def _pair_target_statevector(u_asym: np.ndarray, n_qubits: int) -> np.ndarray:
    psi = np.zeros(2**n_qubits, dtype=complex)
    n = u_asym.shape[0]
    for p in range(n):
        for q in range(p + 1, n):
            psi[(1 << p) | (1 << q)] = u_asym[p, q]
    return psi


def _assert_state_equal_up_to_global_phase(psi, target, atol=1e-10):
    inner = np.vdot(target, psi)
    if abs(inner) < 1e-15:
        assert np.allclose(psi, 0, atol=atol)
        return
    phase = inner / abs(inner)
    assert np.allclose(psi * phase.conjugate(), target, atol=atol), np.abs(
        psi * phase.conjugate() - target
    ).max()


def test_orbital_rotation_acts_on_singles():
    rng = np.random.default_rng(0)
    n = 3
    A = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    A = 0.5 * (A - A.conj().T)
    from scipy.linalg import expm

    V = expm(A)
    UV = orbital_rotation_unitary(V)
    for p in range(n):
        adag_p = jw.jw_a_dagger(p, n)
        lhs = UV @ adag_p @ UV.conj().T
        rhs = sum(V[r, p] * jw.jw_a_dagger(r, n) for r in range(n))
        assert np.allclose(lhs, rhs, atol=1e-8), p


def test_rank2_decomposition_identifies_orthonormal_uv():
    rng = np.random.default_rng(1)
    n = 4
    u = rng.normal(size=n) + 1j * rng.normal(size=n)
    u /= np.linalg.norm(u)
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    v = v - np.vdot(u, v) * u
    v /= np.linalg.norm(v)
    A = _rank2_antisymmetric(u, v)
    uu, vv = rank2_decomposition(A)
    rebuilt = np.outer(uu, vv) - np.outer(vv, uu)
    rebuilt /= np.linalg.norm(rebuilt[np.triu_indices(n, k=1)])
    inner = np.vdot(A.ravel(), rebuilt.ravel())
    phase = inner / max(abs(inner), 1e-16)
    assert np.allclose(rebuilt * np.conj(phase), A, atol=1e-8)


def test_phased_pair_givens_rotates_disjoint_pair_basis_states():
    n = 4
    theta = 0.37
    phi = 0.5
    p, q, r, s = 0, 1, 2, 3
    G = phased_pair_givens_matrix(theta, phi, p, q, r, s, n)
    idx_rs = (1 << r) | (1 << s)
    idx_pq = (1 << p) | (1 << q)

    e_rs = np.zeros(16, dtype=complex)
    e_rs[idx_rs] = 1.0
    e_pq = np.zeros(16, dtype=complex)
    e_pq[idx_pq] = 1.0

    out_rs = G @ e_rs
    out_pq = G @ e_pq

    expected_rs = np.zeros(16, dtype=complex)
    expected_rs[idx_rs] = np.cos(theta)
    expected_rs[idx_pq] = np.exp(1j * phi) * np.sin(theta)
    expected_pq = np.zeros(16, dtype=complex)
    expected_pq[idx_rs] = -np.exp(-1j * phi) * np.sin(theta)
    expected_pq[idx_pq] = np.cos(theta)

    assert np.allclose(out_rs, expected_rs, atol=1e-10)
    assert np.allclose(out_pq, expected_pq, atol=1e-10)


def test_phased_pair_givens_supports_overlapping_pairs():
    n = 4
    theta = -0.41
    phi = 0.23
    p, q, r, s = 0, 2, 0, 1
    G = phased_pair_givens_matrix(theta, phi, p, q, r, s, n)
    idx_rs = (1 << r) | (1 << s)
    idx_pq = (1 << p) | (1 << q)

    e_rs = np.zeros(16, dtype=complex)
    e_rs[idx_rs] = 1.0
    e_pq = np.zeros(16, dtype=complex)
    e_pq[idx_pq] = 1.0

    out_rs = G @ e_rs
    out_pq = G @ e_pq

    expected_rs = np.zeros(16, dtype=complex)
    expected_rs[idx_rs] = np.cos(theta)
    expected_rs[idx_pq] = np.exp(1j * phi) * np.sin(theta)
    expected_pq = np.zeros(16, dtype=complex)
    expected_pq[idx_rs] = -np.exp(-1j * phi) * np.sin(theta)
    expected_pq[idx_pq] = np.cos(theta)

    assert np.allclose(out_rs, expected_rs, atol=1e-10)
    assert np.allclose(out_pq, expected_pq, atol=1e-10)


def test_phased_pair_givens_leaves_orthogonal_pair_states_invariant():
    n = 4
    theta = 0.37
    phi = 0.5
    G = phased_pair_givens_matrix(theta, phi, 0, 2, 0, 1, n)
    affected = {(1 << 0) | (1 << 1), (1 << 0) | (1 << 2)}
    for idx in ((1 << 0) | (1 << 3), (1 << 1) | (1 << 2), (1 << 2) | (1 << 3)):
        basis = np.zeros(16, dtype=complex)
        basis[idx] = 1.0
        out = G @ basis
        assert idx not in affected
        assert np.isclose(out[idx], 1.0, atol=1e-10)
        out[idx] = 0.0
        assert np.allclose(out, 0.0, atol=1e-10)


def test_pair_givens_rejects_invalid_pair_labels():
    with np.testing.assert_raises(ValueError):
        phased_pair_givens_matrix(0.1, 0.2, 1, 0, 0, 2, 4)
    with np.testing.assert_raises(ValueError):
        phased_pair_givens_matrix(0.1, 0.2, 0, 0, 0, 2, 4)
    with np.testing.assert_raises(ValueError):
        phased_pair_givens_matrix(0.1, 0.2, 0, 1, 0, 1, 4)


def test_ladder_prepares_general_complex_target():
    rng = np.random.default_rng(2)
    n = 5
    for _ in range(5):
        A = _random_antisymmetric_unit(n, rng)
        psi = statevector(build_ladder(A))
        _assert_state_equal_up_to_global_phase(psi, _pair_target_statevector(A, n))


def test_number_conserving_ladder_maps_the_reference_pair_to_target():
    rng = np.random.default_rng(3)
    n = 5
    for _ in range(5):
        A = _random_antisymmetric_unit(n, rng)
        angles = solve_angles(A)
        init = np.zeros(2**n, dtype=complex)
        init[(1 << angles.pivot[0]) | (1 << angles.pivot[1])] = 1.0
        psi = statevector(build_number_conserving_ladder(A), init=init)
        _assert_state_equal_up_to_global_phase(psi, _pair_target_statevector(A, n))


def test_number_conserving_two_electron_ladder_preserves_particle_number():
    rng = np.random.default_rng(4)
    A = _random_antisymmetric_unit(4, rng)
    U = unitary(build_number_conserving_ladder(A))
    basis = np.zeros(16, dtype=complex)
    basis[(1 << 0) | (1 << 1) | (1 << 2)] = 1.0
    out = U @ basis
    for idx, amp in enumerate(out):
        if abs(amp) < 1e-12:
            continue
        assert idx.bit_count() == 3


def test_pivot_pair_defaults_to_argmax_amplitude():
    A = np.zeros((4, 4), dtype=complex)
    A[0, 3] = 0.2
    A[3, 0] = -0.2
    A[1, 2] = 0.9j
    A[2, 1] = -0.9j
    A /= np.linalg.norm(A[np.triu_indices(4, k=1)])
    angles = solve_angles(A)
    assert angles.pivot == (1, 2)


def test_explicit_pivot_pair_is_honored_in_both_forms():
    rng = np.random.default_rng(5)
    A = _random_antisymmetric_unit(4, rng)
    angles = solve_angles(A, pivot=(0, 3))
    assert angles.pivot == (0, 3)

    prep = statevector(build_ladder(A, pivot=(0, 3)))
    _assert_state_equal_up_to_global_phase(prep, _pair_target_statevector(A, 4))

    init = np.zeros(16, dtype=complex)
    init[(1 << 0) | (1 << 3)] = 1.0
    rotated = statevector(build_number_conserving_ladder(A, pivot=(0, 3)), init=init)
    _assert_state_equal_up_to_global_phase(rotated, _pair_target_statevector(A, 4))


def test_rank2_compatibility_alias_uses_the_full_ladder():
    rng = np.random.default_rng(6)
    n = 4
    u = rng.normal(size=n) + 1j * rng.normal(size=n)
    u /= np.linalg.norm(u)
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    v = v - np.vdot(u, v) * u
    v /= np.linalg.norm(v)
    A = _rank2_antisymmetric(u, v)
    psi = statevector(build_rank2_ladder(A))
    _assert_state_equal_up_to_global_phase(psi, _pair_target_statevector(A, n))
