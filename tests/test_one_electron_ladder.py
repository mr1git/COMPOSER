"""Verification of one-electron preparation and number-conserving ladders."""
from __future__ import annotations

import numpy as np

from composer.circuits.simulator import statevector, unitary
from composer.ladders.givens import givens_fermionic_matrix
from composer.ladders.one_electron import (
    build_ladder,
    build_number_conserving_ladder,
    solve_angles,
)


def _random_real_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=n)
    return v / np.linalg.norm(v)


def _random_complex_unit(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=n) + 1j * rng.normal(size=n)
    return v / np.linalg.norm(v)


def _target_statevector(u: np.ndarray, n_qubits: int) -> np.ndarray:
    psi = np.zeros(2**n_qubits, dtype=complex)
    for p in range(u.shape[0]):
        psi[1 << p] = u[p]
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


def test_givens_rotates_the_one_electron_subspace_with_paper_convention():
    theta = 0.37
    G = givens_fermionic_matrix(theta, 0, 2, 4)
    e0 = np.zeros(16, dtype=complex)
    e2 = np.zeros(16, dtype=complex)
    e0[1 << 0] = 1.0
    e2[1 << 2] = 1.0

    out0 = G @ e0
    out2 = G @ e2

    expected0 = np.zeros(16, dtype=complex)
    expected0[1 << 0] = np.cos(theta)
    expected0[1 << 2] = -np.sin(theta)
    expected2 = np.zeros(16, dtype=complex)
    expected2[1 << 0] = np.sin(theta)
    expected2[1 << 2] = np.cos(theta)

    assert np.allclose(out0, expected0, atol=1e-10)
    assert np.allclose(out2, expected2, atol=1e-10)


def test_ladder_prepares_real_target():
    rng = np.random.default_rng(0)
    n = 4
    for _ in range(5):
        u = _random_real_unit(n, rng)
        psi = statevector(build_ladder(u))
        _assert_state_equal_up_to_global_phase(psi, _target_statevector(u, n))


def test_ladder_prepares_complex_target():
    rng = np.random.default_rng(1)
    n = 5
    for _ in range(5):
        u = _random_complex_unit(n, rng)
        psi = statevector(build_ladder(u))
        _assert_state_equal_up_to_global_phase(psi, _target_statevector(u, n))


def test_number_conserving_ladder_maps_the_reference_orbital_to_target():
    rng = np.random.default_rng(2)
    n = 5
    for _ in range(5):
        u = _random_complex_unit(n, rng)
        angles = solve_angles(u)
        init = np.zeros(2**n, dtype=complex)
        init[1 << angles.pivot] = 1.0
        psi = statevector(build_number_conserving_ladder(u), init=init)
        _assert_state_equal_up_to_global_phase(psi, _target_statevector(u, n))


def test_number_conserving_ladder_preserves_particle_number_outside_the_target_sector():
    u = np.array([0.2 + 0.1j, -0.3, 0.4j, 0.8], dtype=complex)
    u /= np.linalg.norm(u)
    U = unitary(build_number_conserving_ladder(u))

    # Start in a three-electron basis state and verify support stays in
    # the three-electron sector.
    basis = np.zeros(16, dtype=complex)
    basis[(1 << 0) | (1 << 1) | (1 << 3)] = 1.0
    out = U @ basis
    for idx, amp in enumerate(out):
        if abs(amp) < 1e-12:
            continue
        assert bin(idx).count("1") == 3


def test_pivot_is_argmax_by_default():
    u = np.array([0.3, 0.9, 0.2, -0.25], dtype=complex)
    u /= np.linalg.norm(u)
    angles = solve_angles(u)
    assert angles.pivot == 1


def test_explicit_pivot_honored_in_both_forms():
    u = np.array([0.3, 0.9, 0.2, -0.25j], dtype=complex)
    u /= np.linalg.norm(u)
    angles = solve_angles(u, pivot=3)
    assert angles.pivot == 3

    prep = statevector(build_ladder(u, pivot=3))
    _assert_state_equal_up_to_global_phase(prep, _target_statevector(u, 4))

    init = np.zeros(16, dtype=complex)
    init[1 << 3] = 1.0
    rotated = statevector(build_number_conserving_ladder(u, pivot=3), init=init)
    _assert_state_equal_up_to_global_phase(rotated, _target_statevector(u, 4))


def test_basis_state_is_trivial():
    u = np.array([0, 0, 1, 0], dtype=complex)
    psi = statevector(build_ladder(u))
    assert np.allclose(psi, _target_statevector(u, 4))


def test_pivot_single_qubit_register():
    u = np.array([1.0], dtype=complex)
    psi = statevector(build_ladder(u))
    assert np.isclose(psi[1], 1.0)
