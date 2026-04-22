"""Tests for the scalar QSP utilities and Chebyshev plumbing.

* Chebyshev truncation of ``e^{-i alpha x}`` meets the theoretical
  error bound and converges for increasing degree.
* The Wx-convention QSP unitary is unitary; symmetric-phase sequences
  produce polynomials with known parity.
* The degree-2 ``x -> x^2`` schedule returns Re-part equal to ``x^2``
  exactly.
* ``solve_phases_real`` and ``solve_phases_real_chebyshev`` are scalar
  utilities for parity-valid targets only; the oracle-level use of
  those phase lists is tested separately in ``test_generator_exp.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from composer.qsp.chebyshev import (
    chebyshev_parity,
    cos_alpha_x_coefficients,
    evaluate_chebyshev,
    jacobi_anger_coefficients,
    split_chebyshev_by_parity,
    split_exponential_chebyshev_components,
    recommended_degree,
    sin_alpha_x_coefficients,
    truncation_error_bound,
)
from composer.qsp.phases import (
    compile_exponential_qsp_schedule,
    compile_real_chebyshev_phase_sequence,
    qsp_phase_gate,
    qsp_polynomial,
    qsp_signal,
    qsp_unitary,
    solve_phases_real,
    solve_phases_real_chebyshev,
)
from composer.qsp.qsvt_poly import x_squared_phases


def test_chebyshev_cos_approximation_dense_grid():
    alpha = 2.3
    eps = 1e-8
    d = recommended_degree(alpha, eps)
    coeffs = cos_alpha_x_coefficients(alpha, d)
    xs = np.linspace(-1, 1, 501)
    approx = evaluate_chebyshev(coeffs, xs)
    exact = np.cos(alpha * xs)
    assert np.max(np.abs(approx - exact)) < 10 * eps


def test_chebyshev_sin_approximation_dense_grid():
    alpha = 1.7
    eps = 1e-8
    d = recommended_degree(alpha, eps)
    coeffs = sin_alpha_x_coefficients(alpha, d)
    xs = np.linspace(-1, 1, 501)
    approx = evaluate_chebyshev(coeffs, xs)
    exact = np.sin(alpha * xs)
    assert np.max(np.abs(approx - exact)) < 10 * eps


def test_chebyshev_complex_expansion_matches():
    alpha = 1.3
    d = recommended_degree(alpha, 1e-10)
    coeffs = jacobi_anger_coefficients(alpha, d)
    xs = np.linspace(-1, 1, 301)
    approx = evaluate_chebyshev(coeffs, xs)
    exact = np.exp(-1j * alpha * xs)
    assert np.max(np.abs(approx - exact)) < 1e-9


def test_chebyshev_truncation_bound_is_conservative():
    alpha = 2.0
    # low-ish degree so the bound is informative
    d = 15
    bound = truncation_error_bound(alpha, d)
    coeffs = jacobi_anger_coefficients(alpha, d)
    xs = np.linspace(-1, 1, 501)
    err = np.max(np.abs(evaluate_chebyshev(coeffs, xs) - np.exp(-1j * alpha * xs)))
    assert err <= bound + 1e-14


def test_qsp_signal_unitary():
    for x in np.linspace(-1, 1, 11):
        W = qsp_signal(x)
        assert np.allclose(W @ W.conj().T, np.eye(2), atol=1e-12)


def test_qsp_phase_gate_unitary():
    for phi in [0.0, 0.3, -1.0, 2 * np.pi]:
        S = qsp_phase_gate(phi)
        assert np.allclose(S @ S.conj().T, np.eye(2), atol=1e-14)


def test_qsp_unitary_is_unitary_random_phases():
    rng = np.random.default_rng(7)
    for d in [2, 3, 5, 8]:
        phases = rng.uniform(-np.pi, np.pi, size=d + 1)
        for x in [-0.9, -0.3, 0.0, 0.4, 0.99]:
            U = qsp_unitary(phases, x)
            assert np.allclose(U @ U.conj().T, np.eye(2), atol=1e-10)


def test_wx_qsp_polynomial_has_definite_parity_for_arbitrary_phases():
    rng = np.random.default_rng(17)
    for d in [1, 2, 5, 8]:
        phases = rng.uniform(-np.pi, np.pi, size=d + 1)
        xs = np.linspace(-1, 1, 4 * (d + 1) + 1)
        ys = np.array([qsp_polynomial(phases, float(x)) for x in xs])
        coeffs = np.polynomial.polynomial.polyfit(xs, ys, d)
        wrong_parity = coeffs[1 - (d % 2) :: 2]
        assert np.max(np.abs(wrong_parity), initial=0.0) < 1e-8


def test_qsp_unitary_zero_phases_gives_powers_of_W():
    # phi_k = 0 everywhere => U = W(x)^d
    for d in [1, 2, 5]:
        phases = np.zeros(d + 1)
        for x in [-0.3, 0.5, 0.8]:
            U = qsp_unitary(phases, x)
            W = qsp_signal(x)
            Wd = np.eye(2, dtype=complex)
            for _ in range(d):
                Wd = Wd @ W
            assert np.allclose(U, Wd, atol=1e-12)


def test_x_squared_polynomial_real_part_is_exact():
    phases = x_squared_phases()
    xs = np.linspace(-1, 1, 51)
    for x in xs:
        P = qsp_polynomial(phases, float(x))
        assert np.isclose(P.real, x * x, atol=1e-12)


def test_x_squared_polynomial_full_value():
    # P(x) = x^2 - i (1 - x^2), derived in the docstring of qsvt_poly.py.
    phases = x_squared_phases()
    xs = np.linspace(-1, 1, 21)
    for x in xs:
        P = qsp_polynomial(phases, float(x))
        expected = x * x - 1j * (1 - x * x)
        assert np.allclose(P, expected, atol=1e-12)


def test_solve_phases_real_recovers_odd_target():
    # target: 0.5 * x (odd, degree 1, within |P| <= 1)
    target = np.array([0.0, 0.5], dtype=float)
    phases, loss = solve_phases_real(target, parity=1, rng_seed=3)
    xs = np.linspace(-1, 1, 101)
    approx = np.array([qsp_polynomial(phases, float(x)).real for x in xs])
    expected = 0.5 * xs
    assert np.max(np.abs(approx - expected)) < 1e-6, loss


def test_solve_phases_real_recovers_even_target():
    # target: 1 - 2 x^2 = T_2(x), achievable by d=2 phases (non-symmetric in general).
    target = np.array([1.0, 0.0, -2.0], dtype=float)
    phases, loss = solve_phases_real(target, parity=0, rng_seed=5, max_iter=5000)
    xs = np.linspace(-1, 1, 101)
    approx = np.array([qsp_polynomial(phases, float(x)).real for x in xs])
    expected = 1.0 - 2.0 * xs * xs
    # Slightly loose: numerical optimizer may settle at ~1e-5; target is well-posed.
    assert np.max(np.abs(approx - expected)) < 1e-4, loss


def test_solve_phases_real_rejects_wrong_parity_coefficients():
    target = np.array([0.0, 0.5, 0.1, 0.0], dtype=float)
    with pytest.raises(ValueError, match="wrong parity"):
        solve_phases_real(target, parity=1)


def test_solve_phases_real_rejects_polynomial_outside_unit_disk():
    target = np.array([1.2], dtype=float)
    with pytest.raises(ValueError, match="\\|P\\(x\\)\\| <= 1"):
        solve_phases_real(target, parity=0)


def test_solve_phases_real_chebyshev_recovers_even_cosine_target():
    alpha = 0.5
    degree = 8
    cheb_coeffs = cos_alpha_x_coefficients(alpha, degree)
    phases, loss = solve_phases_real_chebyshev(cheb_coeffs, parity=0, max_iter=500, n_grid=81)
    xs = np.linspace(-1, 1, 101)
    approx = np.array([qsp_polynomial(phases, float(x)).real for x in xs])
    expected = evaluate_chebyshev(cheb_coeffs, xs).real
    assert np.max(np.abs(approx - expected)) < 5e-5, loss


def test_compile_real_chebyshev_phase_sequence_keeps_chebyshev_target_metadata():
    alpha = 0.5
    degree = 8
    cheb_coeffs = cos_alpha_x_coefficients(alpha, degree)
    compiled = compile_real_chebyshev_phase_sequence(
        cheb_coeffs,
        parity=0,
        max_iter=500,
        n_grid=81,
    )

    assert compiled.target_basis == "chebyshev"
    assert compiled.degree == degree
    assert compiled.parity == 0
    assert np.allclose(compiled.target_coeffs, cheb_coeffs, atol=1e-12)

    xs = np.linspace(-1, 1, 101)
    approx = np.array([qsp_polynomial(compiled.phases, float(x)).real for x in xs])
    expected = evaluate_chebyshev(cheb_coeffs, xs).real
    assert np.max(np.abs(approx - expected)) < 5e-5, compiled.loss


def test_compile_exponential_qsp_schedule_tracks_direct_complex_target_and_structured_fallback():
    alpha = 0.5
    eps = 5e-2
    compiled = compile_exponential_qsp_schedule(
        alpha,
        eps,
        strategy="direct_complex",
        n_grid=81,
        max_iter=500,
    )

    assert compiled.requested_strategy == "direct_complex"
    assert compiled.resolved_strategy == "parity_split_due_mixed_parity"
    assert compiled.uses_single_ladder is False
    assert compiled.direct_complex_supported is False
    assert compiled.direct_sequence is None
    assert compiled.fallback_reason is not None
    assert "definite parity" in compiled.fallback_reason
    assert compiled.cos_sequence.target_basis == "chebyshev"
    assert compiled.sin_sequence.target_basis == "chebyshev"
    assert compiled.complex_degree >= 0
    assert compiled.cos_sequence.degree % 2 == 0
    assert compiled.sin_sequence.degree % 2 == 1
    assert chebyshev_parity(compiled.complex_chebyshev_coeffs) is None

    xs = np.linspace(-1, 1, 101)
    complex_target = evaluate_chebyshev(compiled.complex_chebyshev_coeffs, xs)
    exact = np.exp(-1j * alpha * xs)
    assert np.max(np.abs(complex_target - exact)) <= compiled.truncation_error_bound + 1e-12

    cos_approx = np.array([qsp_polynomial(compiled.cos_sequence.phases, float(x)).real for x in xs])
    sin_approx = np.array([qsp_polynomial(compiled.sin_sequence.phases, float(x)).real for x in xs])
    assert np.max(np.abs(cos_approx - np.cos(alpha * xs))) < 5e-5, compiled.cos_sequence.loss
    assert np.max(np.abs(sin_approx - np.sin(alpha * xs))) < 5e-5, compiled.sin_sequence.loss


def test_exponential_fallback_branches_are_derived_from_direct_complex_coefficients():
    alpha = 0.5
    eps = 5e-2
    compiled = compile_exponential_qsp_schedule(alpha, eps, strategy="direct_complex", n_grid=81, max_iter=500)

    even, odd = split_chebyshev_by_parity(compiled.complex_chebyshev_coeffs)
    cos_coeffs, sin_coeffs = split_exponential_chebyshev_components(compiled.complex_chebyshev_coeffs)

    assert np.allclose(compiled.complex_even_chebyshev_coeffs, even, atol=1e-12)
    assert np.allclose(compiled.complex_odd_chebyshev_coeffs, odd, atol=1e-12)
    assert np.allclose(compiled.cos_sequence.target_coeffs, cos_coeffs, atol=1e-12)
    assert np.allclose(compiled.sin_sequence.target_coeffs, sin_coeffs, atol=1e-12)
