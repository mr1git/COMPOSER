"""Scalar QSP phase-sequence utilities (Wx convention, ASSUMPTION #6).

We use the *reflection* convention of Gilyen, Su, Low, Wiebe (2019) —
aka the Wx / W(x) convention:

    W(x) = [[x,        i sqrt(1 - x^2)],
            [i sqrt(1 - x^2), x       ]]    (signal operator, single qubit)

    S(phi) = [[exp(i phi), 0], [0, exp(-i phi)]]   (z-phase shift)

    U(x, Phi) = S(phi_0) * prod_{k=1..d} W(x) * S(phi_k),   Phi = (phi_0, ..., phi_d)

``<0|U|0>`` is a complex polynomial ``P(x)`` of degree ``d``; its
parity matches ``d mod 2``. For *symmetric* phase sequences (i.e.,
``phi_k = phi_{d-k}``) the polynomial ``P`` has the additional
symmetry ``P(x)^* = P(x)`` times a global phase, so ``Re(P(x))`` is the
degree-``d`` real polynomial the QSVT sequence implements.

This module provides scalar forward primitives

* ``qsp_signal(x)``                 - ``W(x)``
* ``qsp_phase_gate(phi)``           - ``S(phi)``
* ``qsp_unitary(phases, x)``        - assemble ``U(x, Phi)``
* ``qsp_polynomial(phases, x)``     - extract ``<0|U(x, Phi)|0>``
* ``qsp_block(phases, x)``          - extract the full 2x2 block (for inspection)

and scalar phase-compilation helpers

* ``solve_phases_real(target_coeffs, parity, initial=None, ...)``
* ``compile_real_chebyshev_phase_sequence(target_cheb_coeffs, parity, ...)``
* ``compile_exponential_qsp_schedule(alpha, eps, ...)``

The direct paper-facing target for generator exponentiation is the
single complex exponential ``exp(-i alpha x)``. The current repo still
resolves that target through parity-valid real branches because the
available scalar phase solver works on real parity-definite targets.
This module now makes that resolution explicit by compiling a direct
complex Chebyshev target first and then materializing the structured
fallback schedule consumed by ``block_encoding/generator_exp.py``.

Anchor the derivation in docstring of
``src/composer/qsp/qsvt_poly.py`` for the closed-form ``x -> x^2``
schedule used by Lemma 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .chebyshev import (
    chebyshev_to_monomial,
    cos_alpha_x_coefficients,
    evaluate_chebyshev,
    jacobi_anger_coefficients,
    recommended_degree,
    recommended_degree_with_parity,
    sin_alpha_x_coefficients,
    truncation_error_bound,
)

__all__ = [
    "CompiledExponentialQSPPhaseSchedule",
    "CompiledRealQSPPhaseSequence",
    "compile_exponential_qsp_schedule",
    "compile_real_chebyshev_phase_sequence",
    "qsp_signal",
    "qsp_phase_gate",
    "qsp_unitary",
    "qsp_polynomial",
    "qsp_block",
    "qsp_polynomial_on_grid",
    "solve_phases_real",
    "solve_phases_real_chebyshev",
]


@dataclass(frozen=True)
class CompiledRealQSPPhaseSequence:
    """Compiled scalar phase sequence for one real parity-valid target."""

    phases: np.ndarray
    parity: int
    degree: int
    target_basis: str
    target_coeffs: np.ndarray
    n_grid: int
    loss: float


@dataclass(frozen=True)
class CompiledExponentialQSPPhaseSchedule:
    """Compiled schedule for ``exp(-i alpha x)``.

    ``complex_chebyshev_coeffs`` retains the direct Appendix-C target,
    while ``cos_sequence`` and ``sin_sequence`` expose the structured
    parity-split realization used by the current repo.
    """

    alpha: float
    eps: float
    requested_strategy: str
    resolved_strategy: str
    complex_degree: int
    complex_chebyshev_coeffs: np.ndarray
    truncation_error_bound: float
    cos_sequence: CompiledRealQSPPhaseSequence
    sin_sequence: CompiledRealQSPPhaseSequence

    @property
    def uses_single_ladder(self) -> bool:
        return False


def qsp_signal(x: float) -> np.ndarray:
    """``W(x)`` in the Wx convention. ``x`` in ``[-1, 1]``."""
    if x < -1.0 - 1e-12 or x > 1.0 + 1e-12:
        raise ValueError(f"x={x} outside [-1, 1]")
    x = float(np.clip(x, -1.0, 1.0))
    s = np.sqrt(max(0.0, 1.0 - x * x))
    return np.array([[x, 1j * s], [1j * s, x]], dtype=complex)


def qsp_phase_gate(phi: float) -> np.ndarray:
    """``S(phi) = exp(i phi Z)``."""
    return np.array([[np.exp(1j * phi), 0.0], [0.0, np.exp(-1j * phi)]], dtype=complex)


def qsp_unitary(phases: np.ndarray, x: float) -> np.ndarray:
    """Assemble ``U(x, Phi) = S(phi_0) W(x) S(phi_1) W(x) ... S(phi_d)``.

    ``len(phases) = d + 1``.
    """
    phases = np.asarray(phases, dtype=float).ravel()
    d = len(phases) - 1
    if d < 0:
        raise ValueError("phases must be non-empty")
    W = qsp_signal(x)
    U = qsp_phase_gate(phases[0])
    for k in range(1, d + 1):
        U = U @ W @ qsp_phase_gate(phases[k])
    return U


def qsp_block(phases: np.ndarray, x: float) -> np.ndarray:
    """Return the full 2x2 QSP unitary at ``x``."""
    return qsp_unitary(phases, x)


def qsp_polynomial(phases: np.ndarray, x: float) -> complex:
    """Return ``<0|U(x, Phi)|0>`` = the top-left entry of the QSP unitary."""
    return complex(qsp_unitary(phases, x)[0, 0])


def qsp_polynomial_on_grid(phases: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Vectorized evaluation of ``P(x) = <0|U(x, Phi)|0>`` on an array ``xs``."""
    return np.array([qsp_polynomial(phases, float(x)) for x in xs], dtype=complex)


def _symmetrize(phases: np.ndarray) -> np.ndarray:
    """Symmetrize ``Phi`` in place: ``phi_k <- (phi_k + phi_{d-k}) / 2``."""
    phases = phases.copy()
    d = len(phases) - 1
    for k in range((d + 1) // 2):
        avg = 0.5 * (phases[k] + phases[d - k])
        phases[k] = avg
        phases[d - k] = avg
    return phases


def _validate_real_target_polynomial(
    coeffs: np.ndarray,
    parity: int,
    xs: np.ndarray,
    *,
    atol: float = 1e-10,
) -> None:
    """Validate the parity and boundedness conditions required by scalar QSP."""
    wrong_parity = coeffs[1 - parity :: 2]
    if wrong_parity.size and not np.allclose(wrong_parity, 0.0, atol=atol):
        raise ValueError("target_coeffs contains terms with the wrong parity for the requested QSP")
    values = np.polyval(coeffs[::-1], xs)
    if np.max(np.abs(values)) > 1.0 + 1e-8:
        raise ValueError("target polynomial must satisfy |P(x)| <= 1 on [-1, 1]")


def _solve_real_target_on_grid(
    target: np.ndarray,
    *,
    degree: int,
    initial: np.ndarray | None,
    xs: np.ndarray,
    tol: float,
    max_iter: int,
    enforce_symmetric: bool,
    rng_seed: int,
) -> tuple[np.ndarray, float]:
    """Solve phases for real target values already sampled on ``xs``."""
    d = degree

    def forward(phases_flat: np.ndarray) -> np.ndarray:
        phi = phases_flat
        if enforce_symmetric:
            phi = _symmetrize(phi)
        return qsp_polynomial_on_grid(phi, xs).real

    def loss(phases_flat: np.ndarray) -> float:
        diff = forward(phases_flat) - target
        return float(np.sum(diff * diff))

    rng = np.random.default_rng(rng_seed)
    if initial is None:
        initial = 0.1 * rng.standard_normal(d + 1)
    result = minimize(loss, initial, method="L-BFGS-B", tol=tol, options={"maxiter": max_iter})
    if not result.success and result.fun > max(tol, 1e-12):
        raise RuntimeError(f"phase optimization failed: {result.message}")
    phases = result.x
    if enforce_symmetric:
        phases = _symmetrize(phases)
    return phases, float(result.fun)


def solve_phases_real(
    target_coeffs: np.ndarray,
    parity: int,
    initial: np.ndarray | None = None,
    n_grid: int = 201,
    tol: float = 1e-10,
    max_iter: int = 2000,
    enforce_symmetric: bool = True,
    rng_seed: int = 0,
) -> tuple[np.ndarray, float]:
    """Find phases ``Phi`` such that ``Re(<0|U(x, Phi)|0>) = P_target(x)``."""
    coeffs = np.asarray(target_coeffs, dtype=float).ravel()
    d = len(coeffs) - 1
    if parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    if d % 2 != parity:
        raise ValueError(f"degree d={d} does not match parity={parity}")
    xs = np.cos(np.linspace(0, np.pi, n_grid))
    _validate_real_target_polynomial(coeffs, parity, xs)
    target = np.polyval(coeffs[::-1], xs)
    return _solve_real_target_on_grid(
        target,
        degree=d,
        initial=initial,
        xs=xs,
        tol=tol,
        max_iter=max_iter,
        enforce_symmetric=enforce_symmetric,
        rng_seed=rng_seed,
    )


def compile_real_chebyshev_phase_sequence(
    target_cheb_coeffs: np.ndarray,
    parity: int,
    initial: np.ndarray | None = None,
    n_grid: int = 201,
    tol: float = 1e-10,
    max_iter: int = 2000,
    enforce_symmetric: bool = True,
    rng_seed: int = 0,
) -> CompiledRealQSPPhaseSequence:
    """Compile one real parity-valid phase sequence from Chebyshev data."""
    coeffs = np.asarray(target_cheb_coeffs, dtype=float).ravel()
    d = len(coeffs) - 1
    if parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    if d % 2 != parity:
        raise ValueError(f"degree d={d} does not match parity={parity}")
    xs = np.cos(np.linspace(0, np.pi, n_grid))
    _validate_real_target_polynomial(chebyshev_to_monomial(coeffs), parity, xs)
    target = evaluate_chebyshev(coeffs, xs).real
    phases, loss = _solve_real_target_on_grid(
        target,
        degree=d,
        initial=initial,
        xs=xs,
        tol=tol,
        max_iter=max_iter,
        enforce_symmetric=enforce_symmetric,
        rng_seed=rng_seed,
    )
    return CompiledRealQSPPhaseSequence(
        phases=phases,
        parity=parity,
        degree=d,
        target_basis="chebyshev",
        target_coeffs=coeffs,
        n_grid=n_grid,
        loss=loss,
    )


def compile_exponential_qsp_schedule(
    alpha: float,
    eps: float,
    *,
    strategy: str = "auto",
    n_grid: int | None = None,
    tol: float = 1e-10,
    max_iter: int = 2000,
    enforce_symmetric: bool = True,
    rng_seed: int = 0,
) -> CompiledExponentialQSPPhaseSchedule:
    """Compile the scalar phase schedule for ``exp(-i alpha x)``."""
    if strategy not in {"auto", "parity_split"}:
        raise ValueError("strategy must be 'auto' or 'parity_split'")
    complex_degree = recommended_degree(alpha, eps)
    complex_coeffs = jacobi_anger_coefficients(alpha, complex_degree)
    cos_degree = recommended_degree_with_parity(alpha, eps / 2.0, parity=0)
    sin_degree = recommended_degree_with_parity(alpha, eps / 2.0, parity=1)
    cos_grid = n_grid or max(81, 8 * cos_degree + 1)
    sin_grid = n_grid or max(81, 8 * sin_degree + 1)
    cos_sequence = compile_real_chebyshev_phase_sequence(
        cos_alpha_x_coefficients(alpha, cos_degree),
        parity=0,
        n_grid=cos_grid,
        tol=tol,
        max_iter=max_iter,
        enforce_symmetric=enforce_symmetric,
        rng_seed=rng_seed,
    )
    sin_sequence = compile_real_chebyshev_phase_sequence(
        sin_alpha_x_coefficients(alpha, sin_degree),
        parity=1,
        n_grid=sin_grid,
        tol=tol,
        max_iter=max_iter,
        enforce_symmetric=enforce_symmetric,
        rng_seed=rng_seed,
    )
    return CompiledExponentialQSPPhaseSchedule(
        alpha=alpha,
        eps=eps,
        requested_strategy=strategy,
        resolved_strategy="parity_split_chebyshev",
        complex_degree=complex_degree,
        complex_chebyshev_coeffs=complex_coeffs,
        truncation_error_bound=truncation_error_bound(alpha, complex_degree),
        cos_sequence=cos_sequence,
        sin_sequence=sin_sequence,
    )


def solve_phases_real_chebyshev(
    target_cheb_coeffs: np.ndarray,
    parity: int,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Solve scalar-QSP phases for a real target given in Chebyshev basis."""
    compiled = compile_real_chebyshev_phase_sequence(target_cheb_coeffs, parity=parity, **kwargs)
    return compiled.phases, compiled.loss
