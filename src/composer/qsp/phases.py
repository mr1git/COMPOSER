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

and a numerical phase finder

* ``solve_phases_real(target_coeffs, parity, initial=None, ...)``
* ``solve_phases_real_chebyshev(target_cheb_coeffs, parity, ...)``

that finds a length-``d+1`` phase sequence whose QSP unitary satisfies
``Re(<0|U|0>) = target_polynomial(x)`` on ``[-1, 1]`` via L-BFGS-B. This
is a scalar verification utility for small parity-valid targets. The
oracle-level QSP/QSVT construction lives separately in
``block_encoding/generator_exp.py``; it consumes these scalar phase
lists, but the scalar helpers themselves do not build oracle circuits.

Anchor the derivation in docstring of
``src/composer/qsp/qsvt_poly.py`` for the closed-form ``x -> x^2``
schedule used by Lemma 2.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from .chebyshev import chebyshev_to_monomial

__all__ = [
    "qsp_signal",
    "qsp_phase_gate",
    "qsp_unitary",
    "qsp_polynomial",
    "qsp_block",
    "qsp_polynomial_on_grid",
    "solve_phases_real",
    "solve_phases_real_chebyshev",
]


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
    """Find phases ``Phi`` such that ``Re(<0|U(x, Phi)|0>) = P_target(x)``.

    Parameters
    ----------
    target_coeffs : np.ndarray
        Monomial-basis real coefficients ``[a_0, a_1, ..., a_d]`` of the
        target polynomial ``P_target(x) = sum_k a_k x^k``. The polynomial
        must have definite parity ``d mod 2`` and satisfy
        ``|P_target(x)| <= 1`` on ``[-1, 1]``.
    parity : int
        0 for even, 1 for odd. Consistency-checked against the degree.
    initial : np.ndarray, optional
        Initial phase guess; defaults to small random Gaussian perturbation.
    n_grid : int
        Number of sample points on ``[-1, 1]`` for the L2 objective.
    tol, max_iter : float, int
        scipy.optimize.minimize parameters (``L-BFGS-B``).
    enforce_symmetric : bool
        If True, symmetrize the phase sequence every evaluation; this
        guides the optimizer toward the Wang-Dong-Lin symmetric branch,
        for which ``P`` is automatically real on ``[-1, 1]``.

    Returns
    -------
    phases : np.ndarray, length d + 1
    final_loss : float
        Final L2 fit residual squared on the grid (for diagnostic use).
    """
    coeffs = np.asarray(target_coeffs, dtype=float).ravel()
    d = len(coeffs) - 1
    if parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    if d % 2 != parity:
        raise ValueError(f"degree d={d} does not match parity={parity}")
    xs = np.cos(np.linspace(0, np.pi, n_grid))
    _validate_real_target_polynomial(coeffs, parity, xs)
    target = np.polyval(coeffs[::-1], xs)

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


def solve_phases_real_chebyshev(
    target_cheb_coeffs: np.ndarray,
    parity: int,
    **kwargs,
) -> tuple[np.ndarray, float]:
    """Solve scalar-QSP phases for a real target given in Chebyshev basis."""
    return solve_phases_real(chebyshev_to_monomial(target_cheb_coeffs), parity=parity, **kwargs)
