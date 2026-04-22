"""Chebyshev expansions of ``e^{-i alpha x}`` and its parity components.

Implements the Jacobi-Anger expansion

    e^{-i alpha x} = J_0(alpha) + 2 sum_{k>=1} (-i)^k J_k(alpha) T_k(x)   (x in [-1, 1])

and convenience helpers for the real and imaginary projections

    cos(alpha x) = J_0(alpha) + 2 sum_{m>=1} (-1)^m J_{2m}(alpha) T_{2m}(x)
    sin(alpha x) = 2 sum_{m>=0} (-1)^m J_{2m+1}(alpha) T_{2m+1}(x)

The truncation degree follows ASSUMPTION #7

    d = ceil(2 (e alpha / 2 + log(2 / eps)))

which is sufficient for a uniform truncation error ``eps`` on ``[-1, 1]``
(standard Jacobi-Anger bound; see Appendix C.2 of the paper).

Used by ``qsp/phases.py`` and ``block_encoding/generator_exp.py`` as
the scalar source for the generator-exponential phase compiler. The
compiler now treats the direct complex Jacobi-Anger target
``exp(-i alpha x)`` as primary, then splits that target into its even
and odd Chebyshev sectors only because the current repo's single-ladder
Wx/top-left model can realize only definite-parity scalar polynomials.
The retained dense Chebyshev evaluation is a reference helper only; it
is not the main construction path for
``build_generator_exp_oracle(...)``.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from scipy.special import jv

__all__ = [
    "chebyshev_parity",
    "recommended_degree",
    "recommended_degree_with_parity",
    "jacobi_anger_coefficients",
    "cos_alpha_x_coefficients",
    "sin_alpha_x_coefficients",
    "split_chebyshev_by_parity",
    "split_exponential_chebyshev_components",
    "trim_chebyshev_coefficients",
    "chebyshev_to_monomial",
    "truncation_error_bound",
    "evaluate_chebyshev",
]


def trim_chebyshev_coefficients(coeffs: np.ndarray, *, atol: float = 1e-14) -> np.ndarray:
    """Drop trailing near-zero Chebyshev coefficients."""
    coeffs = np.asarray(coeffs)
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be one-dimensional")
    last = coeffs.shape[0] - 1
    while last > 0 and abs(coeffs[last]) <= atol:
        last -= 1
    return coeffs[: last + 1].copy()


def split_chebyshev_by_parity(coeffs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the even- and odd-index Chebyshev sectors of ``coeffs``."""
    coeffs = np.asarray(coeffs)
    if coeffs.ndim != 1:
        raise ValueError("coeffs must be one-dimensional")
    even = np.zeros_like(coeffs)
    odd = np.zeros_like(coeffs)
    even[::2] = coeffs[::2]
    odd[1::2] = coeffs[1::2]
    return even, odd


def chebyshev_parity(coeffs: np.ndarray, *, atol: float = 1e-12) -> int | None:
    """Return the definite parity of ``coeffs`` or ``None`` if mixed."""
    even, odd = split_chebyshev_by_parity(coeffs)
    has_even = np.any(np.abs(even) > atol)
    has_odd = np.any(np.abs(odd) > atol)
    if has_even and has_odd:
        return None
    if has_even:
        return 0
    if has_odd:
        return 1
    return 0


def recommended_degree(alpha: float, eps: float) -> int:
    """ASSUMPTION #7 degree heuristic.

    Matches the paper's Eq (C4) up to constants and is sufficient for
    ``|e^{-i alpha x} - sum_{k<=d} c_k T_k(x)| <= eps`` on ``[-1, 1]``.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if not (0 < eps < 1):
        raise ValueError("eps must be in (0, 1)")
    return int(np.ceil(2.0 * (np.e * alpha / 2.0 + np.log(2.0 / eps))))


def recommended_degree_with_parity(alpha: float, eps: float, parity: int) -> int:
    """Return the Jacobi-Anger degree adjusted to the requested parity."""
    if parity not in (0, 1):
        raise ValueError("parity must be 0 or 1")
    degree = recommended_degree(alpha, eps)
    if degree % 2 != parity:
        degree += 1
    return degree


def jacobi_anger_coefficients(alpha: float, degree: int) -> np.ndarray:
    """Complex Chebyshev coefficients ``c_0, ..., c_d`` of ``e^{-i alpha x}``.

    ``c_0 = J_0(alpha)`` and ``c_k = 2 (-i)^k J_k(alpha)`` for ``k >= 1``.
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    ks = np.arange(degree + 1)
    coeffs = np.empty(degree + 1, dtype=complex)
    coeffs[0] = jv(0, alpha)
    if degree >= 1:
        coeffs[1:] = 2.0 * ((-1j) ** ks[1:]) * jv(ks[1:], alpha)
    return coeffs


def cos_alpha_x_coefficients(alpha: float, degree: int) -> np.ndarray:
    """Real Chebyshev coefficients of ``cos(alpha x)`` up to ``degree``.

    The result is a length-``degree+1`` array; odd-index entries are zero.
    """
    coeffs = jacobi_anger_coefficients(alpha, degree).real
    return coeffs


def sin_alpha_x_coefficients(alpha: float, degree: int) -> np.ndarray:
    """Real Chebyshev coefficients of ``sin(alpha x)`` up to ``degree``.

    ``e^{-i alpha x} = cos - i sin``, so the imaginary part of the
    Jacobi-Anger coefficients carries ``-sin``. The even-index entries
    of the result are zero.
    """
    coeffs = -jacobi_anger_coefficients(alpha, degree).imag
    return coeffs


def split_exponential_chebyshev_components(
    complex_coeffs: np.ndarray,
    *,
    atol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive ``cos(alpha x)`` and ``sin(alpha x)`` branches from ``exp(-i alpha x)``.

    The returned arrays are real Chebyshev coefficients trimmed to their
    natural parity-valid degrees. This keeps the structured fallback
    rooted in the direct complex target rather than recompiling the real
    branches independently.
    """
    complex_coeffs = np.asarray(complex_coeffs, dtype=complex).ravel()
    even, odd = split_chebyshev_by_parity(complex_coeffs)
    if np.max(np.abs(even.imag), initial=0.0) > atol:
        raise ValueError("even Chebyshev sector of exp(-i alpha x) should be real")
    if np.max(np.abs(odd.real), initial=0.0) > atol:
        raise ValueError("odd Chebyshev sector of exp(-i alpha x) should be purely imaginary")
    cos_coeffs = trim_chebyshev_coefficients(even.real, atol=atol)
    sin_coeffs = trim_chebyshev_coefficients((1j * odd).real, atol=atol)
    return cos_coeffs, sin_coeffs


def truncation_error_bound(alpha: float, degree: int) -> float:
    """Upper bound on the uniform error ``max_x |e^{-i alpha x} - trunc|``.

    Uses the tail-sum bound ``sum_{k>d} 2 |J_k(alpha)|``. Bessel
    magnitudes decay factorially once ``k > e alpha / 2``, so we sum a
    generous window beyond ``degree`` until contributions drop below
    machine epsilon.
    """
    total = 0.0
    k = degree + 1
    while k < degree + 200:
        term = abs(jv(k, alpha))
        total += 2.0 * term
        if term < 1e-18 and k > np.e * alpha / 2 + 4:
            break
        k += 1
    return float(total)


def chebyshev_to_monomial(coeffs: np.ndarray) -> np.ndarray:
    """Convert Chebyshev-series coefficients into monomial coefficients."""
    return _cheb.cheb2poly(np.asarray(coeffs))


def evaluate_chebyshev(coeffs: np.ndarray, x: np.ndarray | float) -> np.ndarray:
    """Evaluate a Chebyshev series at ``x``. Wrapper around numpy's chebval.

    Supports complex coefficients. ``x`` can be a scalar or array.
    """
    x_arr = np.asarray(x)
    if np.iscomplexobj(coeffs):
        re = _cheb.chebval(x_arr, coeffs.real)
        im = _cheb.chebval(x_arr, coeffs.imag)
        return re + 1j * im
    return _cheb.chebval(x_arr, coeffs)
