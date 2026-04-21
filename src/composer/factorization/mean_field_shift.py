"""Mean-field shift of the one-body Hamiltonian (Eq 12).

Starting from the standard second-quantized Hamiltonian

    H = sum_{pq} h_{pq} a_p^dag a_q
        + (1/2) sum_{pqrs} <pq|rs> a_p^dag a_q^dag a_s a_r         (Eq 10)

we rewrite the two-body term using
``a_p^dag a_q^dag a_s a_r = (a_p^dag a_r)(a_q^dag a_s) - delta_{qr} a_p^dag a_s``
so that the contraction produces an effective one-body shift:

    H = sum_{pq} h~_{pq} a_p^dag a_q
        + (1/2) sum_{mu} O_mu^2                                    (Eq 13-14)

with

    h~_{pq} = h_{pq} - (1/2) sum_t <pt|tq>                          (Eq 12)

and ``O_mu = sum_{pr} L^mu_{pr} a_p^dag a_r`` from the Cholesky
factorization of ``<pq|rs>`` (Eq 11).
"""
from __future__ import annotations

import numpy as np

__all__ = ["mean_field_shifted_h"]


def mean_field_shifted_h(h: np.ndarray, eri: np.ndarray) -> np.ndarray:
    """Return h~_{pq} = h_{pq} - (1/2) sum_t <pt|tq>.

    Parameters
    ----------
    h : (n, n) complex or real array
        Bare one-electron integrals ``h_{pq} = <p|t + v_ext|q>``.
    eri : (n, n, n, n) complex or real array
        Physicist-notation two-electron integrals ``<pq|rs>``.
    """
    h = np.asarray(h, dtype=complex)
    eri = np.asarray(eri, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"h must have shape (n, n), got {h.shape}")
    n = h.shape[0]
    if eri.shape != (n, n, n, n):
        raise ValueError(f"eri shape {eri.shape} incompatible with h shape {h.shape}")
    # shift[p, q] = sum_t eri[p, t, t, q]
    shift = np.einsum("ptts->ps", eri)
    return h - 0.5 * shift
