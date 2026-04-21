"""Antisymmetric pair indexing and wedge-product helpers.

Used by the two-electron state-preparation ladder (App A.2) and the
antisymmetric pair SVD of the cluster generator sigma-hat (Sec II.C,
Eq 18-27).

Conventions
-----------
* A pair ``(p, q)`` with ``p < q`` is ordered lexicographically by
  ``(p, q)``; the flat index is ``pair_index(p, q, n)``.
* An antisymmetric n x n matrix ``u`` is stored in full dense form
  for simplicity; ``pairs_from_matrix`` returns its strictly-upper
  triangular entries as a 1-D array.
* ``pair_matrix_from_vector`` is the inverse (and anti-symmetrizes).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "num_pairs",
    "pair_index",
    "index_to_pair",
    "pairs_from_matrix",
    "pair_matrix_from_vector",
    "wedge_vector_from_matrix",
]


def num_pairs(n: int) -> int:
    return n * (n - 1) // 2


def pair_index(p: int, q: int, n: int) -> int:
    """Flat index of the ordered pair (p < q). n is the orbital count."""
    if not (0 <= p < q < n):
        raise ValueError(f"require 0 <= p < q < n, got (p, q, n) = ({p}, {q}, {n})")
    # sum_{i=0}^{p-1} (n-1-i) + (q - p - 1)
    return p * (2 * n - p - 1) // 2 + (q - p - 1)


def index_to_pair(k: int, n: int) -> tuple[int, int]:
    """Inverse of pair_index."""
    if not (0 <= k < num_pairs(n)):
        raise ValueError(f"k={k} out of range for n={n}")
    p = 0
    remaining = k
    # number of pairs starting with p is (n - 1 - p)
    while remaining >= (n - 1 - p):
        remaining -= (n - 1 - p)
        p += 1
    q = p + 1 + remaining
    return p, q


def pairs_from_matrix(u: np.ndarray) -> np.ndarray:
    """Return the strictly-upper-triangular entries of u as a 1-D vector.

    Entry k corresponds to ``(p, q) = index_to_pair(k, n)`` and equals
    ``u[p, q]`` (no anti-symmetrization applied; u is assumed to be
    antisymmetric or the caller is choosing the upper part).
    """
    n = u.shape[0]
    m = num_pairs(n)
    out = np.zeros(m, dtype=u.dtype)
    for k in range(m):
        p, q = index_to_pair(k, n)
        out[k] = u[p, q]
    return out


def pair_matrix_from_vector(v: np.ndarray, n: int, *, antisymmetric: bool = True) -> np.ndarray:
    """Build an n x n matrix from a flat pair vector.

    If antisymmetric=True (default) the lower-triangular part is set to
    -v[k]; otherwise it is left zero.
    """
    if v.shape[0] != num_pairs(n):
        raise ValueError(f"expected length {num_pairs(n)}, got {v.shape[0]}")
    mat = np.zeros((n, n), dtype=v.dtype)
    for k in range(num_pairs(n)):
        p, q = index_to_pair(k, n)
        mat[p, q] = v[k]
        if antisymmetric:
            mat[q, p] = -v[k]
    return mat


def wedge_vector_from_matrix(u: np.ndarray) -> np.ndarray:
    """Given an n x n matrix U, return the wedge-product vector
    w_{p<q} = U[p, :] wedge U[q, :] projected onto the (p<q) basis,
    which concretely is the 2x2 minor det(U[[p,q], [p,q]])  --
    but for a *one-body* unitary acting on two-electron pair states
    we need the full 2x2 minor over orbital indices, not spatial.

    The helper is used by the two-electron ladder: if the target pair
    state is parameterized by a matrix M of shape (n, n), its (p, q)
    amplitude with p < q is M[p, q] - M[q, p] (i.e., the antisymmetric
    combination). We expose this as a utility so that callers who
    supply an already-antisymmetric matrix can just use
    pairs_from_matrix instead.
    """
    n = u.shape[0]
    asym = u - u.T
    return pairs_from_matrix(asym)
