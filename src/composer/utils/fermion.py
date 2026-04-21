"""Jordan-Wigner fermion utilities.

Implements the fermion-to-qubit mapping used throughout the package
(see ASSUMPTIONS.md #1). We place spin-orbital p on qubit p; the
Jordan-Wigner string is Z_0 Z_1 ... Z_{p-1} on lower indices. Qubit
ordering: qubit 0 is the least-significant bit of the computational
basis index, i.e. state |x_{n-1} ... x_1 x_0> corresponds to integer
sum_p x_p 2**p. Occupation of spin-orbital p is x_p.

Functions here build *exact* dense matrices for second-quantized
operators on 2**n dimensional Fock space, used for numerical
verification of block encodings in ``tests/``. These matrices are
never used inside circuit simulation paths; they are ground-truth
references for Lemma/Theorem checks.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "fock_dim",
    "occupation_bits",
    "jw_a_dagger",
    "jw_a",
    "jw_number",
    "jw_mode_number",
    "one_body_matrix",
    "two_body_matrix",
    "determinant_index",
    "determinant_with_phase",
    "single_excitation_basis_indices",
]


def fock_dim(n_qubits: int) -> int:
    if n_qubits < 0:
        raise ValueError(f"n_qubits must be non-negative, got {n_qubits}")
    return 1 << n_qubits


def occupation_bits(index: int, n_qubits: int) -> np.ndarray:
    """Return occupation vector (x_0, x_1, ..., x_{n-1}) for basis index."""
    dim = fock_dim(n_qubits)
    if not (0 <= index < dim):
        raise ValueError(f"basis index {index} out of range for n_qubits={n_qubits}")
    return np.array([(index >> p) & 1 for p in range(n_qubits)], dtype=np.int8)


def _jw_sign(index: int, p: int) -> int:
    """JW parity sign: (-1)^(x_0 + ... + x_{p-1}).

    Implementing fermionic anticommutation: a_p |x> picks up a Z string
    on qubits 0..p-1.
    """
    mask = (1 << p) - 1
    return 1 - 2 * (bin(index & mask).count("1") & 1)


def _validate_orbital_index(p: int, n_qubits: int) -> None:
    if not (0 <= p < n_qubits):
        raise ValueError(f"orbital index {p} out of range for n_qubits={n_qubits}")


def jw_a_dagger(p: int, n_qubits: int) -> np.ndarray:
    """Creation operator a_p^dagger on 2**n Fock space (dense)."""
    _validate_orbital_index(p, n_qubits)
    dim = fock_dim(n_qubits)
    op = np.zeros((dim, dim), dtype=complex)
    bit = 1 << p
    for j in range(dim):
        if j & bit:
            continue  # already occupied
        sign = _jw_sign(j, p)
        i = j | bit
        op[i, j] = sign
    return op


def jw_a(p: int, n_qubits: int) -> np.ndarray:
    """Annihilation operator a_p on 2**n Fock space (dense)."""
    _validate_orbital_index(p, n_qubits)
    return jw_a_dagger(p, n_qubits).conj().T


def jw_number(p: int, n_qubits: int) -> np.ndarray:
    """Number operator n_p = a_p^dagger a_p (diagonal)."""
    _validate_orbital_index(p, n_qubits)
    dim = fock_dim(n_qubits)
    diag = np.array([(j >> p) & 1 for j in range(dim)], dtype=complex)
    return np.diag(diag)


def jw_mode_number(u: np.ndarray, n_qubits: int | None = None) -> np.ndarray:
    """Return n[u] = a^dagger[u] a[u] as a dense matrix.

    ``u`` is a one-particle coefficient vector in the repo's LSB-first
    spin-orbital ordering. If ``n_qubits`` is larger than ``len(u)``,
    the mode is embedded by zero-padding on higher orbital indices.
    """
    u = np.asarray(u, dtype=complex)
    if u.ndim != 1:
        raise ValueError(f"u must be 1-D, got shape {u.shape}")
    n_orbitals = int(u.shape[0])
    n = n_orbitals if n_qubits is None else n_qubits
    if n < n_orbitals:
        raise ValueError(f"n_qubits={n} < mode length {n_orbitals}")
    h_pq = np.outer(u, u.conj())
    if n > n_orbitals:
        padded = np.zeros((n, n), dtype=complex)
        padded[:n_orbitals, :n_orbitals] = h_pq
        h_pq = padded
    return one_body_matrix(h_pq)


def one_body_matrix(h_pq: np.ndarray) -> np.ndarray:
    """Return sum_{pq} h_{pq} a_p^dagger a_q as a dense matrix.

    Parameters
    ----------
    h_pq : (n, n) complex or real array

    Uses jw_a_dagger/jw_a; intended for small n only (n <= ~12).
    """
    h_pq = np.asarray(h_pq, dtype=complex)
    if h_pq.ndim != 2 or h_pq.shape[0] != h_pq.shape[1]:
        raise ValueError(f"h_pq must be square, got shape {h_pq.shape}")
    n = h_pq.shape[0]
    dim = fock_dim(n)
    H = np.zeros((dim, dim), dtype=complex)
    # Cache jw matrices to avoid rebuilding n^2 times
    adag = [jw_a_dagger(p, n) for p in range(n)]
    a = [jw_a(p, n) for p in range(n)]
    for p in range(n):
        for q in range(n):
            coef = h_pq[p, q]
            if coef == 0:
                continue
            H += coef * adag[p] @ a[q]
    return H


def two_body_matrix(eri_pqrs: np.ndarray) -> np.ndarray:
    """Return (1/2) sum_{pqrs} <pq|rs> a_p^dag a_q^dag a_s a_r (physicist order).

    Follows Eq. (10) of the paper. Intended for small n only.
    """
    eri_pqrs = np.asarray(eri_pqrs, dtype=complex)
    if eri_pqrs.ndim != 4:
        raise ValueError(f"eri_pqrs must be rank-4, got shape {eri_pqrs.shape}")
    n = eri_pqrs.shape[0]
    if eri_pqrs.shape != (n, n, n, n):
        raise ValueError(f"eri_pqrs must have shape (n, n, n, n), got {eri_pqrs.shape}")
    dim = fock_dim(n)
    H = np.zeros((dim, dim), dtype=complex)
    adag = [jw_a_dagger(p, n) for p in range(n)]
    a = [jw_a(p, n) for p in range(n)]
    for p in range(n):
        for q in range(n):
            for r in range(n):
                for s in range(n):
                    coef = eri_pqrs[p, q, r, s]
                    if coef == 0:
                        continue
                    # Operator order: a_p^dag a_q^dag a_s a_r
                    H += 0.5 * coef * adag[p] @ adag[q] @ a[s] @ a[r]
    return H


def determinant_index(occupied_orbitals: list[int] | tuple[int, ...], n_qubits: int) -> int:
    """Return the computational-basis index for a Slater determinant.

    The convention matches ``occupation_bits``: qubit p stores
    occupation of spin-orbital p. The ordering of ``occupied_orbitals``
    does not affect the index, but the *fermionic phase* relative to
    the ordered product a_{p_1}^dag a_{p_2}^dag ... |0> is
    left to the caller (see ``determinant_with_phase`` if needed).
    """
    idx = 0
    for p in occupied_orbitals:
        _validate_orbital_index(p, n_qubits)
        if idx & (1 << p):
            raise ValueError(f"orbital {p} occupied twice")
        idx |= 1 << p
    return idx


def determinant_with_phase(
    ordered_orbitals: list[int] | tuple[int, ...], n_qubits: int
) -> tuple[int, int]:
    """Compute (basis_index, fermionic_phase) for the ordered product
    a_{p_1}^dag ... a_{p_k}^dag |0>.

    The phase is determined by JW strings; we apply creation operators
    in the given order and track the sign. Orbitals must be distinct.
    """
    index = 0
    phase = 1
    for p in ordered_orbitals:
        _validate_orbital_index(p, n_qubits)
        bit = 1 << p
        if index & bit:
            return -1, 0  # vanishes
        phase *= _jw_sign(index, p)
        index |= bit
    return index, phase


def single_excitation_basis_indices(n_qubits: int) -> np.ndarray:
    """Indices of the 1-electron subspace: basis states with exactly one occupied qubit.

    Returns the indices in the order (|0>_p)_{p=0}^{n-1} where |0>_p has
    orbital p occupied. Used in ``test_bilinear_be.py`` to check
    Lemma 1 on the single-excitation subspace only.
    """
    fock_dim(n_qubits)
    return np.array([1 << p for p in range(n_qubits)], dtype=np.int64)
