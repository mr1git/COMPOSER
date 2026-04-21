"""Assemble the rank-one pool of the molecular Hamiltonian (Eq 13-17).

Starting from mean-field-shifted one-body ``h~_{pq}`` (Eq 12) and the
Cholesky factorization ``<pq|rs> = sum_mu L^mu_{pr} L^mu_{qs}`` (Eq 11,
real case), we rewrite

    H = sum_{pq} h~_{pq} a_p^dag a_q  +  (1/2) sum_{mu} O_mu^2      (Eq 13)

with ``O_mu = sum_{pr} L^mu_{pr} a_p^dag a_r``. This module stores the
classical data (``h~`` and the Cholesky tensor ``L``) in a single
``HamiltonianPool`` dataclass and exposes:

* ``dense_matrix()`` that reconstructs the full Hamiltonian via
  Jordan-Wigner for small-system verification (the reference for the
  Theorem 1 / Lemma 2 tests).
* ``one_body_eigendecomposition()`` that diagonalizes ``h~`` into a
  sum of Hermitian rank-one bilinears (the `Def 1, u = v` pool used in
  the LCU of Theorem 1).
* ``hermitian_antihermitian_split(mu)`` that splits a single Cholesky
  matrix ``L^mu = H^mu + i * B^mu`` with both ``H^mu`` and ``B^mu``
  Hermitian; Lemma 2's diagonalized-channel block encoding works
  channel-by-channel on each Hermitian piece and the test in
  ``test_cholesky_channel_be.py`` drives it that way.

Cross terms between the Hermitian and anti-Hermitian sub-channels are
two-body and are *not* converted to an equivalent one-body object.
They are handled at the block-encoding level: Theorem 1's LCU sees the
full unsplit ``O_mu^2`` per channel (one circuit per ``mu``), not the
individual sub-squares; the split is used only inside Lemma 2 for the
``x -> x^2`` polynomial construction of *each* Hermitian sub-channel.

Scope note
----------
The low-level Cholesky primitive in ``factorization/cholesky.py``
supports generic complex Hermitian-PSD matricized ERIs. The Hamiltonian
pool built here does not: Eq. (13)-(17) and the current LCU path are
implemented only for the real-valued physicist-order electronic-integral
case, where the Cholesky factors are real symmetric matrices. Complex
generalizations would require a different quadratic-channel
representation and are intentionally rejected in ``build_pool_from_integrals``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..factorization.cholesky import cholesky_eri, reconstruct_eri
from ..factorization.mean_field_shift import mean_field_shifted_h
from ..utils import fermion as jw
from ..utils.linalg import hermitian_eig

__all__ = [
    "HamiltonianPool",
    "OneBodyEigenChannel",
    "build_pool_from_integrals",
]

_REAL_POOL_ATOL = 1e-10


@dataclass
class OneBodyEigenChannel:
    """A Hermitian rank-one bilinear ``e * a^dag[phi] a[phi]``.

    Sign of ``e`` is kept explicit (the LCU selector carries it).
    """

    phi: np.ndarray
    coeff: float


@dataclass
class HamiltonianPool:
    """Classical data backing the rank-one pool of H."""

    n_orbitals: int
    h_tilde: np.ndarray  # (n, n) mean-field-shifted one-body
    cholesky_factors: np.ndarray  # (K, n, n): L[mu, p, r]
    constant: float = 0.0

    # -- derived views --------------------------------------------------

    def one_body_eigendecomposition(self) -> list[OneBodyEigenChannel]:
        """Diagonalize ``h~`` into Hermitian rank-one channels.

        ``h~ = sum_k e_k phi_k phi_k^dag`` gives
        ``sum_k e_k a^dag[phi_k] a[phi_k]``.
        """
        eigvals, V = hermitian_eig(self.h_tilde)
        return [
            OneBodyEigenChannel(phi=V[:, k].copy(), coeff=float(eigvals[k].real))
            for k in range(self.n_orbitals)
        ]

    def hermitian_antihermitian_split(self, mu: int) -> tuple[np.ndarray, np.ndarray]:
        """Return (H^mu, B^mu) with H^mu Hermitian, B^mu Hermitian, and
        L^mu = H^mu + i B^mu.

        Concretely H^mu = (L^mu + L^mu^dag) / 2, and
        B^mu = (L^mu - L^mu^dag) / (2 i).
        """
        L = self.cholesky_factors[mu]
        H = 0.5 * (L + L.conj().T)
        B = -0.5j * (L - L.conj().T)
        return H, B

    # -- reconstruction ------------------------------------------------

    def dense_matrix(self) -> np.ndarray:
        """Full Hamiltonian as a dense 2**n x 2**n matrix.

        Reconstruction is performed from the stored mean-field-shifted
        one-body matrix and the ERI tensor implied by the Cholesky
        factors. We first undo Eq. (12),

            h_pq = h~_pq + (1/2) sum_t <pt|tq>,

        and then rebuild

            H = sum_pq h~_pq a_p^dag a_q
                + (1/2) sum_pqrs <pq|rs> a_p^dag a_q^dag a_s a_r

        with ``<pq|rs>`` recovered by ``reconstruct_eri``. For the
        supported real-valued pool this equals the paper's
        ``h_tilde + (1/2) sum_mu O_mu^2`` identity. Using the exact
        factorized ERI here keeps the dense reference correct even if a
        ``HamiltonianPool`` is instantiated manually outside that scope.

        Used by tests as the ground-truth reference against Theorem 1's
        block encoding. Intended for small n only (n <= ~6).
        """
        eri = reconstruct_eri(self.cholesky_factors)
        shift = np.einsum("ptts->ps", eri)
        h = self.h_tilde + 0.5 * shift
        H = jw.one_body_matrix(h)
        H += jw.two_body_matrix(eri)
        dim = H.shape[0]
        if self.constant != 0.0:
            H += self.constant * np.eye(dim, dtype=complex)
        return H


def build_pool_from_integrals(
    h: np.ndarray,
    eri: np.ndarray,
    *,
    cholesky_threshold: float = 1e-10,
) -> HamiltonianPool:
    """Construct the rank-one pool from integrals.

    Parameters
    ----------
    h : (n, n) one-electron integrals.
    eri : (n, n, n, n) physicist-order two-electron integrals <pq|rs>.
    cholesky_threshold : residual-diagonal threshold for pivoted Cholesky.
    """
    h = np.asarray(h, dtype=complex)
    eri = np.asarray(eri, dtype=complex)
    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError(f"h must have shape (n, n), got {h.shape}")
    n = h.shape[0]
    if eri.shape != (n, n, n, n):
        raise ValueError(f"eri must have shape {(n, n, n, n)}, got {eri.shape}")
    if not np.allclose(h, h.conj().T, atol=_REAL_POOL_ATOL):
        raise ValueError("h must be Hermitian")

    max_imag = max(
        float(np.max(np.abs(h.imag))),
        float(np.max(np.abs(eri.imag))),
    )
    if max_imag > _REAL_POOL_ATOL:
        raise NotImplementedError(
            "build_pool_from_integrals currently supports only real-valued "
            "physicist-order electronic integrals. Generic complex Hermitian-PSD "
            "ERIs are supported by cholesky_eri(), but the Eq. (13)-(17) "
            "projected-quadratic pool used by HamiltonianPool / LCU is not "
            "implemented for that case."
        )

    h_tilde = mean_field_shifted_h(h, eri)
    L = cholesky_eri(eri, threshold=cholesky_threshold)
    h_tilde = np.real_if_close(h_tilde, tol=1000)
    L = np.real_if_close(L, tol=1000)
    if np.max(np.abs(np.imag(h_tilde))) > _REAL_POOL_ATOL or np.max(np.abs(np.imag(L))) > _REAL_POOL_ATOL:
        raise NotImplementedError("Hamiltonian preprocessing produced unexpected complex data")
    h_tilde = np.asarray(h_tilde.real)
    L = np.asarray(L.real)
    if not np.allclose(h_tilde, h_tilde.T, atol=_REAL_POOL_ATOL):
        raise ValueError("mean-field-shifted h_tilde must be symmetric for the supported real-valued scope")
    if not np.allclose(L, L.transpose(0, 2, 1), atol=_REAL_POOL_ATOL):
        raise NotImplementedError(
            "HamiltonianPool currently requires Cholesky factors L^mu to be real "
            "symmetric. Non-symmetric factors indicate either unsupported complex "
            "orbitals or an integral-convention mismatch outside the paper's "
            "implemented Eq. (13)-(17) scope."
        )
    return HamiltonianPool(
        n_orbitals=int(n),
        h_tilde=h_tilde,
        cholesky_factors=L,
    )
