"""Antisymmetric-pair SVD of a CCSD-like ``t2`` amplitude tensor
(Sec II.C, Eq. 18-27).

Given ``t2[a, b, i, j]`` antisymmetric in ``(a, b)`` and ``(i, j)``, we
unfold it into a matrix of shape ``(NV 2) x (NO 2)`` indexed by
ordered pairs ``(a<b), (i<j)``, SVD that matrix, and return the
sorted singular factors. Each left/right singular vector is a pair
vector that, via the helper ``pair_matrix_from_vector``, reconstructs
an antisymmetric ``NV x NV`` (resp. ``NO x NO``) matrix.

The result is a list of ``PairChannel`` entries:

    t2[a, b, i, j]  =  sum_mu sigma_mu * U_mu[a, b] * V_mu[i, j]^*

with ``||U_mu||_F = ||V_mu||_F = sqrt(2)`` (from the off-diagonal
antisymmetric convention, equivalent to unit pair-norm).

The conjugation on ``V_mu`` is deliberate: ``PairChannel`` stores the
occupied factor in the operator convention of Def. 2,

    L_mu = sigma_mu B^dag[U_mu] B[V_mu],

where ``B[V] = sum_{i<j} V_{ij}^* a_j a_i``. With this convention, the
stored channel data is already aligned with the paper's rank-one
operator oracle rather than only with a tensor outer product.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils.antisymmetric import num_pairs, pair_index, pair_matrix_from_vector, pairs_from_matrix

__all__ = ["PairChannel", "EmbeddedPairChannel", "pair_svd_decompose", "reconstruct_t2"]


@dataclass
class PairChannel:
    """A single rank-one pair channel.

    ``sigma`` is the singular value. ``U`` and ``V`` are antisymmetric
    matrices whose strictly upper-triangular entries have unit Euclidean
    norm, so ``||U||_F = ||V||_F = sqrt(2)``. ``V`` is stored in the
    operator convention of Def. 2, so the coefficient tensor recovered
    from the channel is ``sigma * U[a,b] * V[i,j]^*``.
    """

    sigma: float
    U: np.ndarray
    V: np.ndarray

    def __post_init__(self) -> None:
        self.U = np.asarray(self.U, dtype=complex)
        self.V = np.asarray(self.V, dtype=complex)

    def coefficient_tensor(self) -> np.ndarray:
        """Return the Eq. (18)/(26) doubles-amplitude tensor for this channel."""
        return self.sigma * np.einsum("ab,ij->abij", self.U, self.V.conj())

    def pair_space_matrix(self) -> np.ndarray:
        """Return this channel as a rank-one matrix on pair spaces."""
        return self.sigma * np.outer(pairs_from_matrix(self.U), pairs_from_matrix(self.V).conj())

    def embed(
        self,
        creation_orbitals: tuple[int, ...] | list[int],
        annihilation_orbitals: tuple[int, ...] | list[int],
    ) -> "EmbeddedPairChannel":
        """Attach occupied/virtual embeddings so the channel becomes an explicit
        paper-style operator branch.
        """
        return EmbeddedPairChannel(
            sigma=self.sigma,
            U=self.U,
            V=self.V,
            creation_orbitals=tuple(creation_orbitals),
            annihilation_orbitals=tuple(annihilation_orbitals),
        )


@dataclass
class EmbeddedPairChannel(PairChannel):
    """A pair channel with explicit creation/annihilation embeddings."""

    creation_orbitals: tuple[int, ...]
    annihilation_orbitals: tuple[int, ...]

    def __post_init__(self) -> None:
        super().__post_init__()
        self.creation_orbitals = tuple(int(p) for p in self.creation_orbitals)
        self.annihilation_orbitals = tuple(int(p) for p in self.annihilation_orbitals)

    def as_pair_rank_one(self):
        """Return the explicit Def. 2 operator for this embedded channel."""
        from ..operators.rank_one import PairRankOne

        return PairRankOne(
            U=self.U,
            V=self.V,
            coeff=self.sigma,
            creation_orbitals=self.creation_orbitals,
            annihilation_orbitals=self.annihilation_orbitals,
        )

    def dense_excitation(self, n_qubits: int | None = None) -> np.ndarray:
        return self.as_pair_rank_one().dense_matrix(n_qubits=n_qubits)

    def dense_sigma(self, n_qubits: int | None = None) -> np.ndarray:
        term = self.as_pair_rank_one()
        return term.dense_sigma_term(n_qubits=n_qubits)


def _assert_antisymmetric(t: np.ndarray, axes: tuple[int, int]) -> None:
    swapped = np.swapaxes(t, axes[0], axes[1])
    if not np.allclose(t, -swapped, atol=1e-10):
        raise ValueError(f"tensor not antisymmetric in axes {axes}")


def pair_svd_decompose(t2: np.ndarray, tol: float = 1e-12) -> list[PairChannel]:
    """Decompose ``t2`` into a list of non-negligible pair rank-one channels.

    Parameters
    ----------
    t2 : (NV, NV, NO, NO) array, antisymmetric in (a,b) and (i,j).
    tol : relative tolerance on singular values (drop channels with
        ``sigma < tol * sigma_max``).
    """
    if t2.ndim != 4:
        raise ValueError("t2 must be 4-D")
    NV, NV2, NO, NO2 = t2.shape
    if NV != NV2 or NO != NO2:
        raise ValueError("t2 axes (a,b,i,j) must have shapes NV,NV,NO,NO")
    _assert_antisymmetric(t2, (0, 1))
    _assert_antisymmetric(t2, (2, 3))

    # Unfold into matrix M[(ab), (ij)] using pair indexing.
    n_ab = num_pairs(NV)
    n_ij = num_pairs(NO)
    M = np.zeros((n_ab, n_ij), dtype=t2.dtype)
    for a in range(NV):
        for b in range(a + 1, NV):
            for i in range(NO):
                for j in range(i + 1, NO):
                    M[pair_index(a, b, NV), pair_index(i, j, NO)] = t2[a, b, i, j]

    U_mat, sigma, Vh_mat = np.linalg.svd(M, full_matrices=False)
    threshold = tol * max(sigma.max(), 1.0) if sigma.size else 0.0

    channels: list[PairChannel] = []
    for mu in range(sigma.shape[0]):
        if sigma[mu] < threshold:
            break
        # Map pair vectors back to antisymmetric matrices. ``V`` is stored in
        # the operator convention of Def. 2, so it is the conjugate of the
        # amplitude-space right factor appearing in the unfolded ``t2`` matrix.
        U_vec = U_mat[:, mu]
        V_vec = Vh_mat[mu, :].conj()
        U_am = pair_matrix_from_vector(U_vec, NV)
        V_am = pair_matrix_from_vector(V_vec, NO)
        channels.append(PairChannel(sigma=float(sigma[mu]), U=U_am, V=V_am))
    return channels


def reconstruct_t2(channels: list[PairChannel], NV: int, NO: int) -> np.ndarray:
    """Inverse of ``pair_svd_decompose``: sum rank-one channels back."""
    t2 = np.zeros((NV, NV, NO, NO), dtype=complex)
    for ch in channels:
        t2 += ch.coefficient_tensor()
    return t2
