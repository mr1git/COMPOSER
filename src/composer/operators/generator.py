"""Anti-Hermitian σ̂ generator and its pair-SVD channelization (Eq. 18-27).

The coupled-cluster / UCC-style generator has the structure

    σ̂  =  T - T^dag

with

    T  =  sum_{ai}  t^a_i  a_a^dag a_i                         (singles)
         +  (1/4) sum_{abij}  t^{ab}_{ij}  a_a^dag a_b^dag a_j a_i     (doubles)

This module stores ``t1`` (singles) and ``t2`` (doubles, antisymmetric
in ``(a,b)`` and ``(i,j)``) on a user-defined split of spin-orbitals
into occupied ``O = [0..NO)`` and virtual ``V = [NO..NO+NV)``.
``t1.shape == (NV, NO)`` and ``t2.shape == (NV, NV, NO, NO)``.

It produces
* ``dense_sigma(n_qubits)`` - the full ``2**n x 2**n`` anti-Hermitian
  matrix for small-system verification,
* ``singles_channels()`` - the explicit bilinear singles terms
  ``t_ai a_a^dag a_i``,
* ``doubles_channels()`` / ``pair_rank_one_pool()`` - the explicit
  embedded rank-one doubles channels of the antisymmetric-pair
  factorized ``t2`` tensor, and
* ``generator_channels()`` - the compiled singles+doubles pool used by
  the sigma oracle, and
* ``doubles_excitation_terms()`` / ``dense_doubles_sigma()`` - the
  second-quantized doubles operators reconstructed from those channels.

The paper's channel identity is

    T2 = sum_mu sigma_mu B^dag[U_mu] B[V_mu]
    sigma_doubles = T2 - T2^dag

with ``U_mu`` embedded on the virtual space and ``V_mu`` embedded on
the occupied space. The code below exposes those embedded channel
objects directly so later SELECT/PREP logic can consume them without
first reconstructing dense surrogates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..factorization.pair_svd import EmbeddedPairChannel, pair_svd_decompose
from .rank_one import BilinearRankOne, PairRankOne
from ..utils import fermion as jw

__all__ = ["SingleExcitationChannel", "ClusterGenerator", "build_cluster_generator"]


@dataclass
class SingleExcitationChannel:
    """One rank-one singles excitation ``t_ai a_a^dag a_i`` on the full orbital register."""

    coeff: complex
    creation_orbital: int
    annihilation_orbital: int
    n_orbitals: int

    def _basis_vector(self, orbital: int) -> np.ndarray:
        vec = np.zeros(self.n_orbitals, dtype=complex)
        vec[orbital] = 1.0
        return vec

    def as_bilinear_rank_one(self) -> BilinearRankOne:
        """Return the Def. 1 bilinear with the phase absorbed into ``u``."""
        magnitude = float(abs(self.coeff))
        if magnitude <= 0.0:
            raise ValueError("single-excitation channel coefficient must be nonzero")
        phase = self.coeff / magnitude
        u = phase * self._basis_vector(self.creation_orbital)
        v = self._basis_vector(self.annihilation_orbital)
        return BilinearRankOne(u=u, v=v, coeff=magnitude)

    def coefficient_matrix(self) -> np.ndarray:
        """Return the orbital-space matrix for ``t_ai a_a^dag a_i``."""
        mat = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        mat[self.creation_orbital, self.annihilation_orbital] = self.coeff
        return mat

    def hermitian_generator_matrix(self) -> np.ndarray:
        """Return the Hermitian one-body matrix for ``-i (L - L^dag)``."""
        mat = np.zeros((self.n_orbitals, self.n_orbitals), dtype=complex)
        mat[self.creation_orbital, self.annihilation_orbital] = -1j * self.coeff
        mat[self.annihilation_orbital, self.creation_orbital] = 1j * self.coeff.conjugate()
        return mat

    def dense_excitation(self, n_qubits: int | None = None) -> np.ndarray:
        return self.as_bilinear_rank_one().dense_matrix(n_qubits=n_qubits)

    def dense_sigma(self, n_qubits: int | None = None) -> np.ndarray:
        L = self.dense_excitation(n_qubits=n_qubits)
        return L - L.conj().T


@dataclass
class ClusterGenerator:
    """CCSD-like amplitudes backing the anti-Hermitian σ̂."""

    NO: int  # number of occupied spin-orbitals
    NV: int  # number of virtual spin-orbitals
    t1: np.ndarray  # (NV, NO) singles
    t2: np.ndarray  # (NV, NV, NO, NO) doubles (antisymmetric in (a,b), (i,j))

    @property
    def n_orbitals(self) -> int:
        return self.NO + self.NV

    def dense_sigma(self, n_qubits: int | None = None) -> np.ndarray:
        """Dense ``sigma = T - T^dag`` as a 2**n x 2**n matrix."""
        n_orb = self.n_orbitals
        n = n_orb if n_qubits is None else n_qubits
        dim = jw.fock_dim(n)
        adag = [jw.jw_a_dagger(p, n) for p in range(n_orb)]
        a_ = [jw.jw_a(p, n) for p in range(n_orb)]

        T = np.zeros((dim, dim), dtype=complex)
        for a in range(self.NV):
            p = self.NO + a
            for i in range(self.NO):
                if self.t1[a, i] != 0:
                    T += self.t1[a, i] * (adag[p] @ a_[i])
        for a in range(self.NV):
            p = self.NO + a
            for b in range(self.NV):
                q = self.NO + b
                for i in range(self.NO):
                    for j in range(self.NO):
                        c = self.t2[a, b, i, j]
                        if c != 0:
                            T += 0.25 * c * (adag[p] @ adag[q] @ a_[j] @ a_[i])
        return T - T.conj().T

    def singles_channels(self, tol: float = 1e-12) -> list[SingleExcitationChannel]:
        """Return the explicit singles excitation channels ``t_ai a_a^dag a_i``."""
        channels: list[SingleExcitationChannel] = []
        for a in range(self.NV):
            p = self.NO + a
            for i in range(self.NO):
                coeff = self.t1[a, i]
                if abs(coeff) <= tol:
                    continue
                channels.append(
                    SingleExcitationChannel(
                        coeff=coeff,
                        creation_orbital=p,
                        annihilation_orbital=i,
                        n_orbitals=self.n_orbitals,
                    )
                )
        return channels

    def doubles_channels(self, tol: float = 1e-12) -> list[EmbeddedPairChannel]:
        """Return the paper-style doubles channels ``L_mu = sigma_mu B^dag[U_mu] B[V_mu]``.

        Each channel carries the explicit occupied/virtual embedding
        needed by later SELECT/PREP logic.
        """
        creation_orbitals = tuple(range(self.NO, self.NO + self.NV))
        annihilation_orbitals = tuple(range(self.NO))
        return [
            ch.embed(creation_orbitals=creation_orbitals, annihilation_orbitals=annihilation_orbitals)
            for ch in pair_svd_decompose(self.t2, tol=tol)
        ]

    def pair_rank_one_pool(self, tol: float = 1e-12) -> list[EmbeddedPairChannel]:
        """Compatibility alias for the embedded doubles-channel pool."""
        return self.doubles_channels(tol=tol)

    def generator_channels(self, tol: float = 1e-12) -> list[SingleExcitationChannel | EmbeddedPairChannel]:
        """Return the compiled generator pool used by the sigma oracle.

        The paper's Sec. II.C treats singles as bilinear rank-one terms
        and doubles as pair-rank-one channels. The generator-side oracle
        therefore sees the concatenated pool

            singles first, then pair-SVD doubles channels.
        """
        return [*self.singles_channels(tol=tol), *self.doubles_channels(tol=tol)]

    def doubles_excitation_terms(self, tol: float = 1e-12) -> list[PairRankOne]:
        """Return the rank-one doubles excitations ``sigma_mu B^dag[U_mu] B[V_mu]``.

        ``U_mu`` acts on the virtual block ``[NO, NO + NV)`` and
        ``V_mu`` acts on the occupied block ``[0, NO)``.
        """
        return [ch.as_pair_rank_one() for ch in self.doubles_channels(tol=tol)]

    def dense_doubles_excitation(self, n_qubits: int | None = None, tol: float = 1e-12) -> np.ndarray:
        """Dense doubles excitation operator ``T2`` reconstructed from the pair SVD."""
        n = self.n_orbitals if n_qubits is None else n_qubits
        dim = jw.fock_dim(n)
        T2 = np.zeros((dim, dim), dtype=complex)
        for ch in self.doubles_channels(tol=tol):
            T2 += ch.dense_excitation(n_qubits=n)
        return T2

    def dense_doubles_sigma(self, n_qubits: int | None = None, tol: float = 1e-12) -> np.ndarray:
        """Dense anti-Hermitian doubles generator reconstructed from the pair SVD."""
        n = self.n_orbitals if n_qubits is None else n_qubits
        dim = jw.fock_dim(n)
        sigma = np.zeros((dim, dim), dtype=complex)
        for ch in self.doubles_channels(tol=tol):
            sigma += ch.dense_sigma(n_qubits=n)
        return sigma


def build_cluster_generator(
    NO: int,
    NV: int,
    *,
    t1: np.ndarray | None = None,
    t2: np.ndarray | None = None,
) -> ClusterGenerator:
    """Convenience constructor with zero defaults."""
    if t1 is None:
        t1 = np.zeros((NV, NO), dtype=complex)
    if t2 is None:
        t2 = np.zeros((NV, NV, NO, NO), dtype=complex)
    if t1.shape != (NV, NO):
        raise ValueError(f"t1 shape {t1.shape} != (NV, NO) = ({NV}, {NO})")
    if t2.shape != (NV, NV, NO, NO):
        raise ValueError(f"t2 shape {t2.shape} != (NV, NV, NO, NO)")
    # Antisymmetrize t2 so caller doesn't have to.
    t2 = 0.25 * (
        t2
        - t2.transpose(1, 0, 2, 3)
        - t2.transpose(0, 1, 3, 2)
        + t2.transpose(1, 0, 3, 2)
    )
    return ClusterGenerator(NO=NO, NV=NV, t1=np.asarray(t1, dtype=complex), t2=t2)
