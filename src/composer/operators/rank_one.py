"""Rank-one operator primitives (Sec II.A of the paper).

Implements the three rank-one families of Peng, Liu, Kowalski (2026):

* ``BilinearRankOne`` (Def 1, Eq 1-3):
      L = lambda * a^dag[u] a[v]
  with ``a^dag[u] = sum_p u_p a_p^dag`` and ``a[v] = sum_p v_p^* a_p``.
  ``u`` and ``v`` are unit vectors in C^n; ``lambda`` is a real scalar
  absorbing the overall phase. (Per ASSUMPTION #4 we take
  ``lambda >= 0`` so alpha = |lambda| = lambda.)

* ``PairRankOne`` (Def 2, Eq 4-5):
      L = lambda * B^dag[U] B[V]
  where ``U`` and ``V`` are antisymmetric pair tensors on not
  necessarily identical orbital subspaces, and
  ``B^dag[U] = sum_{p<q} U_{pq} a_p^dag a_q^dag``.

* ``ProjectedQuadraticRankOne`` (Def 3, Eq 8-9):
      L = O O^dag with O = sum_r C_r n[u^(r)]
  and ``n[u] = a^dag[u] a[u]`` the occupation operator of a rotated
  single-particle mode. This is the Lemma-2-target form appearing in
  Eq. (13)-(15) of the paper.

These classes are *data holders*: they store the coefficient vectors /
matrices that define the operator, plus a ``dense_matrix(n_qubits)``
method that materializes the second-quantized matrix via the
Jordan-Wigner helpers. They are **not** the block-encoding circuits;
those live in ``src/composer/block_encoding/``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..utils import fermion as jw
from ..utils.antisymmetric import pairs_from_matrix

__all__ = [
    "BilinearRankOne",
    "PairRankOne",
    "ProjectedQuadraticRankOne",
]


# ------------------------------------------------------------------- Def 1


@dataclass
class BilinearRankOne:
    """L = lambda * a^dag[u] a[v]   (Def 1, Eq 1-3).

    Parameters
    ----------
    u, v : (n,) complex arrays
        Coefficient vectors. Must have unit L2 norm
        (``|u| = |v| = 1``); any overall phase is absorbed into lambda.
    coeff : float
        Real non-negative magnitude lambda.
    """

    u: np.ndarray
    v: np.ndarray
    coeff: float

    def __post_init__(self) -> None:
        self.u = np.asarray(self.u, dtype=complex)
        self.v = np.asarray(self.v, dtype=complex)
        if self.u.ndim != 1 or self.v.ndim != 1 or self.u.shape != self.v.shape:
            raise ValueError("u and v must be 1-D arrays of the same length")
        if not np.isclose(np.linalg.norm(self.u), 1.0, atol=1e-10):
            raise ValueError(f"u not unit norm: {np.linalg.norm(self.u)}")
        if not np.isclose(np.linalg.norm(self.v), 1.0, atol=1e-10):
            raise ValueError(f"v not unit norm: {np.linalg.norm(self.v)}")
        if self.coeff < -1e-15:
            raise ValueError(f"coeff must be non-negative (we absorb phases), got {self.coeff}")

    @property
    def n_orbitals(self) -> int:
        return int(self.u.shape[0])

    def dense_matrix(self, n_qubits: int | None = None) -> np.ndarray:
        """Return L as a dense 2**n x 2**n matrix via Jordan-Wigner.

        ``L = lambda * sum_{p,q} u_p v_q^* a_p^dag a_q``.
        """
        n = self.n_orbitals if n_qubits is None else n_qubits
        if n < self.n_orbitals:
            raise ValueError(f"n_qubits={n} < n_orbitals={self.n_orbitals}")
        h_pq = self.coeff * np.outer(self.u, self.v.conj())
        # Embed into n x n if n > n_orbitals by zero-padding
        if n > self.n_orbitals:
            pad = np.zeros((n, n), dtype=complex)
            pad[: self.n_orbitals, : self.n_orbitals] = h_pq
            h_pq = pad
        return jw.one_body_matrix(h_pq)

    @classmethod
    def from_outer(cls, h_pq: np.ndarray) -> "BilinearRankOne":
        """Build from an arbitrary (n, n) coefficient matrix h_pq by SVD
        on a rank-one input; useful for unit tests but the caller is
        expected to supply an already-rank-one h_pq.
        """
        u_s, s_s, vh_s = np.linalg.svd(h_pq)
        if s_s[0] < 1e-14:
            raise ValueError("input matrix is zero")
        # Enforce rank-one assumption
        if s_s.size > 1 and s_s[1] > 1e-10 * s_s[0]:
            raise ValueError(f"h_pq is not rank-one (singular values {s_s})")
        u = u_s[:, 0]
        v = vh_s[0].conj()
        lam = float(s_s[0])
        return cls(u=u, v=v, coeff=lam)


# ------------------------------------------------------------------- Def 2


@dataclass
class PairRankOne:
    """L = lambda * B^dag[U] B[V]   (Def 2, Eq 4-5).

    Parameters
    ----------
    U, V : complex arrays
        Antisymmetric pair coefficient matrices. ``U`` and ``V`` may
        live on different orbital subspaces; their local row/column
        indices are mapped to global spin-orbital labels by
        ``creation_orbitals`` and ``annihilation_orbitals``.
        We normalize on the strict upper triangle, i.e.
        ``sum_{p<q} |U_{pq}|^2 = sum_{r<s} |V_{rs}|^2 = 1``.
    coeff : complex
        Scalar prefactor ``lambda``.
    creation_orbitals, annihilation_orbitals : sequence[int] or None
        Global spin-orbital labels for the local indices of ``U`` and
        ``V``. If both are omitted and ``U`` and ``V`` have the same
        shape, both default to ``range(n)`` for backward compatibility.
        If their shapes differ, the embedding is ambiguous and both
        label lists must be supplied explicitly.
    """

    U: np.ndarray
    V: np.ndarray
    coeff: complex
    creation_orbitals: tuple[int, ...] | None = None
    annihilation_orbitals: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        self.U = np.asarray(self.U, dtype=complex)
        self.V = np.asarray(self.V, dtype=complex)
        for name, M in [("U", self.U), ("V", self.V)]:
            if M.ndim != 2 or M.shape[0] != M.shape[1]:
                raise ValueError(f"{name} must be square 2-D")
            if not np.allclose(M, -M.T, atol=1e-10):
                raise ValueError(f"{name} must be antisymmetric")
        for name, M in [("U", self.U), ("V", self.V)]:
            vec = pairs_from_matrix(M)
            if not np.isclose(np.linalg.norm(vec), 1.0, atol=1e-10):
                raise ValueError(f"{name} pair vector not unit norm ({np.linalg.norm(vec)})")
        if self.creation_orbitals is None and self.annihilation_orbitals is None:
            if self.U.shape[0] != self.V.shape[0]:
                raise ValueError(
                    "creation_orbitals and annihilation_orbitals are required when U and V "
                    "have different dimensions"
                )
            self.creation_orbitals = tuple(range(self.U.shape[0]))
            self.annihilation_orbitals = tuple(range(self.V.shape[0]))
        elif self.creation_orbitals is None:
            self.creation_orbitals = tuple(range(self.U.shape[0]))
        elif self.annihilation_orbitals is None:
            self.annihilation_orbitals = tuple(range(self.V.shape[0]))

        self.creation_orbitals = tuple(int(p) for p in self.creation_orbitals)
        self.annihilation_orbitals = tuple(int(p) for p in self.annihilation_orbitals)
        if len(self.creation_orbitals) != self.U.shape[0]:
            raise ValueError("creation_orbitals length must match U.shape[0]")
        if len(self.annihilation_orbitals) != self.V.shape[0]:
            raise ValueError("annihilation_orbitals length must match V.shape[0]")
        if len(set(self.creation_orbitals)) != len(self.creation_orbitals):
            raise ValueError("creation_orbitals must be distinct")
        if len(set(self.annihilation_orbitals)) != len(self.annihilation_orbitals):
            raise ValueError("annihilation_orbitals must be distinct")
        if any(p < 0 for p in self.creation_orbitals + self.annihilation_orbitals):
            raise ValueError("orbital labels must be non-negative")

    @property
    def n_orbitals(self) -> int:
        highest = max(self.creation_orbitals + self.annihilation_orbitals, default=-1)
        return highest + 1

    def coefficient_tensor(self) -> np.ndarray:
        """Return the local Eq. (5) coefficient tensor ``lambda U_ab V_ij^*``."""
        return self.coeff * np.einsum("ab,ij->abij", self.U, self.V.conj())

    def adjoint(self) -> "PairRankOne":
        """Return ``L^dag = lambda^* B^dag[V] B[U]`` with swapped embeddings."""
        return PairRankOne(
            U=self.V,
            V=self.U,
            coeff=self.coeff.conjugate(),
            creation_orbitals=self.annihilation_orbitals,
            annihilation_orbitals=self.creation_orbitals,
        )

    def dense_matrix(self, n_qubits: int | None = None) -> np.ndarray:
        """Return ``lambda * B^dag[U] B[V]`` as a dense matrix.

        ``B^dag[U] = sum_{p<q} U_{pq} a_{c_p}^dag a_{c_q}^dag`` and
        ``B[V] = sum_{r<s} V_{rs}^* a_{d_s} a_{d_r}``, where
        ``c_*`` and ``d_*`` are the global orbital labels given by
        ``creation_orbitals`` and ``annihilation_orbitals``.
        """
        required = self.n_orbitals
        n = required if n_qubits is None else n_qubits
        if n < required:
            raise ValueError(f"n_qubits={n} < required orbital label span {required}")
        active_orbitals = sorted(set(self.creation_orbitals + self.annihilation_orbitals))
        adag = {p: jw.jw_a_dagger(p, n) for p in active_orbitals}
        a = {p: jw.jw_a(p, n) for p in active_orbitals}
        dim = jw.fock_dim(n)
        B_dag = np.zeros((dim, dim), dtype=complex)
        B = np.zeros((dim, dim), dtype=complex)
        for p in range(self.U.shape[0]):
            for q in range(p + 1, self.U.shape[0]):
                cp = self.creation_orbitals[p]
                cq = self.creation_orbitals[q]
                B_dag += self.U[p, q] * (adag[cp] @ adag[cq])
        for r in range(self.V.shape[0]):
            for s in range(r + 1, self.V.shape[0]):
                dr = self.annihilation_orbitals[r]
                ds = self.annihilation_orbitals[s]
                B += self.V[r, s].conj() * (a[ds] @ a[dr])
        return self.coeff * (B_dag @ B)

    def dense_sigma_term(self, n_qubits: int | None = None) -> np.ndarray:
        """Return the anti-Hermitian ladder ``L - L^dag`` for this channel."""
        L = self.dense_matrix(n_qubits=n_qubits)
        return L - self.adjoint().dense_matrix(n_qubits=n_qubits)


# ------------------------------------------------------------------- Def 3


@dataclass
class ProjectedQuadraticRankOne:
    """L = gamma * O O^dag   (Def 3, Eq 8-9).

    ``O = sum_r C_r n[u^(r)]`` with ``n[u] = a^dag[u] a[u]`` the
    occupation operator of a rotated single-particle mode. ``L`` is
    manifestly positive semidefinite and number-conserving.

    Parameters
    ----------
    orbitals : (R, n) or (n,) complex array
        Row ``r`` is the mode vector ``u^(r)``. Each row must have unit
        Euclidean norm.
    weights : (R,) complex array
        Coefficients ``C_r`` in ``O = sum_r C_r n[u^(r)]``.
    coeff : complex, default 1.0
        Optional overall prefactor ``gamma``.
    """

    orbitals: np.ndarray
    weights: np.ndarray
    coeff: complex = 1.0

    def __post_init__(self) -> None:
        self.orbitals = np.asarray(self.orbitals, dtype=complex)
        if self.orbitals.ndim == 1:
            self.orbitals = self.orbitals.reshape(1, -1)
        self.weights = np.asarray(self.weights, dtype=complex)
        if self.weights.ndim == 0:
            self.weights = self.weights.reshape(1)
        if self.orbitals.ndim != 2:
            raise ValueError(f"orbitals must be 1-D or 2-D, got shape {self.orbitals.shape}")
        if self.weights.ndim != 1:
            raise ValueError(f"weights must be 1-D, got shape {self.weights.shape}")
        if self.orbitals.shape[0] != self.weights.shape[0]:
            raise ValueError(
                "weights length must match the number of mode vectors in orbitals"
            )
        for r, u_r in enumerate(self.orbitals):
            if not np.isclose(np.linalg.norm(u_r), 1.0, atol=1e-10):
                raise ValueError(f"orbitals[{r}] not unit norm")

    @property
    def n_orbitals(self) -> int:
        return int(self.orbitals.shape[1])

    def dense_matrix(self, n_qubits: int | None = None) -> np.ndarray:
        """Return ``gamma * O O^dag`` as a dense matrix.

        Here ``O = sum_r C_r n[u^(r)]`` and ``n[u] = a^dag[u] a[u]``.
        """
        n = self.n_orbitals if n_qubits is None else n_qubits
        if n < self.n_orbitals:
            raise ValueError(f"n_qubits={n} < n_orbitals={self.n_orbitals}")
        dim = jw.fock_dim(n)
        O = np.zeros((dim, dim), dtype=complex)
        for c_r, u_r in zip(self.weights, self.orbitals):
            O += c_r * jw.jw_mode_number(u_r, n)
        return self.coeff * (O @ O.conj().T)
