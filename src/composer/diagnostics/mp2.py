"""Second-order Moller-Plesset amplitudes (App E.3, Eq E9).

Given Fock-diagonal orbital energies and the (spin-orbital) two-electron
integrals in physicist notation ``<pq|rs>``, the MP2 doubles amplitude
is

    t^{MP2}_{ab,ij}  =  ( <ij||ab> ) / ( e_i + e_j - e_a - e_b )

where ``<ij||ab> = <ij|ab> - <ij|ba>`` is the antisymmetrized integral.
Indices ``i,j`` occupy and ``a,b`` virtual. The amplitude is the
lowest-order estimate of the CCSD doubles tensor; COMPOSER uses it as a
*surrogate* for the full T2 in App E.3 to rank pair-SVD channels
(``mask_selection.py``) and estimate wAUC (``subspace.py``).

Eq. (E9) in the paper prints an explicit ``1/4`` because it also allows
ordered ``(a, b, i, j)`` sums. The same paragraph states that when one
restricts to ``a < b`` and ``i < j`` that factor is omitted, or
equivalently absorbed into the tensor definition. COMPOSER stores the
canonical antisymmetric doubles tensor consumed by
``pair_svd_decompose`` and by the generator convention

    T2 = (1/4) sum_{abij} t^{ab}_{ij} a_a^dag a_b^dag a_j a_i,

so the returned entries do not include an extra ``1/4`` factor.

We return a ``(NV, NV, NO, NO)`` array antisymmetric in ``(a,b)`` and
``(i,j)``, matching the shape consumed by
``factorization.pair_svd.pair_svd_decompose``.

Conventions:
* ``eps_occ`` has length ``NO``; ``eps_vir`` has length ``NV``.
* ``eri`` has shape ``(n, n, n, n)`` with ``n = NO + NV`` and
  occupied-first ordering (indices ``0..NO-1`` occupied, ``NO..n-1``
  virtual). This matches ``operators/hamiltonian.py`` conventions.

Cross-checked with the canonical MP2 energy formula in
``test_mp2_wauc.py``.
"""
from __future__ import annotations

import numpy as np

__all__ = ["mp2_doubles_amplitudes", "mp2_energy"]


def mp2_doubles_amplitudes(
    eri: np.ndarray,
    eps_occ: np.ndarray,
    eps_vir: np.ndarray,
) -> np.ndarray:
    """Return ``t^{MP2}`` with shape ``(NV, NV, NO, NO)``.

    Parameters
    ----------
    eri : (n, n, n, n) physicist-order integrals, ``n = NO + NV``.
    eps_occ : (NO,) occupied orbital energies.
    eps_vir : (NV,) virtual orbital energies.
    """
    NO = int(eps_occ.shape[0])
    NV = int(eps_vir.shape[0])
    n = NO + NV
    if eri.shape != (n, n, n, n):
        raise ValueError(
            f"eri shape {eri.shape} != ({n}, {n}, {n}, {n}) for NO={NO}, NV={NV}"
        )
    t2 = np.zeros((NV, NV, NO, NO), dtype=complex)
    for a in range(NV):
        pa = NO + a
        for b in range(NV):
            pb = NO + b
            for i in range(NO):
                for j in range(NO):
                    # <ij||ab> = <ij|ab> - <ij|ba>
                    antisym = eri[i, j, pa, pb] - eri[i, j, pb, pa]
                    denom = eps_occ[i] + eps_occ[j] - eps_vir[a] - eps_vir[b]
                    if abs(denom) < 1e-15:
                        raise ValueError(
                            "encountered near-zero MP2 denominator for "
                            f"(a, b, i, j)=({a}, {b}, {i}, {j})"
                        )
                    t2[a, b, i, j] = antisym / denom
    return t2


def mp2_energy(
    eri: np.ndarray,
    eps_occ: np.ndarray,
    eps_vir: np.ndarray,
) -> float:
    """Closed-form MP2 correlation energy for spin-orbital integrals.

    ``E_MP2 = (1/4) sum_{ijab}  |<ij||ab>|^2 / (e_i + e_j - e_a - e_b)``.

    The factor 1/4 accounts for the (ab),(ij) antisymmetry double-count
    when summing over *all* (ordered) index tuples.
    """
    t2 = mp2_doubles_amplitudes(eri, eps_occ, eps_vir)
    NO = int(eps_occ.shape[0])
    NV = int(eps_vir.shape[0])
    total = 0.0
    for a in range(NV):
        pa = NO + a
        for b in range(NV):
            pb = NO + b
            for i in range(NO):
                for j in range(NO):
                    antisym = eri[i, j, pa, pb] - eri[i, j, pb, pa]
                    total += 0.25 * (antisym.conjugate() * t2[a, b, i, j]).real
    return float(total)
