"""Generate ``h2_sto3g_integrals.npz``.

H2 / STO-3G at R = 1.4 bohr (canonical). Spin-orbital ordering
``[1s_alpha, 1s_beta, 2s_alpha, 2s_beta]`` (i.e., alpha first, then
beta, for each spatial). Values are the standard textbook integrals.
"""
from __future__ import annotations

import numpy as np

# Spatial MO-basis integrals (2 orbitals).
# h_pq = <p| h_core |q>
h_spatial = np.array(
    [
        [-1.2563, 0.0],
        [0.0, -0.4719],
    ]
)

# chemist (pq|rs) = \int phi_p(1) phi_q(1) (1/r12) phi_r(2) phi_s(2)
eri_chem_spatial = np.zeros((2, 2, 2, 2))
eri_chem_spatial[0, 0, 0, 0] = 0.6746
eri_chem_spatial[1, 1, 1, 1] = 0.6972
eri_chem_spatial[0, 0, 1, 1] = 0.6637
eri_chem_spatial[1, 1, 0, 0] = 0.6637
# (01|01) = (10|10) = (01|10) = (10|01) by real 8-fold symmetry
eri_chem_spatial[0, 1, 0, 1] = 0.1813
eri_chem_spatial[1, 0, 1, 0] = 0.1813
eri_chem_spatial[0, 1, 1, 0] = 0.1813
eri_chem_spatial[1, 0, 0, 1] = 0.1813

E_nuc = 1.0 / 1.4  # 0.7143 Ha


def _to_spin_orbital(h, eri_chem):
    """Promote spatial integrals to spin-orbital, alpha-then-beta per orb.

    Spin-orbital layout: for 2 spatial (indices P=0,1), the spin orbitals
    are indexed p = 2*P + s with s = 0 (alpha), 1 (beta). So orbs are:
    0 = 0alpha, 1 = 0beta, 2 = 1alpha, 3 = 1beta.
    """
    n_sp = h.shape[0]
    n = 2 * n_sp
    h_so = np.zeros((n, n))
    for P in range(n_sp):
        for Q in range(n_sp):
            for s in (0, 1):
                p = 2 * P + s
                q = 2 * Q + s
                h_so[p, q] = h[P, Q]
    eri_chem_so = np.zeros((n, n, n, n))
    for P in range(n_sp):
        for Q in range(n_sp):
            for R in range(n_sp):
                for S in range(n_sp):
                    val = eri_chem[P, Q, R, S]
                    if abs(val) < 1e-15:
                        continue
                    for s1 in (0, 1):
                        for s2 in (0, 1):
                            p = 2 * P + s1
                            q = 2 * Q + s1
                            r = 2 * R + s2
                            s_ = 2 * S + s2
                            eri_chem_so[p, q, r, s_] = val
    return h_so, eri_chem_so


if __name__ == "__main__":
    h_so, eri_chem_so = _to_spin_orbital(h_spatial, eri_chem_spatial)
    # Convert chemist (pq|rs) to physicist <pq|rs> = (pr|qs)
    eri_phys_so = eri_chem_so.transpose(0, 2, 1, 3)
    np.savez(
        "h2_sto3g_integrals.npz",
        h=h_so,
        eri=eri_phys_so,
        E_nuc=np.array(E_nuc),
        NO=np.array(2),
        NV=np.array(2),
    )
    print("wrote h2_sto3g_integrals.npz")
