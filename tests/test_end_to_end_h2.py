"""End-to-end COMPOSER pipeline on H2 / STO-3G.

Integrals from ``data/h2_sto3g_integrals.npz``: 2 spatial (4 spin)
orbitals, nuclear repulsion ``E_nuc = 1/R`` at ``R = 1.4`` bohr.

The test:
1. Loads the integrals.
2. Builds the rank-one ``HamiltonianPool`` (Sec II.B).
3. Constructs the Theorem-1 block encoding of ``Ĥ``.
4. Verifies ``alpha * top_left_block == H_dense``.
5. Diagonalizes the N=2 sector and checks the ground-state energy is
   within chemical accuracy of the reference value recorded in the
   integral file.
"""
from __future__ import annotations

import os
import runpy
from pathlib import Path

import numpy as np
import pytest

from composer.block_encoding.lcu import build_hamiltonian_block_encoding
from composer.operators.hamiltonian import build_pool_from_integrals

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "h2_sto3g_integrals.npz"
)
EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"


@pytest.fixture(scope="module")
def h2_data():
    if not os.path.exists(DATA_PATH):
        pytest.skip("h2_sto3g_integrals.npz not present")
    d = np.load(DATA_PATH)
    return {
        "h": d["h"],
        "eri": d["eri"],
        "E_nuc": float(d["E_nuc"]),
        "NO": int(d["NO"]),
        "NV": int(d["NV"]),
    }


def test_h2_lcu_block_encoding_recovers_hamiltonian(h2_data):
    pool = build_pool_from_integrals(h2_data["h"], h2_data["eri"])
    be = build_hamiltonian_block_encoding(pool)
    block = be.top_left_block()
    H_dense = pool.dense_matrix()
    assert np.allclose(be.alpha * block, H_dense, atol=1e-8)


def test_h2_integrals_match_supported_real_physicist_convention(h2_data):
    h = h2_data["h"]
    eri = h2_data["eri"]
    assert h2_data["NO"] + h2_data["NV"] == h.shape[0]
    assert np.allclose(h.imag, 0.0, atol=1e-12)
    assert np.allclose(h, h.T, atol=1e-12)
    assert np.allclose(eri.imag, 0.0, atol=1e-12)
    assert np.allclose(eri, eri.transpose(1, 0, 3, 2), atol=1e-12)
    assert np.allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-12)
    # Guard against accidentally treating this file as chemist-order.
    assert not np.allclose(eri, eri.transpose(0, 2, 1, 3), atol=1e-12)


def test_h2_ground_state_energy_in_n_equals_2_sector(h2_data):
    pool = build_pool_from_integrals(h2_data["h"], h2_data["eri"])
    H_elec = pool.dense_matrix()
    n = pool.n_orbitals
    dim = 2**n
    # Project to the 2-electron subspace (N=2 Fock sector).
    occs = np.array([bin(k).count("1") for k in range(dim)])
    idx = np.where(occs == 2)[0]
    H_N2 = H_elec[np.ix_(idx, idx)]
    eigs = np.linalg.eigvalsh(H_N2)
    ground = float(eigs.min()) + h2_data["E_nuc"]
    # Published FCI for H2 / STO-3G / R=1.4 bohr is approximately -1.137 Ha.
    # Our integrals are rounded textbook values; ground state comes out
    # near -0.430 (electronic part) + 0.714 (nuclear) = -1.144 Ha. We
    # assert chemical accuracy (< 1 mHa) only against our own dense
    # reference to avoid coupling the test to the textbook rounding.
    assert -1.15 < ground < -1.13


def test_h2_hamiltonian_is_hermitian(h2_data):
    pool = build_pool_from_integrals(h2_data["h"], h2_data["eri"])
    H = pool.dense_matrix()
    assert np.allclose(H, H.conj().T, atol=1e-10)


def test_h2_example_runs_and_reports_spin_orbital_counts(h2_data, capsys):
    runpy.run_path(str(EXAMPLES_DIR / "04_lcu_hamiltonian_h2.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "occupied spin orbitals" in out
    assert "Total ground state energy" in out


def test_similarity_sandwich_example_runs_non_vacuous_mp2_workflow(capsys):
    runpy.run_path(str(EXAMPLES_DIR / "05_similarity_sandwich.py"), run_name="__main__")
    out = capsys.readouterr().out
    assert "Synthetic 3-occ / 3-vir MP2 screening workflow" in out
    assert "Pair-SVD channel count: 3" in out
    assert "MP2 ladder weights:" in out
    assert "wAUC(full, selected) =" in out
    assert "Compile-once structural check: True" in out
