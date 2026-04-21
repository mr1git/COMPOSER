"""Theorem 1 (Eq 35): exact PREP-SELECT-PREP\u2020 block encoding of ``H``.

Given the Hamiltonian rank-one pool

    H  =  sum_k e_k a^dag[phi_k] a[phi_k]  +  (1/2) sum_mu O_mu^2   (Eq 13),

we write ``H`` as a weighted sum of block-encodable rank-one channels
and realize it by PREP-SELECT-PREP\u2020 on a binary selector register
(ASSUMPTION #9):

    alpha_H  =  sum_s |w_s|                                       (Eq 35)
    PREP  |0>  =  sum_s  sqrt(|w_s| / alpha_H)  |s>               (Moettoenen R_y cascade)
    SELECT: controlled-``W_s``
    W  =  PREP\u2020  SELECT  PREP

and the top-left block of ``W`` equals ``H / alpha_H`` restricted to
the system register (ancilla register projected to ``|0>``).

This module is the small-system verification path. Every weight ``w_s``
and every sub-encoding ``W_s`` is realized as a dense matrix; the full
circuit is assembled as a dense unitary and its top-left block is
compared to ``H`` directly in ``tests/test_lcu.py``.

Sub-encodings used here
-----------------------
* For each one-body eigenchannel ``e_k a^dag[phi_k] a[phi_k]`` we
  reuse Lemma 1 (``bilinear.build_bilinear_block_encoding``) with
  ``u = v = phi_k`` and sign carried on the selector (``sgn(e_k)`` is
  applied as a ``Z``-phase on branch ``s``; see the inline sign-convention
  note below).
* For each Cholesky channel ``(1/2) O_mu^2`` we reuse Lemma 2
  (``cholesky_channel.cholesky_channel_block_encoding``); ``(1/2)`` is
  folded into ``w_s`` (non-negative by construction).

Sign convention (ASSUMPTION, inline)
------------------------------------
Negative eigenchannel coefficients are absorbed by applying a single
``Z`` on the selector qubit corresponding to branch ``s`` immediately
before SELECT, which multiplies the branch's contribution by
``sgn(w_s)``. The weight stored in PREP is ``sqrt(|w_s|/alpha)``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from ..circuits.circuit import Circuit, CircuitResourceSummary
from ..circuits.gate import Gate
from ..operators.hamiltonian import HamiltonianPool
from ..operators.rank_one import BilinearRankOne
from .bilinear import build_bilinear_block_encoding
from .cholesky_channel import (
    hermitian_one_body_block_encoding,
    x_squared_qsvt_unitary,
)

__all__ = [
    "LCUResourceSummary",
    "LCUBlockEncoding",
    "build_hamiltonian_block_encoding",
]


@dataclass(frozen=True)
class LCUResourceSummary:
    """Paper-facing resource/accounting summary for Theorem 1."""

    alpha: float
    n_system: int
    n_ancilla: int
    selector_width: int
    subencoding_ancilla: int
    one_body_branch_count: int
    cholesky_branch_count: int
    active_branch_count: int
    compiled_branch_count: int
    null_branch_index: int
    circuit: CircuitResourceSummary


@dataclass
class LCUBlockEncoding:
    """Dense LCU block encoding of a Hamiltonian pool.

    Attributes
    ----------
    W : (2**(n + a) x 2**(n + a)) unitary
        The full PREP-SELECT-PREP\u2020 matrix. Ancilla register occupies
        the ``a`` MSB qubits, system register the ``n`` LSB qubits.
    alpha : float
        Sum of absolute weights (Eq 35); top-left block = H / alpha.
    n_system : int
    n_ancilla : int
    weights : np.ndarray
        Signed weights w_s per branch. ``sum_s |w_s| = alpha``.
    """

    W: np.ndarray
    alpha: float
    n_system: int
    n_ancilla: int
    weights: np.ndarray
    selector_width: int
    null_branch_index: int
    circuit: Circuit
    resources: LCUResourceSummary

    def top_left_block(self) -> np.ndarray:
        """Return the ``2**n x 2**n`` ancilla=|0> block of W."""
        dim_sys = 2**self.n_system
        return self.W[:dim_sys, :dim_sys]


def _prep_unitary(weights_abs: np.ndarray, n_ancilla: int) -> np.ndarray:
    """Return a dense unitary ``P`` on ``n_ancilla`` qubits with
    ``P |0> = sum_s sqrt(w_s / sum_s w_s) |s>``.

    We build the target state-vector, extend to an orthonormal basis via
    QR (so column 0 equals the amplitude vector), and return the
    resulting unitary matrix. This is the small-system stand-in for a
    Moettoenen R_y cascade (ASSUMPTION #9).
    """
    dim = 2**n_ancilla
    amps = np.zeros(dim, dtype=complex)
    total = float(np.sum(weights_abs))
    if total <= 0:
        raise ValueError("total weight must be positive")
    amps[: weights_abs.shape[0]] = np.sqrt(weights_abs / total)
    # QR-extend to unitary whose first column = amps
    M = np.zeros((dim, dim), dtype=complex)
    M[:, 0] = amps
    for j in range(1, dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        for k in range(j):
            e = e - np.vdot(M[:, k], e) * M[:, k]
        nrm = np.linalg.norm(e)
        if nrm < 1e-12:
            # fallback: try another standard basis index
            for j2 in range(dim):
                cand = np.zeros(dim, dtype=complex)
                cand[j2] = 1.0
                for k in range(j):
                    cand = cand - np.vdot(M[:, k], cand) * M[:, k]
                if np.linalg.norm(cand) > 1e-6:
                    e = cand / np.linalg.norm(cand)
                    break
        else:
            e = e / nrm
        M[:, j] = e
    return M


def _select_unitary(
    sub_encodings: Sequence[np.ndarray],
    signs: Sequence[int],
    n_ancilla: int,
    n_system: int,
) -> np.ndarray:
    """Block-diagonal SELECT: branch ``s`` applies ``sgn_s * W_s`` on the
    system register when the selector register is ``|s>``.

    ``sub_encodings[s]`` is the (2^(1+n) x 2^(1+n)) unitary of the
    s-th sub-encoding (bilinear/Lemma-2), where the extra qubit is that
    sub-encoding's private ancilla. Branches with no sub-encoding (null
    branch, ASSUMPTION #10) are identity.

    The combined register has ``n_ancilla_sel + 1 + n_system`` qubits,
    with layout (LSB-first): ``[system (n) | sub-ancilla (1) | selector (a_sel)]``.
    """
    a_sel = n_ancilla - 1
    n = n_system
    sub_dim = 2 ** (n + 1)  # system + sub-ancilla
    sel_dim = 2**a_sel
    full_dim = sel_dim * sub_dim
    S = np.zeros((full_dim, full_dim), dtype=complex)
    for s in range(sel_dim):
        block_start = s * sub_dim
        block_end = block_start + sub_dim
        if s < len(sub_encodings):
            W_s = sub_encodings[s]
            S[block_start:block_end, block_start:block_end] = signs[s] * W_s
        else:
            S[block_start:block_end, block_start:block_end] = np.eye(sub_dim, dtype=complex)
    return S


def _bilinear_subencoding(phi: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the dense (2 * 2^n x 2 * 2^n) Lemma-1 sub-encoding of
    ``a^dag[phi] a[phi]`` (``u = v = phi``, ``coeff = 1``) and the
    sub-encoding normalization (``alpha = 1``).
    """
    L = BilinearRankOne(u=phi, v=phi, coeff=1.0)
    be = build_bilinear_block_encoding(L)
    from ..circuits.simulator import unitary as circuit_unitary

    W = circuit_unitary(be.circuit)
    return W, be.alpha


def _cholesky_subencoding(L_mu: np.ndarray) -> tuple[np.ndarray, float]:
    """Return the dense Lemma-2 sub-encoding of ``O_mu^2``.

    Returns
    -------
    W_full : block encoding of ``O_mu^2 / alpha_O^2`` (2*2^n x 2*2^n).
    alpha_O : operator norm of ``O_mu`` on the Fock space.
    """
    W, alpha = hermitian_one_body_block_encoding(L_mu)
    return x_squared_qsvt_unitary(W), alpha


def build_hamiltonian_block_encoding(pool: HamiltonianPool) -> LCUBlockEncoding:
    """PREP-SELECT-PREP\u2020 block encoding of ``H`` in Theorem 1 form.

    Uses one selector qubit per rank-one branch in the pool plus one
    sub-ancilla (shared across branches in SELECT via block-diagonal
    dispatch). The selector register width
    ``a_sel = ceil(log2(ell_total))`` (ASSUMPTION #9); unused branches
    act as identity (null branch, ASSUMPTION #10).

    Returns an ``LCUBlockEncoding`` whose ``top_left_block()`` satisfies
    ``block * alpha == H`` on the full Fock space.
    """
    n = pool.n_orbitals
    one_body_channels = pool.one_body_eigendecomposition()

    sub_encs: list[np.ndarray] = []
    signs: list[int] = []
    weights: list[float] = []
    one_body_branch_count = 0
    cholesky_branch_count = 0

    # (1) One-body eigenchannels: weight |e_k|, sign sgn(e_k).
    for ch in one_body_channels:
        if abs(ch.coeff) < 1e-14:
            continue
        W, _alpha = _bilinear_subencoding(ch.phi)
        sub_encs.append(W)
        signs.append(1 if ch.coeff >= 0 else -1)
        weights.append(abs(ch.coeff))
        one_body_branch_count += 1

    # (2) Cholesky channels: Lemma 2 gives a sub-encoding whose top-left
    # block equals O_mu^2 / alpha_O^2 exactly. The contribution to H is
    # (1/2) O_mu^2, so the LCU weight is (1/2) * alpha_O^2.
    K = pool.cholesky_factors.shape[0]
    for mu in range(K):
        H_mu, B_mu = pool.hermitian_antihermitian_split(mu)
        # We decompose O_mu = H_mu + i B_mu, so O_mu^2 = H_mu^2 - B_mu^2 + i(H_mu B_mu + B_mu H_mu).
        # For real-integral Hamiltonians B_mu = 0 (L_mu is real symmetric).
        # The small-system test uses real integrals so we verify that path here.
        if not np.allclose(B_mu, 0, atol=1e-10):
            raise NotImplementedError(
                "Non-Hermitian Cholesky factors (complex integrals) require "
                "the extended split from App B.2; test path uses real symmetric L_mu."
            )
        U_mu, alpha_O = _cholesky_subencoding(H_mu)
        sub_encs.append(U_mu)
        signs.append(1)
        weights.append(0.5 * alpha_O * alpha_O)
        cholesky_branch_count += 1

    weights_arr = np.array(weights, dtype=float)
    signs_arr = np.array(signs, dtype=int)

    ell = weights_arr.shape[0]
    # Null branch (ASSUMPTION #10): +1 branch so selector width can be fixed.
    ell_padded = ell + 1
    a_sel = int(np.ceil(np.log2(ell_padded))) if ell_padded > 1 else 1
    null_branch_index = ell

    alpha_total = float(np.sum(weights_arr))
    # PREP amplitudes over ``ell_padded`` branches; the null branch carries
    # the residual so the compiled register width is a_sel.
    prep_weights = np.zeros(2**a_sel, dtype=float)
    prep_weights[:ell] = weights_arr
    # null slot (index ell) left at 0; remaining slots also 0.

    P = _prep_unitary(prep_weights, n_ancilla=a_sel)
    S = _select_unitary(sub_encs, signs_arr, n_ancilla=a_sel + 1, n_system=n)

    # Total ancilla = sub-ancilla (1) + selector (a_sel).
    n_anc_total = a_sel + 1
    dim_sys = 2**n
    sub_dim = 2 ** (n + 1)
    full_dim = 2**a_sel * sub_dim

    # Lift PREP from selector register to full register: P ⊗ I_sub on full.
    P_full = np.kron(P, np.eye(sub_dim, dtype=complex))
    # Lift PREP^dag similarly.
    P_full_dag = P_full.conj().T

    W_full = P_full_dag @ S @ P_full

    # Top-left block: project both sub-ancilla and selector to |0>.
    # In our layout, sub-ancilla is qubit n, selector is qubits n+1..n+a_sel.
    # Ancilla = 0 means index 0 in [selector * sub_dim] and 0 in [sub-ancilla * dim_sys].
    # Full index layout (LSB first): sys (n) | sub-anc (1) | selector (a_sel).
    # So full_index = sel_val * sub_dim + sub_anc_val * dim_sys + sys_val.
    # Ancilla-0 block: sel_val=0, sub_anc_val=0 -> indices [0, dim_sys).
    top_left = W_full[:dim_sys, :dim_sys]

    selector_qubits = tuple(range(n + 1, n + 1 + a_sel))
    full_qubits = tuple(range(n + n_anc_total))
    circuit = Circuit(num_qubits=n + n_anc_total)
    circuit.append(Gate(name="PREP_H", qubits=selector_qubits, matrix=P, kind="PREP_H"))
    circuit.append(Gate(name="SELECT_H", qubits=full_qubits, matrix=S, kind="SELECT_H"))
    circuit.append(Gate(name="PREP_H^dag", qubits=selector_qubits, matrix=P.conj().T, kind="PREP_H"))
    resources = LCUResourceSummary(
        alpha=alpha_total,
        n_system=n,
        n_ancilla=n_anc_total,
        selector_width=a_sel,
        subencoding_ancilla=1,
        one_body_branch_count=one_body_branch_count,
        cholesky_branch_count=cholesky_branch_count,
        active_branch_count=ell,
        compiled_branch_count=ell_padded,
        null_branch_index=null_branch_index,
        circuit=circuit.resource_summary(),
    )

    return LCUBlockEncoding(
        W=W_full,
        alpha=alpha_total,
        n_system=n,
        n_ancilla=n_anc_total,
        weights=weights_arr * signs_arr,
        selector_width=a_sel,
        null_branch_index=null_branch_index,
        circuit=circuit,
        resources=resources,
    )
