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

This module remains the small-system verification path, but the outer
LCU scaffold is now represented structurally: ``PREP_H`` is a
synthesized state-preparation gate and ``SELECT_H`` is an explicit
multi-branch compiled multiplexor over child branch circuits. The dense
simulator still verifies the resulting circuit numerically.

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
from ..circuits.gate import Gate, MultiplexedGate, StatePreparationGate
from ..circuits.simulator import unitary as circuit_unitary
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


def _prep_amplitudes(weights_abs: np.ndarray, n_ancilla: int) -> np.ndarray:
    """Return the padded normalized PREP target amplitudes."""
    dim = 2**n_ancilla
    amps = np.zeros(dim, dtype=complex)
    total = float(np.sum(weights_abs))
    if total <= 0:
        raise ValueError("total weight must be positive")
    amps[: weights_abs.shape[0]] = np.sqrt(weights_abs / total)
    return amps


def _dense_subcircuit(name: str, unitary: np.ndarray, *, width: int, kind: str) -> Circuit:
    circuit = Circuit(num_qubits=width)
    circuit.append(Gate(name=name, qubits=tuple(range(width)), matrix=unitary, kind=kind))
    return circuit


def _identity_circuit(width: int, *, kind: str) -> Circuit:
    return _dense_subcircuit(
        name=kind,
        unitary=np.eye(2**width, dtype=complex),
        width=width,
        kind=kind,
    )


def _bilinear_subencoding(phi: np.ndarray) -> tuple[Circuit, float]:
    """Return the Lemma-1 branch circuit for
    ``a^dag[phi] a[phi]`` (``u = v = phi``, ``coeff = 1``) and the
    sub-encoding normalization (``alpha = 1``).
    """
    L = BilinearRankOne(u=phi, v=phi, coeff=1.0)
    be = build_bilinear_block_encoding(L)
    return be.circuit, be.alpha


def _cholesky_subencoding(L_mu: np.ndarray, *, n_system: int) -> tuple[Circuit, float]:
    """Return the Lemma-2 branch circuit for ``O_mu^2``.

    Returns
    -------
    W_full : circuit for the block encoding of ``O_mu^2 / alpha_O^2``.
    alpha_O : operator norm of ``O_mu`` on the Fock space.
    """
    W, alpha = hermitian_one_body_block_encoding(L_mu)
    return _dense_subcircuit(
        name="W_cholesky_squared",
        unitary=x_squared_qsvt_unitary(W),
        width=n_system + 1,
        kind="W_cholesky_squared",
    ), alpha


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

    branch_circuits: list[Circuit] = []
    branch_phases: list[complex] = []
    weights: list[float] = []
    one_body_branch_count = 0
    cholesky_branch_count = 0

    # (1) One-body eigenchannels: weight |e_k|, sign sgn(e_k).
    for ch in one_body_channels:
        if abs(ch.coeff) < 1e-14:
            continue
        branch_circuit, _alpha = _bilinear_subencoding(ch.phi)
        branch_circuits.append(branch_circuit)
        branch_phases.append(1.0 + 0.0j if ch.coeff >= 0 else -1.0 + 0.0j)
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
        branch_circuit, alpha_O = _cholesky_subencoding(H_mu, n_system=n)
        branch_circuits.append(branch_circuit)
        branch_phases.append(1.0 + 0.0j)
        weights.append(0.5 * alpha_O * alpha_O)
        cholesky_branch_count += 1

    weights_arr = np.array(weights, dtype=float)

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

    # Total ancilla = sub-ancilla (1) + selector (a_sel).
    n_anc_total = a_sel + 1
    dim_sys = 2**n
    prep_amplitudes = _prep_amplitudes(prep_weights, n_ancilla=a_sel)

    selector_qubits = tuple(range(n + 1, n + 1 + a_sel))
    full_qubits = tuple(range(n + n_anc_total))
    circuit = Circuit(num_qubits=n + n_anc_total)
    circuit.append(
        StatePreparationGate(
            name="PREP_H",
            qubits=selector_qubits,
            amplitudes=prep_amplitudes,
            kind="PREP_H",
        )
    )
    circuit.append(
        MultiplexedGate(
            name="SELECT_H",
            qubits=full_qubits,
            selector_width=a_sel,
            branch_circuits=tuple(branch_circuits) + (_identity_circuit(n + 1, kind="NULL_H_branch"),),
            default_circuit=_identity_circuit(n + 1, kind="PADDED_H_branch"),
            branch_phases=tuple(branch_phases) + (1.0 + 0.0j,),
            kind="SELECT_H",
        )
    )
    circuit.append(
        StatePreparationGate(
            name="PREP_H^dag",
            qubits=selector_qubits,
            amplitudes=prep_amplitudes,
            kind="PREP_H",
            adjoint=True,
        )
    )
    W_full = circuit_unitary(circuit)
    top_left = W_full[:dim_sys, :dim_sys]
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
        weights=weights_arr * np.array([phase.real for phase in branch_phases], dtype=float),
        selector_width=a_sel,
        null_branch_index=null_branch_index,
        circuit=circuit,
        resources=resources,
    )
