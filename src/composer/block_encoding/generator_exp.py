"""Generator-side sigma-pool oracle plus oracle/QSP exponentiation.

Paper status
------------

This module now implements the generator-side oracle scaffolding from
Sec. IV.B more literally:

* a fixed-width selector register over the compiled sigma-channel pool,
* a mask-parameterized ``PREP_sigma``,
* a branch-multiplexed ``SELECT_sigma``, and
* an explicit null branch that participates in the compiled PREP data.

The returned ``SigmaOracle`` is the QSP input object for the Hermitian generator

    A = -i σ̂,

so its ancilla-zero system block equals ``A / alpha_bar``.

For ``e^{σ̂}``, the main path now consumes that real oracle instead of
applying a Chebyshev polynomial directly to a dense matrix. Because the
repo currently has only parity-constrained scalar-QSP synthesis, the
implemented exponential is assembled as:

* one QSP sequence for the even real target ``cos(alpha x)``,
* one QSP sequence for the odd real target ``sin(alpha x)``,
* explicit hermitianization of each QSP sequence at the block-encoding
  level, and
* one final LCU that combines those two real oracles into
  ``cos(-iσ) + i sin(-iσ) = e^{σ}``.

This is materially closer to the paper's oracle/QSP construction than
the old dense surrogate, while the dense Chebyshev path is retained as
an optional numerical reference for direct dense-matrix input.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from ..circuits.circuit import Circuit, CircuitResourceSummary
from ..circuits.gate import CircuitCall, Gate, SelectGate
from ..circuits.simulator import unitary as circuit_unitary
from ..factorization.pair_svd import EmbeddedPairChannel
from ..operators.generator import ClusterGenerator, SingleExcitationChannel
from ..operators.mask import ChannelMask
from ..qsp.chebyshev import (
    cos_alpha_x_coefficients,
    jacobi_anger_coefficients,
    recommended_degree,
    recommended_degree_with_parity,
    sin_alpha_x_coefficients,
)
from ..qsp.phases import qsp_phase_gate, solve_phases_real_chebyshev
from .cholesky_channel import hermitian_one_body_block_encoding

__all__ = [
    "SigmaOracleResourceSummary",
    "GeneratorExpResourceSummary",
    "SigmaOracle",
    "GeneratorExpOracle",
    "build_sigma_pool_oracle",
    "build_generator_exp_oracle",
    "dense_masked_generator_sigma",
    "dense_masked_doubles_sigma",
    "hermitian_fock_block_encoding",
    "matrix_chebyshev_eval",
    "dense_generator_exp_reference",
    "generator_exp_top_left_block",
]


@dataclass(frozen=True)
class SigmaOracleResourceSummary:
    """Resource/accounting summary for the compiled sigma-pool oracle."""

    alpha: float
    n_system: int
    n_ancilla: int
    selector_width: int
    active_branch_count: int
    compiled_branch_count: int
    null_branch_index: int
    circuit: CircuitResourceSummary


@dataclass(frozen=True)
class GeneratorExpResourceSummary:
    """Resource/accounting summary for the compiled generator-exp oracle."""

    alpha: float
    n_system: int
    n_ancilla: int
    exp_sign: int
    cos_degree: int
    sin_degree: int
    cos_phase_count: int
    sin_phase_count: int
    cos_qsp_query_count: int
    sin_qsp_query_count: int
    sigma_oracle: SigmaOracleResourceSummary
    cos_qsp_circuit: CircuitResourceSummary
    sin_qsp_circuit: CircuitResourceSummary
    circuit: CircuitResourceSummary
    unitary_circuit: CircuitResourceSummary


@dataclass
class SigmaOracle:
    """Ancilla-resolved PREP-SELECT-PREP† oracle for the sigma pool.

    The oracle encodes the Hermitian QSP input ``A = -i σ̂_pool`` over
    the compiled generator pool: explicit singles channels followed by
    pair-SVD doubles channels.
    """

    W: np.ndarray
    alpha: float
    n_system: int
    n_ancilla: int
    selector_width: int
    circuit: Circuit
    channel_norms: np.ndarray
    active_branch_weights: np.ndarray
    prep_branch_weights: np.ndarray
    prep_amplitudes: np.ndarray
    null_branch_index: int
    ancilla_zero_block_dense: np.ndarray
    sigma_zero_block_dense: np.ndarray
    resources: SigmaOracleResourceSummary

    def top_left_block(self) -> np.ndarray:
        dim_sys = 2**self.n_system
        return self.W[:dim_sys, :dim_sys]


@dataclass
class GeneratorExpOracle:
    """Oracle/QSP/LCU construction approximating ``e^{sign * σ̂_pool}``.

    ``circuit`` is the direct parity-split block encoding whose
    ancilla-zero block equals

        ``exp(sign * sigma_hat) / 2``.

    The factor ``2`` comes only from the final ``cos +/- i sin`` LCU.
    ``unitary_circuit`` applies one round of oblivious amplitude
    amplification to that block encoding, yielding the paper-facing
    ``U_sigma`` whose ancilla-zero block approximates

        ``exp(sign * sigma_hat)``.
    """

    alpha: float
    n_system: int
    n_ancilla: int
    circuit: Circuit
    sigma_oracle: SigmaOracle
    wx_oracle_circuit: Circuit
    cos_qsp_circuit: Circuit
    sin_qsp_circuit: Circuit
    cos_oracle_circuit: Circuit
    sin_oracle_circuit: Circuit
    unitary_circuit: Circuit
    cos_phases: np.ndarray
    sin_phases: np.ndarray
    cos_degree: int
    sin_degree: int
    exp_sign: int
    resources: GeneratorExpResourceSummary
    _W_dense: np.ndarray | None = field(default=None, init=False, repr=False)
    _exp_zero_block_dense: np.ndarray | None = field(default=None, init=False, repr=False)
    _cos_zero_block_dense: np.ndarray | None = field(default=None, init=False, repr=False)
    _sin_zero_block_dense: np.ndarray | None = field(default=None, init=False, repr=False)
    _unitary_zero_block_dense: np.ndarray | None = field(default=None, init=False, repr=False)

    def top_left_block(self) -> np.ndarray:
        dim_sys = 2**self.n_system
        return self.W[:dim_sys, :dim_sys]

    @property
    def W(self) -> np.ndarray:
        if self._W_dense is None:
            self._W_dense = circuit_unitary(self.circuit)
        return self._W_dense

    @property
    def ancilla_zero_block_dense(self) -> np.ndarray:
        return 0.5 * self.exp_zero_block_dense

    @property
    def exp_zero_block_dense(self) -> np.ndarray:
        if self._exp_zero_block_dense is None:
            dim_sys = 2**self.n_system
            self._exp_zero_block_dense = 2.0 * self.W[:dim_sys, :dim_sys]
        return self._exp_zero_block_dense

    @property
    def cos_zero_block_dense(self) -> np.ndarray:
        if self._cos_zero_block_dense is None:
            dim_sys = 2**self.n_system
            cos_u = circuit_unitary(self.cos_oracle_circuit)
            self._cos_zero_block_dense = cos_u[:dim_sys, :dim_sys]
        return self._cos_zero_block_dense

    @property
    def sin_zero_block_dense(self) -> np.ndarray:
        if self._sin_zero_block_dense is None:
            dim_sys = 2**self.n_system
            sin_u = circuit_unitary(self.sin_oracle_circuit)
            self._sin_zero_block_dense = sin_u[:dim_sys, :dim_sys]
        return self._sin_zero_block_dense

    @property
    def unitary_zero_block_dense(self) -> np.ndarray:
        if self._unitary_zero_block_dense is None:
            dim_sys = 2**self.n_system
            u_sigma = circuit_unitary(self.unitary_circuit)
            self._unitary_zero_block_dense = u_sigma[:dim_sys, :dim_sys]
        return self._unitary_zero_block_dense


def _prep_unitary(weights_abs: np.ndarray, n_ancilla: int) -> np.ndarray:
    """Dense selector PREP with column-0 equal to the target amplitudes."""
    dim = 2**n_ancilla
    amps = np.zeros(dim, dtype=complex)
    total = float(np.sum(weights_abs))
    if total <= 0:
        raise ValueError("total PREP weight must be positive")
    amps[: weights_abs.shape[0]] = np.sqrt(weights_abs / total)
    M = np.zeros((dim, dim), dtype=complex)
    M[:, 0] = amps
    for j in range(1, dim):
        e = np.zeros(dim, dtype=complex)
        e[j] = 1.0
        for k in range(j):
            e = e - np.vdot(M[:, k], e) * M[:, k]
        nrm = np.linalg.norm(e)
        if nrm < 1e-12:
            for j2 in range(dim):
                cand = np.zeros(dim, dtype=complex)
                cand[j2] = 1.0
                for k in range(j):
                    cand = cand - np.vdot(M[:, k], cand) * M[:, k]
                cand_nrm = np.linalg.norm(cand)
                if cand_nrm > 1e-6:
                    e = cand / cand_nrm
                    break
        else:
            e = e / nrm
        M[:, j] = e
    return M


def _zero_block_subencoding(n_system: int) -> np.ndarray:
    """One-ancilla block encoding of the zero operator.

    The paper describes a mask-residual null branch. To preserve the
    encoded operator while still occupying selector amplitude, the null
    branch must have zero ancilla-zero block; ``X`` on the signal ancilla
    is the smallest exact choice.
    """
    dim_sys = 2**n_system
    return np.kron(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), np.eye(dim_sys, dtype=complex))


def _select_unitary(sub_encodings: list[np.ndarray], n_ancilla: int, n_system: int) -> np.ndarray:
    """Block-diagonal SELECT over the selector register."""
    a_sel = n_ancilla - 1
    sub_dim = 2 ** (n_system + 1)
    sel_dim = 2**a_sel
    full_dim = sel_dim * sub_dim
    default_block = _zero_block_subencoding(n_system)
    S = np.zeros((full_dim, full_dim), dtype=complex)
    for s in range(sel_dim):
        block = sub_encodings[s] if s < len(sub_encodings) else default_block
        start = s * sub_dim
        stop = start + sub_dim
        S[start:stop, start:stop] = block
    return S


def _single_channel_hermitian_subencoding(
    channel: SingleExcitationChannel,
    n_system: int,
) -> tuple[np.ndarray, float]:
    """Full-Fock block encoding of a Hermitian one-body singles generator term."""
    return hermitian_one_body_block_encoding(
        channel.hermitian_generator_matrix(),
        n_qubits=n_system,
    )


def _channel_hermitian_subencoding(
    channel: SingleExcitationChannel | EmbeddedPairChannel,
    n_system: int,
) -> tuple[np.ndarray, float]:
    """Exact dense block encoding of ``A_s = -i(L_s - L_s^dag)``."""
    if isinstance(channel, SingleExcitationChannel):
        return _single_channel_hermitian_subencoding(channel, n_system=n_system)
    A_s = -1j * channel.dense_sigma(n_qubits=n_system)
    A_s = 0.5 * (A_s + A_s.conj().T)
    return hermitian_fock_block_encoding(A_s)


def dense_masked_generator_sigma(
    generator: ClusterGenerator,
    mask: ChannelMask,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """Dense masked generator ``σ̂_pool(m)`` from the compiled singles+doubles pool."""
    channels = generator.generator_channels(tol=tol)
    if mask.weights.shape != (len(channels),):
        raise ValueError(
            "mask length must match the compiled generator pool: "
            f"got {mask.weights.shape[0]}, expected {len(channels)}"
        )
    n = generator.n_orbitals
    dim = 2**n
    sigma = np.zeros((dim, dim), dtype=complex)
    for w, ch in zip(mask.weights, channels):
        sigma += w * ch.dense_sigma(n_qubits=n)
    return sigma


def dense_masked_doubles_sigma(
    generator: ClusterGenerator,
    mask: ChannelMask,
    *,
    tol: float = 1e-12,
) -> np.ndarray:
    """Backward-compatible alias for the masked generator dense reference."""
    return dense_masked_generator_sigma(generator, mask, tol=tol)


def build_sigma_pool_oracle(
    generator: ClusterGenerator,
    mask: ChannelMask,
    *,
    tol: float = 1e-12,
    alpha_bar: float | None = None,
) -> SigmaOracle:
    """Build the masked sigma-pool oracle of Sec. IV.B, Eq. (39)-(43).

    The oracle is compiled over the fixed generator pool consisting of
    singles channels plus pair-rank-one doubles channels.
    ``mask.weights`` re-dial only the selector PREP amplitudes. The
    ancilla-zero block is the Hermitian generator input

        ``A(m) / alpha_bar = -i σ̂_pool(m) / alpha_bar``.

    If ``alpha_bar`` is omitted, use the paper-style full-pool total
    ``sum_s alpha_s``. This matches selector masks with
    ``0 <= mask.weights[s] <= 1``.
    """
    channels = generator.generator_channels(tol=tol)
    if mask.weights.shape != (len(channels),):
        raise ValueError(
            "mask length must match the compiled generator pool: "
            f"got {mask.weights.shape[0]}, expected {len(channels)}"
        )

    n = generator.n_orbitals
    dim_sys = 2**n
    ell = len(channels)
    a_sel = int(np.ceil(np.log2(ell + 1))) if ell + 1 > 1 else 1
    n_anc = a_sel + 1

    sub_encodings: list[np.ndarray] = []
    channel_norms = np.zeros(ell, dtype=float)
    for idx, channel in enumerate(channels):
        W_s, alpha_s = _channel_hermitian_subencoding(channel, n_system=n)
        sub_encodings.append(W_s)
        channel_norms[idx] = alpha_s

    active_branch_weights = mask.weights * channel_norms
    default_alpha_bar = float(np.sum(channel_norms))
    if ell == 0 and default_alpha_bar <= 0:
        default_alpha_bar = 1.0
    target_alpha = default_alpha_bar if alpha_bar is None else float(alpha_bar)
    active_total = float(np.sum(active_branch_weights))
    if target_alpha + 1e-12 < active_total:
        raise ValueError(f"alpha_bar={target_alpha} too small for compiled sigma weight {active_total}")

    residual = max(target_alpha - active_total, 0.0)
    if mask.null_weight > 1e-12 and not np.isclose(mask.null_weight, residual, atol=1e-10):
        raise ValueError(
            "mask.null_weight is inconsistent with the compiled sigma oracle: "
            f"got {mask.null_weight}, expected {residual}"
        )

    null_branch_index = ell
    prep_branch_weights = np.zeros(ell + 1, dtype=float)
    prep_branch_weights[:ell] = active_branch_weights
    prep_branch_weights[null_branch_index] = residual if ell > 0 or residual > 0 else target_alpha
    prep_amplitudes = np.zeros(2**a_sel, dtype=float)
    if target_alpha > 0:
        prep_amplitudes[: prep_branch_weights.shape[0]] = np.sqrt(prep_branch_weights / target_alpha)

    P = _prep_unitary(prep_branch_weights, n_ancilla=a_sel)
    sub_encodings_with_null = list(sub_encodings)
    sub_encodings_with_null.append(_zero_block_subencoding(n))
    S = _select_unitary(sub_encodings_with_null, n_ancilla=n_anc, n_system=n)

    sub_dim = 2 ** (n + 1)
    P_full = np.kron(P, np.eye(sub_dim, dtype=complex))
    W_full = P_full.conj().T @ S @ P_full
    top_left = W_full[:dim_sys, :dim_sys]

    selector_qubits = tuple(range(n + 1, n + 1 + a_sel))
    full_qubits = tuple(range(n + n_anc))
    circuit = Circuit(num_qubits=n + n_anc)
    circuit.append(Gate(name="PREP_sigma", qubits=selector_qubits, matrix=P, kind="PREP_sigma"))
    circuit.append(
        Gate(
            name="SELECT_sigma",
            qubits=full_qubits,
            matrix=S,
            kind="SELECT_sigma",
        )
    )
    circuit.append(Gate(name="PREP_sigma^dag", qubits=selector_qubits, matrix=P.conj().T, kind="PREP_sigma"))

    sigma_block = 1j * top_left
    resources = SigmaOracleResourceSummary(
        alpha=target_alpha,
        n_system=n,
        n_ancilla=n_anc,
        selector_width=a_sel,
        active_branch_count=ell,
        compiled_branch_count=ell + 1,
        null_branch_index=null_branch_index,
        circuit=circuit.resource_summary(),
    )
    return SigmaOracle(
        W=W_full,
        alpha=target_alpha,
        n_system=n,
        n_ancilla=n_anc,
        selector_width=a_sel,
        circuit=circuit,
        channel_norms=channel_norms,
        active_branch_weights=active_branch_weights,
        prep_branch_weights=prep_branch_weights,
        prep_amplitudes=prep_amplitudes,
        null_branch_index=null_branch_index,
        ancilla_zero_block_dense=top_left,
        sigma_zero_block_dense=sigma_block,
        resources=resources,
    )


def _signal_s_gate() -> np.ndarray:
    """Single-qubit basis change with ``S R(x) S = W(x)``."""
    return np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)


def _wx_oracle_circuit(sigma_oracle: SigmaOracle) -> Circuit:
    """Convert the reflection-style Hermitian block encoding into Wx form."""
    n_qubits = sigma_oracle.n_system + sigma_oracle.n_ancilla
    signal_qubit = sigma_oracle.n_system
    full_qubits = tuple(range(n_qubits))

    circuit = Circuit(num_qubits=n_qubits)
    circuit.append(Gate(name="S_wx", qubits=(signal_qubit,), matrix=_signal_s_gate(), kind="S_wx"))
    circuit.append(
        CircuitCall(
            name="U_sigma_reflection",
            qubits=full_qubits,
            subcircuit=sigma_oracle.circuit,
            kind="U_sigma_reflection",
        )
    )
    circuit.append(Gate(name="S_wx", qubits=(signal_qubit,), matrix=_signal_s_gate(), kind="S_wx"))
    return circuit


def _qsp_sequence_circuit(
    oracle_circuit: Circuit,
    *,
    n_qubits: int,
    signal_qubit: int,
    phases: np.ndarray,
    oracle_kind: str,
) -> Circuit:
    """Build the scalar-QSP sequence ``S(phi0) U S(phi1) ... U S(phid)``."""
    phases = np.asarray(phases, dtype=float).ravel()
    circuit = Circuit(num_qubits=n_qubits)
    for idx, phi in enumerate(phases):
        circuit.append(
            Gate(
                name=f"QSP_phi_{idx}",
                qubits=(signal_qubit,),
                matrix=qsp_phase_gate(float(phi)),
                kind="QSP_phi",
            )
        )
        if idx < len(phases) - 1:
            circuit.append(
                CircuitCall(
                    name=f"{oracle_kind}_{idx}",
                    qubits=tuple(range(n_qubits)),
                    subcircuit=oracle_circuit,
                    kind=oracle_kind,
                )
            )
    return circuit


def _hadamard() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


def _hermitianize_block_encoding(
    circuit_in: Circuit,
    *,
    n_base_qubits: int,
    prefix: str,
) -> Circuit:
    """Return a block encoding of ``Re(<0|unitary|0>)`` via ``(U + U†)/2``."""
    H = _hadamard()

    anc = n_base_qubits
    circuit = Circuit(num_qubits=n_base_qubits + 1)
    circuit.append(Gate(name=f"PREP_{prefix}", qubits=(anc,), matrix=H, kind=f"PREP_{prefix}"))
    circuit.append(
        SelectGate(
            name=f"SELECT_{prefix}",
            qubits=tuple(range(n_base_qubits + 1)),
            zero_circuit=circuit_in,
            one_circuit=circuit_in.inverse(),
            kind=f"SELECT_{prefix}",
        )
    )
    circuit.append(
        Gate(name=f"PREP_{prefix}^dag", qubits=(anc,), matrix=H.conj().T, kind=f"PREP_{prefix}")
    )
    return circuit


def _combine_real_exp_components(
    cos_circuit: Circuit,
    sin_circuit: Circuit,
    *,
    n_base_qubits: int,
    exp_sign: int,
) -> Circuit:
    """Final LCU for ``cos(H) + i sign sin(H)`` with ``H = -i sigma``."""
    if exp_sign not in (-1, 1):
        raise ValueError("exp_sign must be +1 or -1")
    H = _hadamard()
    relative_phase = np.diag([1.0, 1j * exp_sign]).astype(complex)

    anc = n_base_qubits
    circuit = Circuit(num_qubits=n_base_qubits + 1)
    circuit.append(Gate(name="PREP_exp", qubits=(anc,), matrix=H, kind="PREP_exp"))
    circuit.append(Gate(name="PHASE_exp", qubits=(anc,), matrix=relative_phase, kind="PHASE_exp"))
    circuit.append(
        SelectGate(
            name="SELECT_exp",
            qubits=tuple(range(n_base_qubits + 1)),
            zero_circuit=cos_circuit,
            one_circuit=sin_circuit,
            kind="SELECT_exp",
        )
    )
    circuit.append(Gate(name="PREP_exp^dag", qubits=(anc,), matrix=H.conj().T, kind="PREP_exp"))
    return circuit


def _ancilla_zero_reflection(n_system: int, n_ancilla: int) -> np.ndarray:
    """Reflection ``2|0_anc><0_anc| - I`` on the joint ancilla space."""
    if n_ancilla <= 0:
        raise ValueError("ancilla-zero reflection requires at least one ancilla qubit")
    dim_sys = 2**n_system
    dim_total = 2 ** (n_system + n_ancilla)
    diag = -np.ones(dim_total, dtype=complex)
    diag[:dim_sys] = 1.0
    return np.diag(diag)


def _oblivious_amplitude_amplified_unitary(
    exp_block_circuit: Circuit,
    *,
    n_system: int,
) -> Circuit:
    """Lift an ``exp(sigma)/2`` block encoding to a paper-facing ``U_sigma``.

    For an exact block encoding with top-left block ``A = U / 2`` and
    unitary ``U``, one oblivious-amplitude-amplification step

        ``-W R W^dag R W``

    maps the signal singular value ``1/2`` to ``1`` via the cubic
    polynomial ``x -> x (3 - 4 x^2)``, so the ancilla-zero block becomes
    ``U``. We use the same compiled ancilla layout as the input block
    encoding and keep the construction structural by reusing the input
    subcircuit rather than collapsing to a dense system gate.
    """
    n_qubits = exp_block_circuit.num_qubits
    n_ancilla = n_qubits - n_system
    reflection = _ancilla_zero_reflection(n_system, n_ancilla)
    full_qubits = tuple(range(n_qubits))

    circuit = Circuit(num_qubits=n_qubits)
    circuit.append(
        CircuitCall(
            name="U_sigma_raw",
            qubits=full_qubits,
            subcircuit=exp_block_circuit,
            kind="U_sigma_raw",
        )
    )
    circuit.append(
        Gate(
            name="REFLECT_ancilla_zero",
            qubits=full_qubits,
            matrix=reflection,
            kind="REFLECT_ancilla_zero",
        )
    )
    circuit.append(
        CircuitCall(
            name="U_sigma_raw^dag",
            qubits=full_qubits,
            subcircuit=exp_block_circuit.inverse(),
            kind="U_sigma_raw",
        )
    )
    circuit.append(
        Gate(
            name="REFLECT_ancilla_zero",
            qubits=full_qubits,
            matrix=reflection,
            kind="REFLECT_ancilla_zero",
        )
    )
    circuit.append(
        CircuitCall(
            name="U_sigma_raw",
            qubits=full_qubits,
            subcircuit=exp_block_circuit,
            kind="U_sigma_raw",
        )
    )
    circuit.append(
        Gate(
            name="GLOBAL_MINUS",
            qubits=full_qubits,
            matrix=-np.eye(2**n_qubits, dtype=complex),
            kind="GLOBAL_MINUS",
        )
    )
    return circuit


@lru_cache(maxsize=128)
def _trig_qsp_phases(
    kind: str,
    alpha: float,
    degree: int,
    n_grid: int,
    max_iter: int,
) -> tuple[float, ...]:
    """Cached scalar phase synthesis for the real even/odd trigonometric targets."""
    if kind == "cos":
        coeffs = cos_alpha_x_coefficients(alpha, degree)
        parity = 0
    elif kind == "sin":
        coeffs = sin_alpha_x_coefficients(alpha, degree)
        parity = 1
    else:
        raise ValueError("kind must be 'cos' or 'sin'")
    phases, _ = solve_phases_real_chebyshev(
        coeffs,
        parity=parity,
        n_grid=n_grid,
        max_iter=max_iter,
        rng_seed=0,
    )
    return tuple(float(phi) for phi in phases)


def build_generator_exp_oracle(
    generator_or_oracle: ClusterGenerator | SigmaOracle,
    mask: ChannelMask | None = None,
    *,
    tol: float = 1e-12,
    alpha_bar: float | None = None,
    eps: float = 1e-3,
    qsp_n_grid: int | None = None,
    qsp_max_iter: int = 1200,
    exp_sign: int = 1,
) -> GeneratorExpOracle:
    """Build the paper-facing oracle/QSP construction for ``e^{sign * σ̂}``.

    ``generator_or_oracle`` can be either an already compiled
    ``SigmaOracle`` or a ``ClusterGenerator`` plus ``mask``. The direct
    parity-split block encoding retained in ``circuit`` has ancilla-zero
    block ``exp(sign * sigma_hat) / 2``, while ``unitary_circuit`` is
    the amplified paper-facing ``U_sigma`` whose ancilla-zero block
    approximates ``exp(sign * sigma_hat)``.
    """
    if isinstance(generator_or_oracle, SigmaOracle):
        sigma_oracle = generator_or_oracle
    else:
        if mask is None:
            raise ValueError("mask is required when building the sigma oracle from a ClusterGenerator")
        sigma_oracle = build_sigma_pool_oracle(
            generator_or_oracle,
            mask,
            tol=tol,
            alpha_bar=alpha_bar,
        )

    base_qubits = sigma_oracle.n_system + sigma_oracle.n_ancilla
    signal_qubit = sigma_oracle.n_system
    wx_oracle_circuit = _wx_oracle_circuit(sigma_oracle)

    cos_degree = recommended_degree_with_parity(sigma_oracle.alpha, eps / 2.0, parity=0)
    sin_degree = recommended_degree_with_parity(sigma_oracle.alpha, eps / 2.0, parity=1)
    cos_grid = qsp_n_grid or max(81, 8 * cos_degree + 1)
    sin_grid = qsp_n_grid or max(81, 8 * sin_degree + 1)

    cos_phases = np.array(
        _trig_qsp_phases("cos", float(sigma_oracle.alpha), cos_degree, cos_grid, qsp_max_iter),
        dtype=float,
    )
    sin_phases = np.array(
        _trig_qsp_phases("sin", float(sigma_oracle.alpha), sin_degree, sin_grid, qsp_max_iter),
        dtype=float,
    )

    cos_qsp_circuit = _qsp_sequence_circuit(
        wx_oracle_circuit,
        n_qubits=base_qubits,
        signal_qubit=signal_qubit,
        phases=cos_phases,
        oracle_kind="QSP_sigma_cos",
    )
    sin_qsp_circuit = _qsp_sequence_circuit(
        wx_oracle_circuit,
        n_qubits=base_qubits,
        signal_qubit=signal_qubit,
        phases=sin_phases,
        oracle_kind="QSP_sigma_sin",
    )

    cos_oracle_circuit = _hermitianize_block_encoding(
        cos_qsp_circuit,
        n_base_qubits=base_qubits,
        prefix="cos",
    )
    sin_oracle_circuit = _hermitianize_block_encoding(
        sin_qsp_circuit,
        n_base_qubits=base_qubits,
        prefix="sin",
    )
    exp_circuit = _combine_real_exp_components(
        cos_oracle_circuit,
        sin_oracle_circuit,
        n_base_qubits=base_qubits + 1,
        exp_sign=exp_sign,
    )
    unitary_circuit = _oblivious_amplitude_amplified_unitary(
        exp_circuit,
        n_system=sigma_oracle.n_system,
    )

    resources = GeneratorExpResourceSummary(
        alpha=2.0,
        n_system=sigma_oracle.n_system,
        n_ancilla=sigma_oracle.n_ancilla + 2,
        exp_sign=exp_sign,
        cos_degree=cos_degree,
        sin_degree=sin_degree,
        cos_phase_count=len(cos_phases),
        sin_phase_count=len(sin_phases),
        cos_qsp_query_count=sum(g.kind == "QSP_sigma_cos" for g in cos_qsp_circuit.gates),
        sin_qsp_query_count=sum(g.kind == "QSP_sigma_sin" for g in sin_qsp_circuit.gates),
        sigma_oracle=sigma_oracle.resources,
        cos_qsp_circuit=cos_qsp_circuit.resource_summary(),
        sin_qsp_circuit=sin_qsp_circuit.resource_summary(),
        circuit=exp_circuit.resource_summary(),
        unitary_circuit=unitary_circuit.resource_summary(),
    )
    return GeneratorExpOracle(
        alpha=2.0,
        n_system=sigma_oracle.n_system,
        n_ancilla=sigma_oracle.n_ancilla + 2,
        circuit=exp_circuit,
        sigma_oracle=sigma_oracle,
        wx_oracle_circuit=wx_oracle_circuit,
        cos_qsp_circuit=cos_qsp_circuit,
        sin_qsp_circuit=sin_qsp_circuit,
        cos_oracle_circuit=cos_oracle_circuit,
        sin_oracle_circuit=sin_oracle_circuit,
        unitary_circuit=unitary_circuit,
        cos_phases=cos_phases,
        sin_phases=sin_phases,
        cos_degree=cos_degree,
        sin_degree=sin_degree,
        exp_sign=exp_sign,
        resources=resources,
    )


def hermitian_fock_block_encoding(H: np.ndarray) -> tuple[np.ndarray, float]:
    """Reflection block encoding helper for a Hermitian Fock-space operator."""
    H = np.asarray(H, dtype=complex)
    if not np.allclose(H, H.conj().T, atol=1e-10):
        raise ValueError("H must be Hermitian")
    eigvals, eigvecs = np.linalg.eigh(H)
    alpha = float(max(abs(eigvals.min()), abs(eigvals.max()), 1e-16))
    A = H / alpha
    rad = np.sqrt(np.clip(1.0 - (eigvals / alpha) ** 2, 0.0, 1.0))
    S = (eigvecs * rad) @ eigvecs.conj().T
    S = 0.5 * (S + S.conj().T)
    dim = H.shape[0]
    W = np.zeros((2 * dim, 2 * dim), dtype=complex)
    W[:dim, :dim] = A
    W[:dim, dim:] = S
    W[dim:, :dim] = S
    W[dim:, dim:] = -A
    return W, alpha


def matrix_chebyshev_eval(coeffs: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Evaluate ``sum_k coeffs[k] T_k(A)`` via the Clenshaw recurrence."""
    n = A.shape[0]
    I = np.eye(n, dtype=complex)
    bk_plus_2 = np.zeros_like(A)
    bk_plus_1 = np.zeros_like(A)
    for k in range(len(coeffs) - 1, 0, -1):
        bk = 2.0 * A @ bk_plus_1 - bk_plus_2 + coeffs[k] * I
        bk_plus_2 = bk_plus_1
        bk_plus_1 = bk
    return A @ bk_plus_1 - bk_plus_2 + coeffs[0] * I


def dense_generator_exp_reference(sigma_hat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Dense Chebyshev reference for ``e^{σ̂}``.

    This bypasses the oracle/QSP construction and should therefore be
    used only as a numerical reference when a dense anti-Hermitian
    matrix is already available.
    """
    sigma_hat = np.asarray(sigma_hat, dtype=complex)
    if not np.allclose(sigma_hat, -sigma_hat.conj().T, atol=1e-10):
        raise ValueError("sigma_hat must be anti-Hermitian")
    H = -1j * sigma_hat
    H = 0.5 * (H + H.conj().T)
    eigvals = np.linalg.eigvalsh(H)
    kappa = float(max(abs(eigvals.min()), abs(eigvals.max()), 1e-16))
    A = H / kappa
    degree = recommended_degree(kappa, eps)
    coeffs = jacobi_anger_coefficients(-kappa, degree)
    return matrix_chebyshev_eval(coeffs, A)


def generator_exp_top_left_block(
    sigma_or_matrix: SigmaOracle | np.ndarray,
    eps: float = 1e-3,
    **kwargs,
) -> np.ndarray:
    """Return the dense system block for ``e^{σ̂}``.

    * ``SigmaOracle`` input: build and evaluate the oracle/QSP/LCU path.
    * dense matrix input: use the dense Chebyshev reference path.
    """
    if isinstance(sigma_or_matrix, SigmaOracle):
        return build_generator_exp_oracle(sigma_or_matrix, eps=eps, **kwargs).exp_zero_block_dense
    return dense_generator_exp_reference(np.asarray(sigma_or_matrix, dtype=complex), eps=eps)
