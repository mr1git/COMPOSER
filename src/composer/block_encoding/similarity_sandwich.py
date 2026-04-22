r"""Similarity sandwich ``P e^{-sigma} H e^{sigma} P`` (Sec. IV.C, Eq. 47-53).

The paper's oracle is the *unprojected* similarity sandwich

    W_eff^(m) = U_sigma^(m)^\dagger W_H U_sigma^(m),

while the model-space projector ``P^(m)`` is applied only after ancilla
projection when reporting the effective Hamiltonian. This module now
builds that sandwich from the repo's real Hamiltonian oracle
``W_H`` and the real generator-side singles+doubles oracle/QSP
construction used to obtain ``U_sigma``.

Current scope
-------------
The compiled generator oracle now acts on the masked generator pool
consisting of explicit singles channels plus pair-rank-one doubles
channels. The dense reference helper ``effective_hamiltonian_dense(...)``
uses that same masked full-generator semantics for small-system
verification.

The outer returned circuit is now the literal nested compiled-object
composition

    U_sigma(m)^\dagger W_H U_sigma(m),

where ``U_sigma(m)`` is obtained from the compiled generator-exp
block encoding by one round of oblivious amplitude amplification. On the
supported verification-scale systems, its ancilla-zero block is the
paper-facing approximation to ``e^{sigma(m)}``. The reported
``encoded_system_block_dense`` is therefore the exact ancilla-zero
system block of the returned outer circuit, and is claimed to equal
``e^{-sigma(m)} H e^{sigma(m)} / alpha_H`` up to the generator/QSP
approximation error, before the external model-space projector is
applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm

from ..circuits.circuit import Circuit, CircuitResourceSummary
from ..circuits.gate import CircuitCall
from ..circuits.simulator import ancilla_zero_system_block
from ..operators.generator import ClusterGenerator
from ..operators.hamiltonian import HamiltonianPool
from ..operators.mask import ChannelMask, uniform_mask
from .generator_exp import (
    GeneratorExpOracle,
    GeneratorExpResourceSummary,
    build_generator_exp_oracle,
    build_sigma_pool_oracle,
    dense_masked_generator_sigma,
)
from .lcu import LCUBlockEncoding, LCUResourceSummary, build_hamiltonian_block_encoding

__all__ = [
    "ModelSpaceProjector",
    "SimilaritySandwichResourceSummary",
    "SimilaritySandwich",
    "build_similarity_sandwich",
    "effective_hamiltonian_dense",
]


@dataclass
class ModelSpaceProjector:
    """Projector onto a user-supplied list of Slater-determinant indices."""

    determinants: tuple[int, ...]

    def dense_matrix(self, n_qubits: int) -> np.ndarray:
        dim = 2**n_qubits
        P = np.zeros((dim, dim), dtype=complex)
        for d in self.determinants:
            if d < 0 or d >= dim:
                raise ValueError(f"determinant index {d} out of range for n={n_qubits}")
            P[d, d] = 1.0
        return P


@dataclass(frozen=True)
class SimilaritySandwichResourceSummary:
    """Resource/accounting summary for the compiled outer sandwich.

    ``alpha`` is the normalization carried by the underlying
    Hamiltonian block encoding ``W_H``. ``u_sigma_circuit`` is the
    amplified paper-facing similarity-side unitary used inside the
    returned outer sandwich.
    """

    alpha: float
    n_system: int
    n_ancilla: int
    compiled_alpha_bar: float
    projector_rank: int
    u_sigma_call_count: int
    hamiltonian_oracle: LCUResourceSummary
    generator_exp_oracle: GeneratorExpResourceSummary
    u_sigma_circuit: CircuitResourceSummary
    circuit: CircuitResourceSummary


@dataclass
class SimilaritySandwich:
    """Bundle of data plus compile-once metadata for the sandwich oracle.

    ``circuit`` is the literal nested compiled-object composition
    ``U_sigma^dagger W_H U_sigma``.

    ``encoded_system_block_dense`` is the exact ancilla-zero system
    block of that returned circuit.

    ``H_eff_dense`` is the dense paper target
    ``P e^{-sigma(m)} H e^{sigma(m)} P`` used for verification, with the
    projector kept external exactly as in Eq. (51)-(52).
    """

    encoded_system_block_dense: np.ndarray
    H_eff_dense: np.ndarray
    circuit: Circuit
    topology_hash: str
    compiled_signature_hash: str
    alpha: float
    mask: ChannelMask
    compiled_alpha_bar: float
    resources: SimilaritySandwichResourceSummary
    generator_exp_oracle: GeneratorExpOracle = field(repr=False)
    hamiltonian_oracle: LCUBlockEncoding = field(repr=False)
    _pool: HamiltonianPool = field(repr=False)
    _generator: ClusterGenerator = field(repr=False)
    _projector: ModelSpaceProjector = field(repr=False)
    _exp_eps: float = field(repr=False)
    _qsp_n_grid: int | None = field(repr=False)
    _qsp_max_iter: int = field(repr=False)
    _tol: float = field(repr=False)

    def redial_mask(self, mask: ChannelMask) -> "SimilaritySandwich":
        """Rebuild mask-dependent PREP data on the fixed compiled template.

        The compiled sigma selector width, Hamiltonian oracle, and QSP
        phase structure are kept fixed through ``compiled_alpha_bar``.
        If the new mask would require a larger compiled normalization,
        the caller must rebuild from scratch with a larger
        ``compiled_alpha_bar``.
        """
        return _build_similarity_sandwich_from_compiled_parts(
            self._pool,
            self._generator,
            mask,
            self._projector,
            hamiltonian_oracle=self.hamiltonian_oracle,
            compiled_alpha_bar=self.compiled_alpha_bar,
            channel_norms=self.generator_exp_oracle.sigma_oracle.channel_norms,
            exp_eps=self._exp_eps,
            qsp_n_grid=self._qsp_n_grid,
            qsp_max_iter=self._qsp_max_iter,
            tol=self._tol,
        )


def effective_hamiltonian_dense(
    pool: HamiltonianPool,
    generator: ClusterGenerator,
    mask: ChannelMask,
    projector: ModelSpaceProjector,
) -> np.ndarray:
    """Exact dense ``P e^{-sigma(m)} H e^{sigma(m)} P`` on the full Fock space."""
    H = pool.dense_matrix()
    sigma = _mask_rescaled_sigma(generator, mask)
    e_pos = expm(sigma)
    e_neg = expm(-sigma)
    P = projector.dense_matrix(pool.n_orbitals)
    return P @ e_neg @ H @ e_pos @ P


def _mask_rescaled_sigma(generator: ClusterGenerator, mask: ChannelMask) -> np.ndarray:
    """Dense masked ``sigma`` over the same compiled singles+doubles pool as the oracle."""
    return dense_masked_generator_sigma(generator, mask)


def _validate_mask_length(generator: ClusterGenerator, mask: ChannelMask, *, tol: float = 1e-12) -> None:
    channels = generator.generator_channels(tol=tol)
    if mask.weights.shape != (len(channels),):
        raise ValueError(
            "mask length must match the compiled generator pool: "
            f"got {mask.weights.shape[0]}, expected {len(channels)}"
        )


def _compiled_sigma_template(
    generator: ClusterGenerator,
    *,
    tol: float,
    compiled_alpha_bar: float | None,
) -> tuple[np.ndarray, float]:
    channels = generator.generator_channels(tol=tol)
    template_oracle = build_sigma_pool_oracle(generator, uniform_mask(len(channels)), tol=tol)
    target_alpha = template_oracle.alpha if compiled_alpha_bar is None else float(compiled_alpha_bar)
    return template_oracle.channel_norms, target_alpha


def _build_similarity_sandwich_from_compiled_parts(
    pool: HamiltonianPool,
    generator: ClusterGenerator,
    mask: ChannelMask,
    projector: ModelSpaceProjector,
    *,
    hamiltonian_oracle: LCUBlockEncoding,
    compiled_alpha_bar: float,
    channel_norms: np.ndarray,
    exp_eps: float,
    qsp_n_grid: int | None,
    qsp_max_iter: int,
    tol: float,
) -> SimilaritySandwich:
    _validate_mask_length(generator, mask, tol=tol)
    compiled_mask = mask.with_compiled_alpha_bar(channel_norms, alpha_bar=compiled_alpha_bar)
    sigma_oracle = build_sigma_pool_oracle(
        generator,
        compiled_mask,
        tol=tol,
        alpha_bar=compiled_alpha_bar,
    )
    generator_exp_oracle = build_generator_exp_oracle(
        sigma_oracle,
        eps=exp_eps,
        qsp_n_grid=qsp_n_grid,
        qsp_max_iter=qsp_max_iter,
        exp_sign=1,
    )

    total_alpha = hamiltonian_oracle.alpha

    n_sys = pool.n_orbitals
    total_ancilla = max(generator_exp_oracle.n_ancilla, hamiltonian_oracle.n_ancilla)
    circuit = Circuit(num_qubits=n_sys + total_ancilla)
    circuit.append(
        CircuitCall(
            name="U_sigma",
            qubits=tuple(range(n_sys + generator_exp_oracle.n_ancilla)),
            subcircuit=generator_exp_oracle.unitary_circuit,
            kind="U_sigma_oracle",
        )
    )
    circuit.append(
        CircuitCall(
            name="W_H",
            qubits=tuple(range(n_sys + hamiltonian_oracle.n_ancilla)),
            subcircuit=hamiltonian_oracle.circuit,
            kind="W_H_oracle",
        )
    )
    circuit.append(
        CircuitCall(
            name="U_sigma^dag",
            qubits=tuple(range(n_sys + generator_exp_oracle.n_ancilla)),
            subcircuit=generator_exp_oracle.unitary_circuit.inverse(),
            kind="U_sigma_oracle",
        )
    )

    encoded_system_block = ancilla_zero_system_block(circuit, system_width=n_sys)
    H_eff = effective_hamiltonian_dense(pool, generator, compiled_mask, projector)

    resources = SimilaritySandwichResourceSummary(
        alpha=total_alpha,
        n_system=n_sys,
        n_ancilla=total_ancilla,
        compiled_alpha_bar=compiled_alpha_bar,
        projector_rank=len(projector.determinants),
        u_sigma_call_count=sum(g.kind == "U_sigma_oracle" for g in circuit.gates),
        hamiltonian_oracle=hamiltonian_oracle.resources,
        generator_exp_oracle=generator_exp_oracle.resources,
        u_sigma_circuit=generator_exp_oracle.unitary_circuit.resource_summary(),
        circuit=circuit.resource_summary(),
    )

    return SimilaritySandwich(
        encoded_system_block_dense=encoded_system_block,
        H_eff_dense=H_eff,
        circuit=circuit,
        topology_hash=circuit.two_qubit_topology_hash(),
        compiled_signature_hash=circuit.compiled_signature_hash(),
        alpha=total_alpha,
        mask=compiled_mask,
        compiled_alpha_bar=compiled_alpha_bar,
        resources=resources,
        generator_exp_oracle=generator_exp_oracle,
        hamiltonian_oracle=hamiltonian_oracle,
        _pool=pool,
        _generator=generator,
        _projector=projector,
        _exp_eps=exp_eps,
        _qsp_n_grid=qsp_n_grid,
        _qsp_max_iter=qsp_max_iter,
        _tol=tol,
    )


def build_similarity_sandwich(
    pool: HamiltonianPool,
    generator: ClusterGenerator,
    mask: ChannelMask,
    projector: ModelSpaceProjector,
    *,
    compiled_alpha_bar: float | None = None,
    exp_eps: float = 1e-3,
    qsp_n_grid: int | None = None,
    qsp_max_iter: int = 1200,
    tol: float = 1e-12,
) -> SimilaritySandwich:
    """Build the unprojected sandwich from the real compiled child oracles.

    The returned ``circuit`` is the literal nested compiled-object
    composition ``U_sigma(m)^dagger W_H U_sigma(m)``, where ``U_sigma``
    is the amplified paper-facing unitary derived from the compiled
    generator-exp block encoding and ``W_H`` is the Hamiltonian
    block-encoding circuit.

    ``encoded_system_block_dense`` is the exact ancilla-zero system
    block of that returned circuit.

    ``H_eff_dense`` remains the dense paper target
    ``P^(m) e^{-sigma(m)} H e^{sigma(m)} P^(m)``. The model-space
    projector therefore stays external, matching Sec. IV.C exactly.
    """
    _validate_mask_length(generator, mask, tol=tol)
    hamiltonian_oracle = build_hamiltonian_block_encoding(pool)
    channel_norms, target_alpha = _compiled_sigma_template(
        generator,
        tol=tol,
        compiled_alpha_bar=compiled_alpha_bar,
    )
    return _build_similarity_sandwich_from_compiled_parts(
        pool,
        generator,
        mask,
        projector,
        hamiltonian_oracle=hamiltonian_oracle,
        compiled_alpha_bar=target_alpha,
        channel_norms=channel_norms,
        exp_eps=exp_eps,
        qsp_n_grid=qsp_n_grid,
        qsp_max_iter=qsp_max_iter,
        tol=tol,
    )
