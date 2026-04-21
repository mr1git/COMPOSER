"""Low-degree polynomial helpers used by the verification-scale QSVT paths.

Two degree-2 views are kept side-by-side:

* ``x_squared_phases`` is the closed-form scalar Wx-QSP schedule whose
  top-left entry has real part ``x^2``.
* ``degree_two_projector_transform`` is the exact block-level
  composition used by the more literal Appendix B.2 Lemma-2 builder.
  It inserts the ancilla-zero projector between two calls to an
  existing block encoding, using one signal qubit and the standard
  Hadamard-sandwiched reflection gadget. This is a constant-overhead
  exact degree-2 transform on the encoded operator.

Derivation of the ``x^2`` phases (Wx convention)
-------------------------------------------------

For 3 phases ``Phi = (phi_0, phi_1, phi_2)`` the Wx QSP unitary is

    U(x, Phi) = S(phi_0) W(x) S(phi_1) W(x) S(phi_2)

with top-left entry (after expansion)

    P(x)  =  x^2 e^{i A}  -  (1 - x^2) e^{i B}

where ``A = phi_0 + phi_1 + phi_2`` and ``B = phi_0 - phi_1 + phi_2``.
Setting ``A = 0`` and ``B = pi/2`` yields

    Re(P(x)) = x^2 cos(0) - (1 - x^2) cos(pi/2) = x^2     (exact)

with an imaginary ``-i (1 - x^2)`` contribution that sits on the
off-block projector (and therefore does not appear in
``<0| U |0>``'s real part). The symmetric choice
``phi_0 = phi_2 = pi / 8``, ``phi_1 = -pi / 4`` solves ``A = 0`` and
``B = pi / 2`` and is the schedule exported here.

Note
----
``Re(<0| U |0>)`` equals ``x^2`` exactly; the full ``<0| U |0>`` is
``x^2 - i (1 - x^2)``. ``block_encoding/cholesky_channel.py`` keeps
this schedule as a scalar reference, but exposes the exact Hermitian
channel ``A^2`` directly at the block-encoding level rather than
synthesizing that transformation from the phase sequence.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "ancilla_zero_projector_block_encoding",
    "degree_two_projector_transform",
    "x_squared_phases",
]


def _hadamard() -> np.ndarray:
    return np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex) / np.sqrt(2.0)


def ancilla_zero_projector_block_encoding(n_block_ancilla: int) -> np.ndarray:
    """Return a single-signal block encoding of ``|0...0><0...0|``.

    The returned unitary acts on one signal qubit plus the
    ``n_block_ancilla``-qubit ancilla register whose zero projector is
    being encoded. The qubit layout follows the repo's usual
    ancilla-as-MSB convention: the new signal qubit is the most
    significant qubit.
    """
    if n_block_ancilla < 0:
        raise ValueError(f"n_block_ancilla must be non-negative, got {n_block_ancilla}")
    dim = 1 << n_block_ancilla
    projector = np.zeros((dim, dim), dtype=complex)
    projector[0, 0] = 1.0
    reflection = np.eye(dim, dtype=complex) - 2.0 * projector
    controlled_minus_reflection = np.block(
        [
            [np.eye(dim, dtype=complex), np.zeros((dim, dim), dtype=complex)],
            [np.zeros((dim, dim), dtype=complex), -reflection],
        ]
    )
    H = _hadamard()
    H_full = np.kron(H, np.eye(dim, dtype=complex))
    with np.errstate(all="ignore"):
        return H_full @ controlled_minus_reflection @ H_full


def degree_two_projector_transform(
    block_encoding: np.ndarray, *, n_block_ancilla: int
) -> np.ndarray:
    """Return an exact degree-2 transform of an existing block encoding.

    Parameters
    ----------
    block_encoding
        Unitary whose ancilla-zero block encodes ``A``.
    n_block_ancilla
        Number of ancilla qubits inside ``block_encoding``.

    Returns
    -------
    np.ndarray
        Unitary on one extra signal qubit plus the original registers
        whose all-zero ancilla block equals ``A^2`` exactly.
    """
    block_encoding = np.asarray(block_encoding, dtype=complex)
    dim = block_encoding.shape[0]
    if block_encoding.shape != (dim, dim):
        raise ValueError("block_encoding must be square")
    anc_dim = 1 << n_block_ancilla
    if dim % anc_dim != 0:
        raise ValueError(
            "block_encoding dimension is incompatible with the stated ancilla width"
        )
    system_dim = dim // anc_dim
    lifted = np.kron(np.eye(2, dtype=complex), block_encoding)
    projector = np.kron(
        ancilla_zero_projector_block_encoding(n_block_ancilla),
        np.eye(system_dim, dtype=complex),
    )
    with np.errstate(all="ignore"):
        return lifted @ projector @ lifted


def x_squared_phases() -> np.ndarray:
    """Three Wx-convention phases whose QSP polynomial has ``Re P(x) = x^2``."""
    return np.array([np.pi / 8.0, -np.pi / 4.0, np.pi / 8.0], dtype=float)
