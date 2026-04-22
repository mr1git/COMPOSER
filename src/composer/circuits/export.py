"""Optional export adapters for compiled COMPOSER circuits.

The semantic core remains the repo's own ``Circuit`` object model. This
module provides thin backend adapters that lower compiled circuits into
external SDK objects for transpilation, simulation, and resource
inspection without making those SDKs mandatory for the reference
workflow.
"""
from __future__ import annotations

from .circuit import Circuit
from .backends.qiskit import qiskit_available, to_qiskit

__all__ = [
    "available_export_backends",
    "export_circuit",
    "qiskit_available",
    "to_qiskit",
]


def available_export_backends() -> tuple[str, ...]:
    """Return the export backends supported by this checkout."""
    return ("qiskit",)


def export_circuit(circuit: Circuit, *, backend: str = "qiskit", **kwargs):
    """Export ``circuit`` into an SDK-native circuit object.

    Parameters
    ----------
    circuit:
        The compiled COMPOSER circuit to export.
    backend:
        Backend identifier. The current repo scope supports ``"qiskit"``.
    kwargs:
        Forwarded to the backend-specific exporter.
    """
    if backend == "qiskit":
        return to_qiskit(circuit, **kwargs)
    raise ValueError(
        f"unsupported export backend {backend!r}; "
        f"available backends: {', '.join(available_export_backends())}"
    )
