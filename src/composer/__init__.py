"""COMPOSER: reference implementation of Peng, Liu, Kowalski (2026).

Compile-Once Modular Parametric Oracle for Similarity-Encoded Effective
Reduction. This package provides:

* rank-one representations of second-quantized operators (Sec II),
* deterministic ladder state-preparation circuits (Sec III, App A),
* block encodings for bilinear and projected-quadratic rank-one operators
  (Sec IV.A, Lemmas 1-2, Theorem 1),
* scalar QSP utilities plus an oracle/QSP/LCU construction for
  anti-Hermitian generator exponentiation, with a dense Chebyshev
  reference branch retained for verification (Sec IV.B, App C target),
* the mask-aware similarity sandwich (Sec IV.C),
* MP2-based selector-mask diagnostics and rank-cumulative wAUC overlap
  metrics (App E).

See PAPER_MAPPING.md for the full section/equation -> file table.

The top-level package intentionally exports only ``__version__``.
Import concrete functionality from subpackages such as
``composer.block_encoding`` or ``composer.operators``.

Supported runtime: Python ``>=3.10`` as declared in ``pyproject.toml``.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
