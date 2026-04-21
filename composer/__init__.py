"""Checkout-local shim for the ``src`` layout.

When Python is started from the repository root, this package forwards
imports to ``src/composer`` so plain

    python -c "import composer"

works without a manual ``PYTHONPATH`` export. Installed environments
continue to use the normal package metadata under ``src/``.

This shim is a checkout-local convenience only; the supported runtime
floor is still the package floor declared in ``pyproject.toml``
(``Python >= 3.10``).
"""
from __future__ import annotations

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parents[1] / "src" / "composer"
_SRC_INIT = _SRC_PACKAGE / "__init__.py"

__file__ = str(_SRC_INIT)
__path__ = [str(_SRC_PACKAGE)]

exec(compile(_SRC_INIT.read_text(encoding="utf-8"), __file__, "exec"), globals(), globals())
