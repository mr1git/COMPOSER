"""Sanity checks for package foundations, the JW mapping, and the simulator.

These are preconditions for every subsequent Lemma/Theorem test.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pytest

import composer
from composer.circuits.circuit import Circuit
from composer.circuits.gate import MultiplexedGate, SelectGate, StatePreparationGate, Gate
from composer.circuits.simulator import statevector, unitary
from composer.utils import fermion as jw
from composer.utils.antisymmetric import (
    index_to_pair,
    num_pairs,
    pair_index,
    pair_matrix_from_vector,
    pairs_from_matrix,
)
from composer.utils.linalg import is_hermitian, is_unitary, top_left_block

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "src" / "composer"
EXPECTED_SUBPACKAGES = (
    "composer.block_encoding",
    "composer.circuits",
    "composer.diagnostics",
    "composer.factorization",
    "composer.ladders",
    "composer.operators",
    "composer.qsp",
    "composer.utils",
)


def _pyproject_text() -> str:
    return (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")


def _pyproject_value(name: str) -> str:
    match = re.search(rf'^{name}\s*=\s*"([^"]+)"', _pyproject_text(), re.MULTILINE)
    assert match, f"missing {name} in pyproject.toml"
    return match.group(1)


def _pyproject_list(name: str) -> list[str]:
    match = re.search(rf"^{name}\s*=\s*\[(.*?)\]", _pyproject_text(), re.MULTILINE | re.DOTALL)
    assert match, f"missing list {name} in pyproject.toml"
    return re.findall(r'"([^"]+)"', match.group(1))


def _optional_dependency_list(extra: str) -> list[str]:
    section_match = re.search(
        r"^\[project\.optional-dependencies\]\n(.*?)(?:^\[|\Z)",
        _pyproject_text(),
        re.MULTILINE | re.DOTALL,
    )
    assert section_match, "missing [project.optional-dependencies] in pyproject.toml"
    section = section_match.group(1)
    match = re.search(rf"^{re.escape(extra)}\s*=\s*\[(.*?)\]", section, re.MULTILINE | re.DOTALL)
    assert match, f"missing optional dependency list {extra!r} in pyproject.toml"
    return re.findall(r'"([^"]+)"', match.group(1))


def _package_module_names() -> list[str]:
    names = ["composer"]
    for info in pkgutil.walk_packages([str(PACKAGE_ROOT)], prefix="composer."):
        names.append(info.name)
    return sorted(names)


def _supported_python_floor() -> tuple[int, int]:
    requires_python = _pyproject_value("requires-python")
    match = re.fullmatch(r">=\s*(\d+)\.(\d+)", requires_python)
    assert match, f"unsupported requires-python format: {requires_python}"
    return int(match.group(1)), int(match.group(2))


def _stdlib_module_names() -> set[str]:
    names = set(getattr(sys, "stdlib_module_names", ()))
    if names:
        return names
    names.update(sys.builtin_module_names)
    stdlib_path = sysconfig.get_path("stdlib")
    if stdlib_path:
        for info in pkgutil.iter_modules([stdlib_path]):
            names.add(info.name)
    return names


def _third_party_import_roots() -> set[str]:
    roots: set[str] = set()
    for path in PACKAGE_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.level > 0 or node.module is None:
                    continue
                roots.add(node.module.split(".")[0])
    stdlib = _stdlib_module_names()
    return {
        root
        for root in roots
        if root not in stdlib and root not in {"__future__", "composer"}
    }


# --------------------------------------------------------- package baseline


def test_top_level_version_matches_pyproject():
    assert composer.__version__ == _pyproject_value("version")


def test_top_level_package_exports_only_version():
    assert composer.__all__ == ["__version__"]


def test_declared_python_floor_is_enforced():
    minimum = _supported_python_floor()
    if sys.version_info[:2] < minimum:
        pytest.xfail(
            "current interpreter is below the package floor declared in pyproject.toml; "
            "metadata is correct, but this environment is intentionally unsupported"
        )
    assert sys.version_info[:2] >= minimum


def test_declared_dependencies_cover_runtime_imports():
    runtime_dependencies = {dep.split(">=")[0].split("<")[0].strip() for dep in _pyproject_list("dependencies")}
    optional_dependencies = {
        dep.split(">=")[0].split("<")[0].strip()
        for dep in _optional_dependency_list("qiskit")
    }
    third_party_imports = _third_party_import_roots()
    assert third_party_imports <= (runtime_dependencies | optional_dependencies)
    assert runtime_dependencies == {"numpy", "scipy"}
    assert optional_dependencies == {"qiskit"}


def test_pytest_pythonpath_includes_src_checkout():
    match = re.search(r"^pythonpath\s*=\s*\[(.*?)\]", _pyproject_text(), re.MULTILINE | re.DOTALL)
    assert match, "pytest pythonpath must be declared in pyproject.toml"
    paths = re.findall(r'"([^"]+)"', match.group(1))
    assert "src" in paths


def test_all_package_modules_import():
    for module_name in _package_module_names():
        importlib.import_module(module_name)


def test_repo_root_python_process_can_import_package_without_manual_pythonpath():
    proc = subprocess.run(
        [sys.executable, "-c", "import composer; print(composer.__version__)"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == composer.__version__


def test_checkout_import_uses_src_layout_shim_from_repo_root():
    proc = subprocess.run(
        [sys.executable, "-c", "import composer; print(composer.__file__)"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    imported_init = Path(proc.stdout.strip()).resolve()
    assert imported_init == (PACKAGE_ROOT / "__init__.py").resolve()


def test_subpackages_exist():
    for module_name in EXPECTED_SUBPACKAGES:
        importlib.import_module(module_name)


def test_readme_claimed_repo_assets_exist():
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    example_paths = re.findall(r"^python (examples/\S+\.py)$", readme, re.MULTILINE)
    assert example_paths, "README no longer references any example scripts"
    for rel_path in example_paths:
        assert (REPO_ROOT / rel_path).is_file(), rel_path
    assert (REPO_ROOT / "data" / "h2_sto3g_integrals.npz").is_file()


def test_h2_reference_dataset_has_expected_schema():
    data_path = REPO_ROOT / "data" / "h2_sto3g_integrals.npz"
    with np.load(data_path) as data:
        assert {"h", "eri", "E_nuc", "NO", "NV"} <= set(data.files)
        assert data["h"].ndim == 2
        assert data["eri"].ndim == 4


# ---------------------------------------------------------------- JW mapping


def _manual_create(index: int, p: int, n: int) -> tuple[int | None, complex]:
    bits = [(index >> q) & 1 for q in range(n)]
    if bits[p]:
        return None, 0.0
    sign = -1.0 if sum(bits[:p]) % 2 else 1.0
    return index | (1 << p), sign


def test_anticommutation_small():
    n = 4
    for p in range(n):
        adag_p = jw.jw_a_dagger(p, n)
        a_p = jw.jw_a(p, n)
        # {a_p, a_p^dag} = I on occupation bit p, canonical JW
        # Actually full anticommutation: {a_p, a_p^dag} = I globally
        anticomm = adag_p @ a_p + a_p @ adag_p
        assert np.allclose(anticomm, np.eye(2**n)), p
        # {a_p, a_p} = 0
        assert np.allclose(a_p @ a_p, 0)
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            ac = jw.jw_a(p, n) @ jw.jw_a(q, n) + jw.jw_a(q, n) @ jw.jw_a(p, n)
            assert np.allclose(ac, 0), (p, q)


def test_jw_creation_action_matches_lsb_occupation_convention():
    n = 4
    dim = 2**n
    for p in range(n):
        adag_p = jw.jw_a_dagger(p, n)
        for basis_index in range(dim):
            ket = np.zeros(dim, dtype=complex)
            ket[basis_index] = 1.0
            actual = adag_p @ ket
            target_index, amp = _manual_create(basis_index, p, n)
            expected = np.zeros(dim, dtype=complex)
            if target_index is not None:
                expected[target_index] = amp
            assert np.allclose(actual, expected), (p, basis_index)


def test_number_operator():
    n = 3
    for p in range(n):
        np_ = jw.jw_number(p, n)
        expected = jw.jw_a_dagger(p, n) @ jw.jw_a(p, n)
        assert np.allclose(np_, expected)


def test_mode_number_matches_outer_product_definition():
    rng = np.random.default_rng(7)
    n = 4
    u = rng.normal(size=n) + 1j * rng.normal(size=n)
    u /= np.linalg.norm(u)
    expected = jw.one_body_matrix(np.outer(u, u.conj()))
    actual = jw.jw_mode_number(u)
    assert np.allclose(actual, expected, atol=1e-12)
    total_number = sum(jw.jw_number(p, n) for p in range(n))
    assert np.allclose(total_number @ actual - actual @ total_number, 0, atol=1e-12)


def test_determinant_with_phase_matches_ordered_product():
    n = 4
    for ordered in [(0,), (0, 1), (1, 0), (0, 2, 3), (3, 1, 0)]:
        idx, phase = jw.determinant_with_phase(ordered, n)
        # Build by applying creation ops on vacuum
        psi = np.zeros(2**n, dtype=complex)
        psi[0] = 1.0
        for p in ordered:
            psi = jw.jw_a_dagger(p, n) @ psi
        # Must be supported only at idx with amplitude = phase
        assert np.isclose(psi[idx], phase)
        assert np.isclose(np.linalg.norm(psi) ** 2, 1.0)


def test_determinant_index_rejects_out_of_range_orbitals():
    with pytest.raises(ValueError):
        jw.determinant_index((3,), 3)
    with pytest.raises(ValueError):
        jw.determinant_with_phase((0, 3), 3)


def test_one_body_matrix_restricts_to_input_matrix_on_single_excitation_sector():
    rng = np.random.default_rng(8)
    n = 4
    h = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    H = jw.one_body_matrix(h)
    idx = jw.single_excitation_basis_indices(n)
    assert np.allclose(H[np.ix_(idx, idx)], h, atol=1e-12)


def test_two_body_matrix_respects_physicist_order_signs():
    n = 4
    pair_basis = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    idx = np.array([(1 << p) | (1 << q) for p, q in pair_basis], dtype=int)
    row = pair_basis.index((0, 1))
    col = pair_basis.index((2, 3))

    for integral_indices, expected in [
        ((0, 1, 2, 3), 0.5),
        ((1, 0, 2, 3), -0.5),
        ((0, 1, 3, 2), -0.5),
        ((1, 0, 3, 2), 0.5),
    ]:
        eri = np.zeros((n, n, n, n), dtype=complex)
        eri[integral_indices] = 1.0
        H = jw.two_body_matrix(eri)
        block = H[np.ix_(idx, idx)]
        assert np.isclose(block[row, col], expected), integral_indices
        block[row, col] = 0.0
        assert np.allclose(block, 0.0), integral_indices


# -------------------------------------------------- antisymmetric pair indexing


def test_pair_index_roundtrip():
    for n in [2, 3, 5, 7]:
        for k in range(num_pairs(n)):
            p, q = index_to_pair(k, n)
            assert pair_index(p, q, n) == k
            assert 0 <= p < q < n


def test_pair_index_uses_lexicographic_upper_triangle_order():
    n = 4
    expected_pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for k, pair in enumerate(expected_pairs):
        assert index_to_pair(k, n) == pair
        assert pair_index(*pair, n) == k


def test_pairs_from_matrix_roundtrip():
    rng = np.random.default_rng(0)
    n = 5
    v = rng.normal(size=num_pairs(n))
    mat = pair_matrix_from_vector(v, n, antisymmetric=True)
    assert np.allclose(mat, -mat.T)
    assert np.allclose(pairs_from_matrix(mat), v)


# ---------------------------------------------------------- simulator sanity


def _x_gate() -> np.ndarray:
    return np.array([[0, 1], [1, 0]], dtype=complex)


def _h_gate() -> np.ndarray:
    return (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]], dtype=complex)


def _cnot_gate() -> np.ndarray:
    # control on qubit 0 (LSB), target on qubit 1
    # basis order |q1 q0>: 00, 01, 10, 11
    return np.array(
        [
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
        ],
        dtype=complex,
    )


def test_x_flips_bit():
    n = 3
    for p in range(n):
        c = Circuit(n)
        c.append(Gate("X", (p,), _x_gate()))
        psi = statevector(c)
        assert np.isclose(psi[1 << p], 1)


def test_h_cnot_creates_bell():
    c = Circuit(2)
    c.append(Gate("H", (0,), _h_gate()))
    c.append(Gate("CNOT", (0, 1), _cnot_gate(), kind="CNOT"))
    psi = statevector(c)
    target = (1 / np.sqrt(2)) * np.array([1, 0, 0, 1], dtype=complex)
    assert np.allclose(psi, target)


def test_unitary_matches_statevector_on_basis():
    rng = np.random.default_rng(1)
    n = 3
    c = Circuit(n)
    c.append(Gate("X", (1,), _x_gate()))
    c.append(Gate("H", (0,), _h_gate()))
    c.append(Gate("CNOT", (0, 2), _cnot_gate(), kind="CNOT"))
    U = unitary(c)
    assert is_unitary(U)
    for k in range(2**n):
        init = np.zeros(2**n, dtype=complex)
        init[k] = 1.0
        psi = statevector(c, init=init)
        assert np.allclose(psi, U[:, k]), k


def test_circuit_inverse():
    n = 3
    c = Circuit(n)
    c.append(Gate("H", (0,), _h_gate()))
    c.append(Gate("CNOT", (0, 1), _cnot_gate(), kind="CNOT"))
    c.append(Gate("X", (2,), _x_gate()))
    U = unitary(c)
    Udag = unitary(c.inverse())
    assert np.allclose(U.conj().T @ U, np.eye(2**n))
    assert np.allclose(Udag, U.conj().T)


def test_topology_hash_invariant_under_angle_changes():
    # Build two circuits with the same multi-qubit layout but
    # different single-qubit content. Hashes must match.
    n = 2
    c1 = Circuit(n)
    c2 = Circuit(n)
    # Different single-qubit gates
    c1.append(Gate("RY(0.1)", (0,), np.eye(2, dtype=complex)))
    c2.append(Gate("RY(0.7)", (0,), np.eye(2, dtype=complex)))
    # Same multi-qubit gate
    c1.append(Gate("CNOT", (0, 1), _cnot_gate(), kind="CNOT"))
    c2.append(Gate("CNOT", (0, 1), _cnot_gate(), kind="CNOT"))
    assert c1.two_qubit_topology_hash() == c2.two_qubit_topology_hash()


def test_circuit_resource_summary_tracks_compiled_gate_inventory():
    c = Circuit(3)
    c.append(Gate("H", (0,), _h_gate()))
    c.append(Gate("CNOT", (0, 1), _cnot_gate(), kind="CNOT"))
    c.append(Gate("FULL", (0, 1, 2), np.eye(8, dtype=complex), kind="FULL"))

    summary = c.resource_summary()

    assert summary.num_qubits == 3
    assert summary.gate_count == 3
    assert summary.multi_qubit_gate_count == 2
    assert summary.full_width_gate_count == 1
    assert summary.max_gate_arity == 3
    assert summary.gate_count_by_kind == {"CNOT": 1, "FULL": 1, "H": 1}
    assert summary.compiled_signature_hash == c.compiled_signature_hash()


def test_circuit_resource_report_tracks_recursive_selector_and_ancilla_overhead():
    identity_branch = Circuit(1)
    identity_branch.append(Gate("I", (0,), np.eye(2, dtype=complex), kind="I"))

    x_branch = Circuit(1)
    x_branch.append(Gate("X", (0,), _x_gate(), kind="X"))

    c = Circuit(3)
    c.append(Gate("H", (0,), _h_gate(), kind="H"))
    c.append(
        StatePreparationGate(
            name="PREP",
            qubits=(1, 2),
            amplitudes=np.array([1.0, 1.0, 0.0, 0.0], dtype=complex) / np.sqrt(2.0),
            kind="PREP",
        )
    )
    c.append(
        SelectGate(
            name="SELECT",
            qubits=(0, 1),
            zero_circuit=identity_branch,
            one_circuit=x_branch,
            kind="SELECT",
        )
    )
    c.append(
        MultiplexedGate(
            name="MUX",
            qubits=(0, 1, 2),
            selector_width=2,
            branch_circuits=(identity_branch, x_branch),
            default_circuit=identity_branch,
            kind="MUX",
        )
    )

    report = c.resource_report(system_width=1)

    assert report.logical == c.resource_summary()
    assert report.compiled.num_qubits == 3
    assert report.compiled.system_qubits == 1
    assert report.compiled.ancilla_qubits == 2
    assert report.compiled.logical_summary == c.resource_summary()
    assert report.compiled.expanded_gate_count_by_kind == {
        "H": 1,
        "I": 3,
        "MUX": 1,
        "PREP": 1,
        "SELECT": 1,
        "X": 2,
    }
    assert report.compiled.dense_leaf_gate_count == 6
    assert report.compiled.structural_gate_count == 3
    assert report.compiled.selector_control.select_gate_count == 1
    assert report.compiled.selector_control.multiplexed_gate_count == 1
    assert report.compiled.selector_control.compiled_selector_state_count == 6
    assert report.compiled.selector_control.explicit_branch_count == 4
    assert report.compiled.selector_control.default_routed_state_count == 2
    assert report.compiled.selector_control.max_selector_width == 2
    assert report.compiled.selector_control.max_control_width == 2
    assert report.compiled.selector_control.selector_width_histogram == {1: 1, 2: 1}


# --------------------------------------------------------- linalg helpers

def test_top_left_block_of_embedded():
    # W = [[A/alpha, .], [., .]] constructed manually
    alpha = 2.5
    A = np.array([[0.3, 0.1], [0.1, -0.2]], dtype=complex)
    W = np.zeros((4, 4), dtype=complex)
    W[:2, :2] = A / alpha
    # Don't require W unitary here -- just block extraction
    got = top_left_block(W, 2)
    assert np.allclose(got, A / alpha)


def test_one_body_matrix_is_hermitian():
    rng = np.random.default_rng(42)
    n = 3
    h_ = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    h_ = 0.5 * (h_ + h_.conj().T)  # Hermitize
    H = jw.one_body_matrix(h_)
    assert is_hermitian(H)


def test_two_body_matrix_chemist_symmetry():
    # With real symmetric ERIs obeying (pq|rs) = (qp|rs) = (pq|sr) = (rs|pq),
    # the two-body matrix is Hermitian.
    rng = np.random.default_rng(7)
    n = 3
    # Physicist-order ERIs from (chemist) 2e integrals
    g = rng.normal(size=(n, n, n, n))
    g = 0.5 * (g + g.transpose(1, 0, 3, 2))  # <pq|rs> = <qp|sr>  (from (pq|rs)=(qp|rs)=(pq|sr))
    g = 0.5 * (g + g.transpose(2, 3, 0, 1))  # swap bra/ket
    H = jw.two_body_matrix(g)
    assert is_hermitian(H)
