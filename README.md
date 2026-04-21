# COMPOSER

A reference implementation of **COMPOSER** (Peng, Liu, Kowalski, 2026):
*Compile-Once Modular Parametric Oracle for Similarity-Encoded Effective
Reduction*. The repository provides paper-anchored, numerically
verifiable building blocks for the mask-aware, rank-one block-encoding
architecture described in the paper.

## What is in the box

* Second-quantized rank-one operator primitives (Def 1/2/3, Sec II.A).
* Pivoted Cholesky of the two-electron integrals and the mean-field
  shift (Sec II.B, Eq 11-12).
* Deterministic ladder state-preparation circuits for one- and two-electron
  targets (Sec III, App A).
* Explicit sigma-hat singles channels
  `t_ai a_a^dag a_i` plus explicit doubles channels
  `L_s = omega_s B^dag[U(s)] B[V(s)]` with occupied/virtual embeddings,
  plus dense reconstruction only as a reference check (Sec II.C).
* Block encodings for bilinear rank-one operators (Lemma 1), projected
  quadratic rank-one operators via an explicit rotated-mode
  PREP/SELECT/occupation-flag construction plus a constant-overhead
  exact degree-2 transform (Lemma 2), and the binary multiplexed LCU of
  the full rank-one pool (Theorem 1).
* A real sigma-pool oracle for the compiled generator pool:
  explicit singles channels plus pair-SVD doubles channels on a fixed
  selector register, mask-aware `PREP_sigma`, branch-multiplexed
  `SELECT_sigma`, and an explicit null branch over the compiled pool
  (Sec IV.B, Eq 39-43).
* Scalar QSP utilities plus an oracle-facing generator-exponential
  construction: parity-resolved QSP ladders for `cos` / `sin` on the
  compiled `U_sigma`, explicit block-level Hermitian-part extraction,
  and a final LCU for `e^{sigma}`. The returned object now keeps these
  as reusable compiled subcircuits rather than collapsing them into one
  dense full-width gate; dense matrices are synthesized only in the
  verification helpers/properties. A dense Chebyshev matrix path is
  retained only as a numerical reference (Sec IV.B, App C target
  function).
* The mask-aware similarity sandwich `W_eff = U_sigma^dag W U_sigma`,
  where the returned circuit is now the literal nested compiled-object
  composition `U_sigma^dag W_H U_sigma` built from the real
  generator-exp oracle and the real Hamiltonian oracle, not from a
  dense extracted `U_sigma` system gate. The generator-exp path now
  applies one round of oblivious amplitude amplification to its
  parity-split `e^sigma / 2` block encoding, so the similarity-side
  `U_sigma` subcircuit has ancilla-zero block `e^{sigma(m)}` up to the
  QSP approximation error. Consequently, the returned
  `encoded_system_block_dense` is now claimed to satisfy
  `alpha_H * encoded_system_block_dense ~= e^{-sigma(m)} H e^{sigma(m)}`
  before the model-space projector is applied. The projector `P^(m)`
  remains external and the returned `H_eff_dense` is the dense paper
  target `P^(m) e^{-sigma(m)} H e^{sigma(m)} P^(m)`. Compile-once
  verification checks the fixed nested subcircuit structure plus
  mask-dependent PREP re-dialing on that template.
* Logical resource/accounting summaries derived from the actual
  compiled objects: Theorem-1 LCU branch counts and selector width,
  sigma-oracle compiled/active branch counts, generator-exp QSP query
  counts, and outer-sandwich call counts. These are App-D-style
  logical summaries tied to the returned circuits/oracles, not
  fault-tolerant Toffoli/T counts.
* MP2 doubles amplitudes, App-E.3 MP2-weighted cumulative-coverage
  selector masks, and the App-E.2 rank-cumulative wAUC
  subspace-overlap diagnostic.

Every major module points at the paper section/equation it implements;
see [`PAPER_MAPPING.md`](PAPER_MAPPING.md) for the full table and
[`ASSUMPTIONS.md`](ASSUMPTIONS.md) for every implementation choice made
where the paper leaves room for interpretation.

## Dependencies

* Python >= 3.10
* `numpy >= 1.24`
* `scipy >= 1.10`
* `pytest >= 7.0` (for tests)

The supported dev/test environment is a Python `3.10+` virtualenv. If
your shell exposes the interpreter as `python3` rather than `python`,
use that executable to create the virtualenv and then use `python`
inside the activated environment.

No quantum-computing library is required. A tiny dense-matrix
statevector simulator ships with the package
(`src/composer/circuits/simulator.py`) and is used only for numerical
verification on small systems (up to ~14 qubits).

The top-level `composer` package intentionally keeps a narrow surface:
import concrete functionality from submodules such as
`composer.block_encoding`, `composer.operators`, or `composer.qsp`.

## Supported workflow

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest
python examples/04_lcu_hamiltonian_h2.py
python examples/05_similarity_sandwich.py
```

Any Python `3.10+` interpreter is supported; `python3.12` is shown only
as a concrete example. The supported reproducibility contract is:

* create an activated Python `3.10+` virtualenv,
* install the repo editable with dev dependencies,
* run `python -m pytest` from the repository root, and
* run the two shipped examples from that same activated environment.

For `zsh`, keep the quotes around `'.[dev]'` so the extras spec is not
treated as a glob.

The H2 dataset in `data/` and the scripts in `examples/` are repository
assets used by tests and demos; they are not installed as package data.
The repo also keeps checkout-local import conveniences for the `src`
layout: the repo-root `composer/__init__.py` shim, pytest's
`pythonpath = ["src"]`, and the example bootstraps let the active
interpreter run repo-root imports and examples without a manual
`PYTHONPATH` export. Those conveniences are part of the repo UX, but
the supported reference workflow remains the editable-install path
above.

## Running tests

```bash
python -m pytest
```

On the validated closure pass (`2026-04-21`), this command completed as
`173 passed` with no skips, no xfails, and no emitted warnings on
Python `3.12.13`.

`tests/test_foundation.py` still treats Python `<3.10` as explicitly
unsupported: the floor check is reported as such instead of failing for
an unrelated import/stdlib-introspection reason.

## Design intent

This implementation is a *research* reference, not a production engine.
The goal is one-to-one traceability from paper equations to lines of
code, with numerical equality tests for every Lemma/Theorem on small
systems. See [`IMPLEMENTATION_LOG.md`](IMPLEMENTATION_LOG.md) for a
running account of decisions, verifications and follow-ups.

## Verification status

Reference validation snapshot (`2026-04-21`, Python `3.12.13`):

* `python -m pytest` -> `173 passed in 744.76s` with no skips, xfails,
  or warning summary.
* `python examples/04_lcu_hamiltonian_h2.py` ran cleanly in `0.28s`.
* `python examples/05_similarity_sandwich.py` ran cleanly in `622.63s`.
  This example is intentionally verification-scale and is the slowest
  supported workflow because it includes scalar QSP phase fitting plus
  dense oracle verification on a synthetic `3 occ / 3 vir` system.

The pytest suite covers these baseline numerical checks:

* **Lemma 1** — top-left block of the bilinear adaptor equals the dyad
  `|u><v|` on the `N=1` sector (`tests/test_bilinear_be.py`).
* **Lemma 2** — explicit rotated-mode / occupation-flag construction:
  branch blocks equal the returned rotated occupations, the
  PREP-SELECT-PREP block equals `O_mu / Gamma_mu`, and the final
  signal-added degree-2 unitary has top-left block
  `O_mu^2 / Gamma_mu^2` (`tests/test_cholesky_channel_be.py`).
* **Theorem 1** — `alpha * top_left_block == H_dense` to
  `1e-8`, and the returned object reports branch/selector resource
  counts from the compiled `PREP_H`/`SELECT_H`/`PREP_H^dag` structure
  (`tests/test_lcu.py`, `tests/test_end_to_end_h2.py`).
* **Sigma-pool oracle** — the ancilla-zero block of the returned
  `PREP_sigma`/`SELECT_sigma`/`PREP_sigma^dag` oracle equals
  `-i sigma_pool(m) / alpha_bar` for the full compiled singles+doubles
  generator pool, mask updates change only PREP, and the null branch is
  part of the compiled bookkeeping; the returned object also reports
  selector/branch resource accounting from that compiled oracle
  (`tests/test_generator_exp.py`, `tests/test_similarity_sandwich.py`).
* **Generator-exp oracle/QSP path** — the main path now builds
  `e^{sigma}` from the compiled sigma oracle using parity-split QSP
  plus LCU, keeps repeated oracle queries and the `cos`/`sin` /
  final-LCU layers as reusable compiled subcircuits, is checked
  numerically against `scipy.linalg.expm` on small masked generators
  including mixed singles+doubles cases, and reports actual compiled
  `cos`/`sin` QSP query counts
  (`tests/test_generator_exp.py`).
* **Dense Chebyshev reference for `exp(sigma)`** — kept only as a dense
  reference and still checked against both `scipy.linalg.expm` and the
  exact truncated polynomial (`tests/test_generator_exp.py`).
* **Sigma-hat generator channels** — explicit singles channels plus the
  embedded doubles channel pool satisfy the operator identities used by
  the oracle path; the doubles pool also satisfies
  both the tensor identity
  `t2 = sum_s omega_s U^(s) \otimes V^(s)*`
  and the operator identity
  `T2 = sum_s omega_s B^dag[U^(s)] B[V^(s)]`
  (`tests/test_generator_exp.py`, `tests/test_rank_one.py`).
* **Similarity sandwich** — the returned object now uses the real
  generator oracle/QSP data instead of the old dense exponential
  reference, the returned circuit is a literal nested oracle
  composition, and the returned circuit's ancilla-zero block now
  matches the unprojected paper target
  `e^{-sigma(m)} H e^{sigma(m)} / alpha_H` on the supported
  verification-scale cases. The reported `H_eff_dense` remains the
  separate projected paper target
  `P e^{-sigma(m)} H e^{sigma(m)} P`, with `P` kept external exactly as
  in Sec. IV.C. Tests also check fixed compiled signatures, PREP-only
  mask re-dialing, ancilla-zero-block behavior across masks, and the nested resource
  summaries carried by the compiled
  Hamiltonian/generator/sandwich objects
  (`tests/test_similarity_sandwich.py`).
* **End-to-end H2 / STO-3G** — block-encoded Hamiltonian reproduces
  the dense Hamiltonian to machine precision; N=2 ground state
  matches published FCI within the integrals' rounding
  (`tests/test_end_to_end_h2.py`).

Run the end-to-end demo via

```bash
python examples/04_lcu_hamiltonian_h2.py
python examples/05_similarity_sandwich.py
```

## Final scope notes

This repository is closed as a small-system reference implementation,
not an active feature branch. Remaining intentional limitations are:

* Hamiltonian preprocessing / Theorem-1 support is limited to the
  real-integral electronic-structure case documented in
  [`ASSUMPTIONS.md`](ASSUMPTIONS.md).
* Generator exponentiation uses parity-split real QSP (`cos` / `sin`
  plus LCU and one round of oblivious amplitude amplification), not one
  direct complex phase sequence.
* Resource summaries are logical compiled-object accounting only; the
  repo does not estimate fault-tolerant synthesis costs.
