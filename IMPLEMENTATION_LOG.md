# Implementation log

Running account of decisions, verifications and follow-ups. Newest
entries on top.

---

## 2026-04-22 — Structural Hamiltonian ancilla width no longer forces eager dense full-unitary synthesis

* Making the Theorem-1 Hamiltonian path structurally honest exposed one
  last verification-model gap: the repo still formed
  `circuit_unitary(build_hamiltonian_block_encoding(...).circuit)`
  eagerly inside `build_hamiltonian_block_encoding(...)`. Once the
  Cholesky branches stopped hiding behind one-ancilla dense wrappers,
  that eager full-unitary synthesis became the true blocker for the
  supported workflow rather than the structural circuit itself.
* The fix was not to shrink the compiled circuit back down. Instead,
  `src/composer/circuits/simulator.py` now exposes
  `ancilla_zero_system_block(...)`, which extracts the paper-facing
  ancilla-zero system block directly from a compiled circuit without
  materializing the full `2^(n+a) x 2^(n+a)` unitary.
* `src/composer/block_encoding/lcu.py` now stores that ancilla-zero
  block eagerly and makes the full dense `W_H` matrix lazy. The
  returned Theorem-1 object therefore keeps the honest widened ancilla
  footprint in `circuit`/`resources`, while `top_left_block()` remains
  cheap enough for the supported H2 workflow and the full dense matrix
  is still available to small-system tests that explicitly ask for it.
* `src/composer/block_encoding/similarity_sandwich.py` now reports
  `encoded_system_block_dense` by extracting the ancilla-zero block of
  the actual compiled outer sandwich circuit, instead of depending on a
  separately materialized full dense Hamiltonian oracle matrix.
* Added a regression in `tests/test_lcu.py` that fails if the Theorem-1
  builder goes back to eagerly materializing the full compiled unitary
  just to provide `top_left_block()`.
* Closure validation after this fix showed the remaining supported-workflow
  bottleneck more cleanly: the old Hamiltonian-memory failure is gone,
  but the long pole is still repeated generator/QSP phase fitting and
  its dense verification path (`tests/test_generator_exp.py`, the full
  suite, and `examples/05_similarity_sandwich.py`).

## 2026-04-22 — Opaque one-body and Cholesky branch wrappers removed from the main scalable path

* Re-audited the remaining dense leaves in the paper-facing scalable
  path after the QSP/parity work. Two avoidable shortcuts were still in
  the main oracle flow:
  the Hamiltonian LCU still wrapped each Lemma-2 Cholesky branch in one
  dense `W_cholesky_squared` leaf, and the sigma oracle still wrapped
  each singles branch in one dense `W_sigma_single` one-body wrapper.
  Those were no longer justified because the repo already had enough
  structure to express the rotated-mode PREP/SELECT/occupation-flag
  construction directly.
* `src/composer/block_encoding/cholesky_channel.py` now exposes an
  explicit `build_hermitian_one_body_block_encoding(...)` builder for
  the `O / Gamma` stage itself and keeps the degree-2 transform
  structural as well: two `CircuitCall`s to the one-body block encoding
  plus an explicit ancilla-projector gadget (`H`, `SelectGate`,
  `AncillaZeroReflectionGate`, `H`) instead of one dense squaring
  wrapper.
* `src/composer/block_encoding/lcu.py` now reuses the structural
  Lemma-2 circuits directly inside `SELECT_H`. This forced a real
  resource-accounting correction: Hamiltonian branches no longer all fit
  inside a fictitious one-ancilla workspace, so
  `LCUResourceSummary.subencoding_ancilla` now reports the actual
  maximum branch workspace width and the outer oracle ancilla count
  grows accordingly on channels that retain index qubits.
* `src/composer/block_encoding/generator_exp.py` now reuses the same
  structural one-body builder for singles branches in `SELECT_sigma`.
  Regressions back to opaque one-body wrappers are now caught by
  `tests/test_generator_exp.py`, which checks that the recursive
  dense-leaf report no longer contains `W_sigma_single` and instead
  exposes the remaining low-level `Givens(...)` leaves explicitly.
* `src/composer/circuits/resources.py` now reports
  `dense_leaf_gate_count_by_kind` in addition to the aggregate dense
  leaf count. This makes the final unavoidable leaves explicit in both
  code and docs: after the wrapper cleanup, the main remaining dense
  primitives are low-level full-register fermionic ladder gates,
  `PAIR_branch_reflection`, and exact exported state-preparation
  unitaries in the optional backend path.
* Added/updated focused regressions:
  `tests/test_cholesky_channel_be.py` now locks in that the stored
  one-body and degree-2 matrices match the structural circuits;
  `tests/test_lcu.py` now asserts that the recursive report no longer
  contains `W_cholesky_squared` and that the widened branch workspace is
  surfaced honestly;
  `tests/test_qiskit_export.py` now exports the shipped H2 LCU and
  fails if the optional backend path collapses that compiled oracle into
  one top-level dense SDK unitary.

## 2026-04-21 — Foundation dependency audit no longer misclassifies stdlib extension modules

* Re-audited the remaining repo-hygiene closure issue in
  `tests/test_foundation.py`. The dependency audit was still too
  shallow when `sys.stdlib_module_names` was unavailable: it listed only
  builtins plus top-level modules under `sysconfig.get_path("stdlib")`,
  which misses extension modules living under stdlib `lib-dynload`.
  That could incorrectly classify stdlib imports such as `cmath`
  (currently used by the optional Qiskit backend) as third-party
  dependencies.
* `tests/test_foundation.py` now keeps the strong audit but changes the
  classifier: on Python `3.10+` it trusts `sys.stdlib_module_names`,
  and otherwise it resolves each import root through
  `importlib.util.find_spec(...)` and checks that the discovered module
  origin lives under stdlib/platstdlib roots or `DESTSHARED` while
  explicitly excluding `purelib` / `platlib` (`site-packages`).
  This preserves the audit's value instead of weakening it into a broad
  allowlist.
* Added a focused regression check that `cmath` is recognized as stdlib,
  so the foundation suite fails if extension-module detection regresses
  again.
* Updated `README.md` to describe that version-specific dependency-audit
  behavior accurately while leaving the supported workflow unchanged:
  the honest contract is still an activated Python `3.10+` editable
  install plus `python -m pytest`.
* Re-ran the relevant foundation suite in the supported environment:
  `./.venv-supported/bin/python -m pytest tests/test_foundation.py -q`
  passed cleanly.

## 2026-04-21 — Resource-estimation views now separate logical summaries, compiled synthesis overhead, and optional backend counts

* Audited the current closure state against the scoped "final scalable
  phase" requirement. The repo already had logical `resources`
  summaries on returned oracle objects plus an optional Qiskit export
  adapter, but there was still a public-contract gap: users could see
  top-level branch/query counts, yet they could not ask one stable API
  for recursive ancilla/selector overhead on the compiled circuits
  themselves or for backend-side depth / two-qubit counts without
  manually re-exporting and re-transpiling examples.
* `src/composer/circuits/resources.py` now introduces
  `resource_report(...)`, exposed both as
  `composer.circuits.resource_report(...)` and
  `Circuit.resource_report(...)`. The report is layered intentionally:
  1. the existing logical `Circuit.resource_summary()`,
  2. a recursive compiled-synthesis view over the structural COMPOSER
     circuit (ancilla count, recursive gate-family inventory,
     selector/control-state overhead, dense-leaf counts), and
  3. an optional backend/export view for Qiskit that reports exported
     instruction families and, when a transpilation basis is supplied,
     transpiled depth plus two-qubit counts/depth.
* This closes an important public-story gap. Before this change, the
  repo could say "the scalable circuit path exists" but the actionable
  resource surface was still mostly hand-built example printouts and
  top-level logical counters. After this change, the scalable backend
  story is visible through one API without collapsing the dense
  reference layer and the optional export layer into one ambiguous
  notion of "resources."
* Tests were strengthened in three directions:
  `tests/test_foundation.py` now locks in recursive selector/ancilla
  reporting on a synthetic structural circuit,
  `tests/test_lcu.py` checks that Theorem-1 reports selector overhead
  and ancilla width through the new compiled view, and
  `tests/test_similarity_sandwich.py` checks that the outer sandwich
  report recurses into nested oracles rather than only restating its
  top-level three-subcircuit shell.
* `tests/test_qiskit_export.py` now also validates the optional backend
  report: with a flattened Qiskit export and a concrete `("u", "cx")`
  basis, depth and two-qubit counts become available and are exposed
  separately from the compiled COMPOSER-side ancilla/selector report.
  This fails if the public resource API regresses to "logical summary
  only" or stops using the export backend for representable backend
  counts.
* `examples/04_lcu_hamiltonian_h2.py`,
  `examples/05_similarity_sandwich.py`, and
  `examples/06_qiskit_export_h2.py` now show the intended split
  explicitly:
  reference validation output first,
  compiled synthesis/resource reporting second,
  and optional backend/export analysis third.
* `README.md`, `PAPER_MAPPING.md`, and `ASSUMPTIONS.md` now describe the
  repo as a stable two-layer implementation core
  (reference semantics + scalable synthesized circuits) with an
  explicitly optional export/backend layer for SDK-side inspection.

## 2026-04-21 — Optional SDK export layer added on top of the compiled circuit model

* Audited the paper's compile-once intent against the repo's current
  circuit/object model and the requested scope. The semantic core had
  already moved past the older "one dense matrix for the whole oracle"
  limitation: compiled COMPOSER circuits now keep structural objects
  such as `CircuitCall`, `SelectGate`, `MultiplexedGate`,
  `StatePreparationGate`, and `AncillaZeroReflectionGate`.
  The remaining gap was that those compiled objects terminated only in
  the repo's dense verification simulator, so there was no real SDK
  adapter for transpilation, simulation, or SDK-side resource analysis.
* `src/composer/circuits/export.py` and
  `src/composer/circuits/backends/qiskit.py` now add an optional export
  layer with Qiskit as the first backend. The base workflow remains free
  of any mandatory quantum-SDK dependency; `pyproject.toml` now exposes
  that path through the optional `qiskit` extra.
* The exporter lowers the core compiled object model recursively:
  primitive dense `Gate` leaves go to native Qiskit gates when
  recognized and otherwise to exact `UnitaryGate`s; `CircuitCall`s are
  preserved as reusable child subcircuits; `SelectGate` and
  `MultiplexedGate` are lowered into state-conditioned controlled
  subcircuit applications; and `AncillaZeroReflectionGate` is emitted as
  an ancilla-local reflection circuit rather than a full-width dense
  diagonal.
* One deliberate semantic-preservation choice is called out explicitly:
  `StatePreparationGate` does **not** export via Qiskit's native
  prepared-state primitive, because COMPOSER defines a specific full
  verification unitary via Gram-Schmidt completion. The adapter
  therefore exports the exact COMPOSER state-preparation unitary so the
  SDK object matches the reference semantics.
* `tests/test_qiskit_export.py` now validates export on small compiled
  circuits and fails if the adapter regresses to flattening the whole
  COMPOSER circuit into one dense top-level `UnitaryGate`.
  `examples/06_qiskit_export_h2.py` demonstrates export of the shipped
  H2 Theorem-1 Hamiltonian oracle and reports SDK-side instruction/depth
  data together with a statevector consistency check against COMPOSER's
  simulator.
* What remains deferred is lower-level scalable synthesis for dense leaf
  primitives already present in the semantic core. The adapter now makes
  those leaves visible to real SDK tooling, but it does not yet replace
  those leaf matrices with new gate-by-gate scalable constructions.

## 2026-04-21 — Generator-exp phase compilation now requests the direct complex path first and records why one ladder still fails on the current model

* Re-audited Sec. IV.B Eq. (44)-(46) and Appendix C against the current
  `qsp/{chebyshev,phases}.py` and `block_encoding/generator_exp.py`
  path. The paper's intent is a single complex exponential target
  `exp(-i alpha x)`, while the repo had been compiling generator-exp
  phases by issuing two separate trig-specific scalar fits directly from
  `generator_exp.py`.
* `src/composer/qsp/phases.py` now introduces compiled phase-sequence
  objects for real Chebyshev targets plus a compiled exponential phase
  schedule rooted in the direct complex Jacobi-Anger series. The
  Chebyshev-basis solve no longer round-trips through a monomial target
  before optimization; it keeps the target in Chebyshev form until the
  scalar loss is sampled on the cosine grid.
* `src/composer/block_encoding/generator_exp.py` now consumes that
  compiled exponential schedule instead of calling trig-specific scalar
  fit helpers directly. The returned `GeneratorExpOracle` and its
  resource summary now expose the resolved compilation strategy,
  whether a single ladder was used, and the direct complex truncation
  degree used to anchor the schedule.
* The direct complex request is now explicit rather than implied:
  `build_generator_exp_oracle(...)` asks for a direct complex
  single-ladder compile first, while `qsp/phases.py` records why that
  request still falls back on the current repo scope. The narrow reason
  is no longer documented merely as "the scalar solver is real-only":
  on the current Wx/top-left circuit model, one ladder can only realize
  a definite-parity scalar polynomial, whereas `exp(-i alpha x)` has
  both even and odd Chebyshev sectors.
* The resolved implementation remains truthful about the remaining gap:
  the repo still synthesizes `e^{sigma}` through the structured
  parity-split `cos` / `sin` fallback plus LCU, but those fallback
  branch targets are now derived directly from the complex Jacobi-Anger
  coefficients rather than being treated as the primary compile inputs.
* `tests/test_qsp.py` now checks the compiled Chebyshev-target metadata
  plus the parity-based infeasibility bookkeeping for one direct ladder,
  while
  `tests/test_generator_exp.py` now locks in that the generator oracle
  requests the direct complex schedule first and only then consumes the
  explicit parity-split fallback object rather than two ad hoc
  trig-specific fits.

## 2026-04-21 — Reusable oracle primitives moved from dense wrappers to structural circuit objects

* Audited the recurring oracle scaffold against Sec. IV.B / IV.C and
  Figure 3 / Figure 4 of the paper: the repo already kept some
  hierarchy through `CircuitCall` and two-branch `SelectGate`, but the
  repeated PREP layers, multi-branch SELECT stages, and ancilla-zero
  reflections were still mostly represented as dense `Gate` matrices.
* `src/composer/circuits/gate.py` now introduces first-class
  structural primitives for synthesized state preparation
  (`StatePreparationGate`), multi-branch compiled selector dispatch
  (`MultiplexedGate`), and ancilla-zero reflections
  (`AncillaZeroReflectionGate`). `src/composer/circuits/circuit.py`
  and `src/composer/circuits/simulator.py` were updated so compiled
  signatures, resource summaries, and dense verification still work on
  those new primitives.
* `src/composer/block_encoding/lcu.py` now builds Theorem-1 as a
  structural `PREP_H` / `SELECT_H` / `PREP_H^dag` circuit over branch
  subcircuits instead of materializing the outer PREP and SELECT stages
  only as dense matrices.
* `src/composer/block_encoding/generator_exp.py` now uses the same
  structural pattern for `PREP_pair`, `SELECT_pair`, `PREP_sigma`,
  `SELECT_sigma`, the one-qubit PREP layers in the `cos` / `sin` /
  final-exp LCUs, and the ancilla-zero reflections inside oblivious
  amplitude amplification. The branch-local doubles adaptor added
  earlier remains in place and now plugs into the outer structural
  selector directly.
* Tests were strengthened so the suite now fails if these paths regress
  to dense placeholders: `tests/test_lcu.py` locks in structural
  `PREP_H`/`SELECT_H`, `tests/test_generator_exp.py` locks in
  structural pair/sigma/exp/reflection primitives, and
  `tests/test_similarity_sandwich.py` checks that compile-once redialing
  still reuses those structural child oracles.
* What remains deferred is the gate-set synthesis *inside* the generic
  structural primitives themselves. PREP is still compiled to a dense
  verification unitary from its target amplitudes, and `MultiplexedGate`
  still represents the selector-routing intent at the compiled-object
  level rather than as a decomposed elementary controlled-gate tree.

## 2026-04-21 — Doubles sigma-branch adaptor lifted out of the dense full-Fock fallback

* Audited `block_encoding/generator_exp.py`, `operators/generator.py`,
  `factorization/pair_svd.py`, and the paper’s Sec. IV.B / rank-one
  generator discussion against the repo’s then-current doubles branch
  path.
* Before this change, singles branches already used the explicit
  one-body Hermitian encoding, but doubles branches still collapsed each
  channel `-i(L_s - L_s^dag)` to one dense full-Fock Hermitian matrix
  and passed that matrix through `hermitian_fock_block_encoding(...)`
  inside the sigma oracle.
* `src/composer/block_encoding/generator_exp.py` now replaces that
  doubles fallback with an explicit channel-local internal adaptor:
  each doubles channel is compiled as
  `PREP_pair^dag SELECT_pair PREP_pair` over the active canonical
  pair-pair basis terms `U_ab V_ij^*`, and each internal `SELECT_pair`
  branch applies the corresponding local Hermitian four-orbital
  canonical pair excitation `-i(e^{i phi_abij} a_a^dag a_b^dag a_j a_i - h.c.)`.
  The sigma oracle pads those branch workspaces to one fixed compiled
  width, keeps singles first / doubles second ordering unchanged, and
  still exposes the same masked generator top-left block semantics.
* `tests/test_generator_exp.py` now checks the doubles adaptor at the
  branch level: the adaptor’s ancilla-zero block must match the
  corresponding channel sigma term, and the sigma oracle must expose the
  compiled doubles adaptors directly. Those tests fail if doubles
  regress to the old dense branch path.
* Updated `README.md`, `ASSUMPTIONS.md`, and `PAPER_MAPPING.md` to make
  the new doubles implementation contract explicit: the repo now has a
  literal channel-local internal pair-basis adaptor, while the more
  compressed Eq. (37) pair-state adaptor and the optional wedge-factor
  decomposition remain deferred.

## 2026-04-21 — Closure verification in the supported environment

* Built a clean supported virtualenv with `/opt/homebrew/bin/python3.12`
  and installed the repo via
  `python -m pip install -e '.[dev]'`. The supported workflow is now
  anchored to that editable-install path rather than to an implicit
  no-install checkout.
* Re-verified packaging/import ergonomics from the repository root:
  editable-install import resolves to `src/composer/__init__.py`,
  repo-root `python -c "import composer"` works through the shim, the
  example entry points bootstrap `src/` when run by path, and
  `tests/test_foundation.py` passes in the supported environment.
* Ran the full test suite in the supported environment:
  `python -m pytest` completed as `173 passed in 744.76s`.
  There were no skips, no xfails, and no warning summary on that run.
* Ran both documented examples under `PYTHONWARNINGS=default`:
  `examples/04_lcu_hamiltonian_h2.py` completed cleanly in `0.28s`,
  and `examples/05_similarity_sandwich.py` completed cleanly in
  `622.63s`.
* Investigated the previously reported warning classes
  (`divide by zero`, `overflow`, `invalid value`). They do not
  reproduce in the supported Python `3.10+` workflow now documented in
  the repo. The remaining suppression is intentionally localized to a
  few dense verification helpers where benign low-level floating-point
  warnings had previously appeared during exact unitary assembly; the
  normal supported workflow is now warning-clean.
* Updated `README.md`, `PAPER_MAPPING.md`, and `ASSUMPTIONS.md` to the
  final public contract: what is implemented, what environment is
  supported, how the repo should be run, and what small residual scope
  limits remain.

## 2026-04-21 — Final repo-facing truth pass: supported Python floor, checkout-local imports, and reproducible workflow

* Audited the repo-facing surfaces after the final oracle work:
  `pyproject.toml`, `composer/__init__.py`, `src/composer/__init__.py`,
  `README.md`, `tests/test_foundation.py`, and the example entry
  points. The implementation-facing oracle docs were already mostly in
  sync; the remaining drift was in environment assumptions and how the
  foundation suite reported them.
* `tests/test_foundation.py` no longer fails misleadingly on Python
  `<3.10`. It now:
  1. parses the supported floor from `pyproject.toml`,
  2. reports older interpreters as explicitly unsupported via `xfail`,
     rather than as a broken runtime assertion, and
  3. uses a version-portable stdlib-module discovery helper instead of
     relying unconditionally on `sys.stdlib_module_names` (which is not
     present on Python 3.9).
* The foundation suite also now checks the checkout-local import shim
  more literally: the repo-root subprocess import must resolve to
  `src/composer/__init__.py`, not just import some `composer` module of
  the right name/version.
* `README.md` now states the intended workflow precisely:
  create a Python `3.10+` virtualenv first, then install with
  `python -m pip install -e .[dev]`, then run `python -m pytest` from
  that activated environment. The checkout-local no-`PYTHONPATH`
  behavior is now phrased in terms of the active interpreter rather
  than assuming a `python` launcher exists globally.
* `ASSUMPTIONS.md` now records the repo-level environment assumption
  explicitly: Python `>=3.10` is the supported floor, while the root
  import shim / pytest `pythonpath` / example bootstraps are deliberate
  checkout-local usability choices rather than paper claims.

## 2026-04-21 — Singles are now part of the compiled generator oracle path

* Re-audited Sec. II.C plus Sec. IV.B/IV.C of the paper PDF, especially
  page 8 ("Singles contributions are inherently rank-one and may be
  treated analogously") and pages 11-13 (Eq. (36)-(53)). The prior repo
  state was inconsistent: dense helper paths included singles, while
  the compiled sigma oracle and similarity sandwich rejected any
  nonzero `t1`.
* `src/composer/operators/generator.py` now exposes explicit singles
  channels `t_ai a_a^dag a_i` alongside the existing pair-SVD doubles
  channels, and provides one compiled generator pool
  (`generator_channels()`) used by the oracle path.
* `src/composer/block_encoding/generator_exp.py` now builds
  `PREP_sigma` / `SELECT_sigma` / `PREP_sigma^dag` over that full
  singles+doubles pool. Singles branches are compiled through exact
  full-Fock Hermitian one-body block encodings of `-i(L-L^dag)`, while
  doubles branches keep the existing dense Hermitian subencoding path.
  The ancilla-zero block of the returned sigma oracle therefore matches
  the masked full generator rather than only the masked doubles piece.
* `src/composer/block_encoding/similarity_sandwich.py` no longer
  rejects generators with nonzero singles. Its dense reference path now
  uses the same masked compiled pool semantics as the sigma oracle,
  instead of masking only doubles while leaving singles always on.
* `tests/test_generator_exp.py` now checks that explicit singles
  channels are exposed, that the sigma oracle matches the masked full
  generator with singles present, and that the generator-exp oracle
  still matches dense `expm` on mixed singles+doubles cases.
* `tests/test_similarity_sandwich.py` now verifies that the compiled
  similarity sandwich accepts nonzero singles, that the reported dense
  effective Hamiltonian matches the full masked-generator reference, and
  that mask-length validation is tied to the compiled generator pool.
## 2026-04-21 — Similarity sandwich now closes the Sec. IV.C `U_sigma` gap

* Re-audited Sec. IV.C Eq. (47)-(53) and Appendix C against the paper
  PDF. The decisive point is that Eq. (50) treats `U_sigma^(m)` as a
  unitary on the joint ancilla-system space, while Eq. (44) already
  assigns its ancilla-zero block directly to `e^{sigma(m)}` rather than
  to a further-scaled quantity. The previous repo state stopped one step
  short: the parity-split `cos`/`sin` construction produced a literal
  compiled block encoding of `e^{sigma} / 2`, and the outer sandwich
  used that raw object directly.
* `src/composer/block_encoding/generator_exp.py` now keeps that direct
  parity-split block encoding as the raw `circuit`, but also builds a
  paper-facing `unitary_circuit` by one round of oblivious amplitude
  amplification, `-W R W^dag R W`, with `R = 2|0_anc><0_anc| - I`.
  This stays within the compiled-object model: the amplified circuit is
  assembled from repeated `CircuitCall`s to the real generator-exp
  subcircuit plus explicit ancilla-zero reflections, rather than
  collapsing to a dense extracted system gate.
* `src/composer/block_encoding/similarity_sandwich.py` now uses that
  amplified `U_sigma` inside the outer oracle, so the returned circuit
  is the literal nested composition `U_sigma^dag W_H U_sigma` with the
  intended Sec. IV.C semantics. The reported
  `encoded_system_block_dense` is now claimed to satisfy
  `alpha_H * encoded_system_block_dense ~= e^{-sigma(m)} H e^{sigma(m)}`
  on the supported small-system scope, before any external application
  of the model-space projector. The projector remains external exactly
  as in Eq. (51)-(52), so `H_eff_dense` is still
  `P e^{-sigma(m)} H e^{sigma(m)} P`.
* `tests/test_similarity_sandwich.py` now fail if the implementation
  regresses to the weaker nested-block claim: they directly compare the
  returned circuit's ancilla-zero block against the dense unprojected
  target `e^{-sigma} H e^{sigma} / alpha_H`, verify the projected
  equality after applying `P`, and check that the outer oracle still
  calls the amplified compiled `U_sigma` subcircuit rather than a dense
  system-register shortcut.
* `README.md`, `PAPER_MAPPING.md`, and `ASSUMPTIONS.md` now state the
  stronger final truth explicitly: the Sec. IV.C similarity sandwich
  gap is closed on the repo's supported verification-scale cases, while
  the model-space projector remains an external post-processing step.

## 2026-04-21 — Similarity sandwich now returns the real nested outer oracle

* Re-audited Sec. IV.C Eq. (47)-(53) against the current
  `similarity_sandwich.py` path and the paper PDF. The prior repo state
  still flattened the outer sandwich back to a dense system-register
  `U_sigma` gate even after the generator-exp builder itself had been
  made hierarchical.
* `src/composer/block_encoding/similarity_sandwich.py` now builds the
  returned outer `circuit` as the literal nested compiled-object
  composition `U_sigma_block^dag W_H U_sigma_block`, using `CircuitCall`
  references to the real generator-exp oracle circuit and the real
  Hamiltonian oracle circuit on a shared ancilla layout. The outer
  object no longer uses a dense extracted `U_sigma` system gate as its
  main construction.
* The object contract is now explicit:
  `encoded_system_block_dense` is the exact ancilla-zero system block of
  that returned nested circuit, while `H_eff_dense` remains the dense
  paper target `P e^{-sigma(m)} H e^{sigma(m)} P` with the model-space
  projector kept external.
* The remaining gap is now stated precisely rather than blurred by the
  old shortcut: the current generator-exp object is still a block
  encoding of `e^{sigma(m)}`, not an ancilla-free system unitary, so the
  nested circuit's ancilla-zero block is not yet claimed to equal the
  paper target divided by `alpha_H`.
* `tests/test_similarity_sandwich.py` now rejects regressions back to
  the dense shortcut by requiring the outer sandwich to be composed of
  nested subcircuit calls whose compiled payload hashes match the real
  generator/Hamiltonian oracles. The tests also make the
  unprojected-vs-projected distinction exact: the returned circuit block
  must match its reported ancilla-zero block, and the dense projected
  paper target is verified separately.
* `examples/05_similarity_sandwich.py`, `README.md`,
  `PAPER_MAPPING.md`, and `ASSUMPTIONS.md` now describe the returned
  object honestly and show the compile-once mask re-dial workflow on the
  real nested outer sandwich.

## 2026-04-21 — Generator-exp hierarchy over compiled oracle calls

* Audited Sec. IV.B / Eq. (39)-(46) of the paper against the current
  `generator_exp.py` path. The sigma oracle already had literal
  `PREP_sigma` / `SELECT_sigma` structure, but the exponential builder
  still collapsed the QSP ladders with `_circuit_matrix(...)` and then
  rebuilt the Hermitianized `cos` / `sin` pieces and final LCU as dense
  full-width matrices.
* `src/composer/circuits/gate.py` now distinguishes dense primitive
  gates from two hierarchical operation types:
  `CircuitCall` for reusable compiled subcircuits and `SelectGate` for
  explicit two-branch multiplexing over compiled children. Their dense
  matrices are synthesized lazily only when the simulator / verification
  helpers ask for them.
* `src/composer/circuits/circuit.py` now includes the implementation
  mode and nested compiled hashes in `compiled_signature_hash()`, and
  `Circuit.resource_summary()` reports composite-gate /
  subcircuit-call / select-gate counts. This lets tests distinguish a
  real hierarchical oracle from a regressed dense placeholder.
* `src/composer/block_encoding/generator_exp.py` now builds the main
  exponentiation path compositionally:
  1. a fixed Wx-form oracle circuit around the compiled sigma oracle,
  2. repeated `CircuitCall`s to that oracle inside the `cos` / `sin`
     QSP ladders,
  3. explicit Hermitianization via `SelectGate(U, U^dag)`, and
  4. the final `cos +/- i sin` combination via one more `SelectGate`.
  The returned `GeneratorExpOracle` keeps these compiled layers
  explicitly (`wx_oracle_circuit`, `cos_qsp_circuit`,
  `sin_qsp_circuit`, `cos_oracle_circuit`, `sin_oracle_circuit`,
  `circuit`). Dense `W` / zero-block views are now lazy verification
  properties rather than eager construction intermediates.
* `tests/test_generator_exp.py` now fails if the generator-exp path
  regresses to a dense wrap: it checks that QSP query gates are real
  `CircuitCall`s to one reusable compiled oracle, that the
  Hermitianization/final-combination layers are `SelectGate`s over the
  expected child circuits, and that the logical resource summaries count
  those composite operations. The small-system ancilla-zero blocks still
  match dense `expm` references.
* `src/composer/qsp/chebyshev.py`, `README.md`, `PAPER_MAPPING.md`, and
  `ASSUMPTIONS.md` now describe the dense Chebyshev branch honestly as a
  reference helper only, not the main generator-exp implementation.

## 2026-04-21 — Paper-backed logical resource accounting and reproducible pytest imports

* Audited the paper directly against Sec. IV.C, Sec. V, and Appendix D
  (especially the logical selector-width / ancilla-budget discussion in
  the resource section and the compile-once scope note). The paper's
  emphasis is logical-oracle accounting tied to a fixed selector/adaptor
  scaffold, not fault-tolerant Toffoli counts.
* The repo previously claimed "resource counters in `lcu.py`", but in
  practice there was no first-class compiled-object accounting on the
  returned Hamiltonian oracle, sigma oracle, generator-exp oracle, or
  similarity sandwich. Users had to infer branch counts and QSP usage by
  re-reading builder internals, and the examples did not surface that
  information.
* `src/composer/circuits/circuit.py` now exposes
  `Circuit.resource_summary()`, which reports gate inventory,
  multi-qubit/full-width gate counts, maximum gate arity, qubit count,
  and the compiled-signature hash derived from the actual compiled gate
  list.
* `src/composer/block_encoding/lcu.py` now returns a real compiled
  `PREP_H` / `SELECT_H` / `PREP_H^dag` circuit together with nested
  `LCUResourceSummary` data: selector width, one-body vs Cholesky
  branch counts, compiled null-branch slot, and compiled gate
  inventory.
* `src/composer/block_encoding/generator_exp.py` now attaches
  `SigmaOracleResourceSummary` and `GeneratorExpResourceSummary` to the
  returned oracle objects. These summaries are derived from the built
  sigma circuit, the actual `cos` / `sin` QSP circuits, and the final
  exponential LCU, so the reported QSP query counts are counts of real
  compiled oracle uses rather than hand-maintained metadata.
* `src/composer/block_encoding/similarity_sandwich.py` now reports a
  nested `SimilaritySandwichResourceSummary` carrying the outer
  compiled circuit inventory, the number of `U_sigma` calls in that
  circuit, projector rank, and the underlying Hamiltonian/generator
  oracle summaries.
* The repo now supports the checkout-local dev workflow explicitly:
  `pyproject.toml` declares `pythonpath = ["src"]` for pytest, the
  example scripts bootstrap `src/` themselves when run by path, and a
  small top-level import shim makes repo-root `python -c "import
  composer"` work without a manual `PYTHONPATH` export. This fixes the
  clean-checkout test/import ergonomics that were previously implicit.
* `tests/test_foundation.py`, `tests/test_lcu.py`,
  `tests/test_generator_exp.py`, and
  `tests/test_similarity_sandwich.py` now fail if these summaries drift
  from the compiled objects they are supposed to describe.
* `examples/04_lcu_hamiltonian_h2.py` and
  `examples/05_similarity_sandwich.py` now print the compiled logical
  resource summaries so the demos expose the paper-relevant accounting
  directly.
* `README.md`, `PAPER_MAPPING.md`, and `ASSUMPTIONS.md` now state the
  scope precisely: the repo provides logical compiled-object accounting
  for the real constructions it returns, but still does not provide
  fault-tolerant synthesis estimates or exact selector-controlled
  two-qubit depth constants for the dense stand-ins.

## 2026-04-21 — Similarity sandwich lifted onto the real generator oracle

* Audited `block_encoding/similarity_sandwich.py`,
  `operators/mask.py`, `circuits/circuit.py`,
  `tests/test_similarity_sandwich.py`, and
  `examples/05_similarity_sandwich.py` against Sec. IV.C Eq. (47)-(53)
  plus the repo's earlier Sec. IV.B generator-oracle work.
* The old similarity-sandwich path still wrapped the Hamiltonian oracle
  with dense system-register surrogates for `e^{+-sigma}`. Its
  compile-once check was mostly "same topology hash", so it could not
  distinguish a fixed compiled oracle template from a shallow
  placeholder.
* `src/composer/block_encoding/similarity_sandwich.py` now replaces the
  old dense-Chebyshev `e^{+-sigma}` surrogate with the real compiled
  generator oracle/QSP output. The returned object carries that real
  generator oracle together with the fixed Hamiltonian oracle `W_H`, and
  the outer returned circuit uses the oracle-generated `U_sigma` system
  block around `W_H`. This is materially closer to the paper's
  `U_sigma^dag W_H U_sigma` construction while keeping the returned
  ancilla-zero block numerically literal for the current verification
  path. The returned `H_eff_dense` still keeps the model-space
  projector external, exactly as the paper states.
* The current generator oracle in the repo remains doubles-pool only.
  Instead of silently mixing nonzero singles back into a dense in-circuit
  surrogate, `build_similarity_sandwich(...)` now rejects generators
  with nonzero `t1`. The dense helper `effective_hamiltonian_dense(...)`
  still serves as the exact small-system reference path and continues to
  include singles.
* After checking the paper text directly, the remaining explicit outer-
  sandwich gap is now documented more precisely: a fully nested
  ancilla-resolved `U_sigma^dag W_H U_sigma` is still deferred because
  the current verification-scale oracle objects do not yet expose the
  required lifting/composition machinery while preserving the reported
  ancilla-zero block.
* `SimilaritySandwich.redial_mask(...)` now demonstrates the intended
  compile-once workflow more literally: it reuses the fixed compiled
  normalization / QSP structure and rebuilds only the mask-dependent
  PREP data and resulting `U_sigma` oracle on that template.
* `src/composer/circuits/circuit.py` now exposes
  `compiled_signature()` / `compiled_signature_hash()` so tests can
  assert a fixed ordered gate schedule, qubit support, and gate shapes,
  not just a coarse topology hash.
* `tests/test_similarity_sandwich.py` now checks:
  the doubles-only restriction, fixed compiled signatures across mask
  re-dialing, QSP phase reuse, PREP-only sigma-oracle changes on the
  fixed template, and ancilla-zero block behavior across masks for a
  low-ancilla literal sandwich instance.
* `examples/05_similarity_sandwich.py` now shows a genuine compile-once
  re-dialing workflow: build once on the full mask, re-dial to the
  selected mask on the same compiled template, and report the fixed
  compiled signature together with PREP-vs-SELECT behavior.

## 2026-04-20 — Sigma-pool oracle scaffolding (PREP / SELECT / null branch)

* Audited `block_encoding/generator_exp.py`, `operators/mask.py`, and
  the generator / similarity tests against Sec. IV.B Eq. (39)-(43) of
  the paper.
* The old generator-side path exposed only a dense Chebyshev surrogate
  for `e^{sigma}` and treated the mask/null branch as dense-side
  bookkeeping. There was no ancilla-resolved sigma oracle over the
  compiled pair-channel pool.
* `src/composer/block_encoding/generator_exp.py` now builds a real
  sigma-pool oracle over the fixed compiled doubles-channel list:
  selector width `ceil(log2(ell_sigma + 1))`, a mask-aware
  `PREP_sigma`, a multiplexed `SELECT_sigma`, and an explicit null
  branch. The returned object exposes the full oracle matrix/circuit,
  compiled branch weights, and the ancilla-zero system block
  `-i sigma_pool(m) / alpha_bar`.
* The null branch is now part of the oracle behavior rather than an
  ignored scalar field. The implementation realizes it as an exact
  zero-block branch so the residual weight keeps `alpha_bar` fixed
  without changing the encoded generator.
* `src/composer/operators/mask.py` now includes
  `with_compiled_alpha_bar(...)` and `compiled_weight_sum(...)` so
  mask residuals can be computed in the compiled branch-scale units used
  by the sigma oracle, not just raw selector counts.
* `tests/test_generator_exp.py` and `tests/test_similarity_sandwich.py`
  now fail if the sigma path regresses to a single dense placeholder:
  they check for real selector ancillas, PREP/SELECT gate structure,
  PREP-only mask updates, and null-branch participation in the compiled
  oracle.
* The final QSP ladder for `e^{sigma}` remains deferred. The repo still
  uses the dense Chebyshev surrogate for the exponential itself, but it
  now does so on top of a real generator-side oracle scaffold.

## 2026-04-20 — Sigma-hat channel audit and operator-level doubles-channel fix

* Audited `factorization/pair_svd.py`, `operators/{generator,rank_one}.py`,
  and the sigma-hat tests against Sec. II.C / Eq. (18)-(27) plus
  Definition 2 / Eq. (4)-(5) of the paper.
* The old path treated the pair-SVD pool mainly as a tensor-factor list
  and rebuilt dense doubles generators from that list. It also used the
  occupied factor with inconsistent semantics across tensor and operator
  views: the operator path followed `B[V] = sum V_ij^* a_j a_i`, but the
  direct tensor reconstruction path had been omitting that conjugation.
* `PairChannel` now stores channels in the paper's operator convention,
  exposes the actual amplitude tensor
  `sigma U[a,b] V[i,j]^*`, and supports explicit occupied/virtual
  embedding via `EmbeddedPairChannel`.
* `ClusterGenerator` now exposes a first-class embedded sigma-channel
  pool through `doubles_channels()` / `pair_rank_one_pool()`. Each
  channel can be turned directly into the paper's Def.-2 operator
  `L_s = omega_s B^dag[U^(s)] B[V^(s)]` and its anti-Hermitian ladder
  `L_s - L_s^dag`.
* `PairRankOne` now exposes `coefficient_tensor()`, `adjoint()`, and
  `dense_sigma_term()` so tests and later oracle code can verify and
  consume the operator object directly rather than only a dense matrix
  surrogate.
* `tests/test_generator_exp.py`, `tests/test_rank_one.py`, and the
  adjacent `tests/test_similarity_sandwich.py` were strengthened so
  they catch:
  wrong occupied-factor conjugation, missing occupied/virtual
  embeddings, failure of the channel-to-operator identity, and
  regression of the masked dense-reference path.

## 2026-04-20 — Appendix-E diagnostic audit corrections

* Audited `diagnostics/{mp2,subspace,mask_selection}.py`,
  `operators/mask.py`, `examples/{04_lcu_hamiltonian_h2,05_similarity_sandwich}.py`,
  and the H2 / diagnostics tests against Appendix E of the paper.
* `src/composer/diagnostics/subspace.py` now implements the paper's
  Eq. (E7)-(E8) rank-cumulative wAUC definition rather than the earlier
  best-match-per-channel surrogate. The remaining assumption is only the
  natural generalization from equal-rank manifolds to a shorter masked
  list, documented in `ASSUMPTIONS.md`.
* `src/composer/diagnostics/mask_selection.py` now ranks channels by the
  paper's MP2 ladder weight `w_s^MP2 = |omega_s|^2 ||U^(s)||_F^2 ||V^(s)||_F^2`
  and returns a binary selector mask. The previous sigma-weight mask was
  inconsistent with the paper's classical-mask semantics and with the
  repo's own `uniform_mask` convention.
* `src/composer/operators/mask.py::top_k_mask()` now also returns a
  binary selector mask, so the dense similarity-sandwich path keeps
  selected channels at their native singular values instead of squaring
  them accidentally.
* `src/composer/diagnostics/mp2.py` now documents the Eq. (E9) `1/4`
  convention explicitly and rejects near-zero MP2 denominators instead
  of producing silent infinities.
* `examples/04_lcu_hamiltonian_h2.py` no longer mislabels `NO` and `NV`
  as spatial-orbital counts. `examples/05_similarity_sandwich.py` is now
  a non-vacuous MP2-screening workflow on a small synthetic 3-occ/3-vir
  system; the shipped H2 dataset is too low-rank to showcase mask
  selection or wAUC meaningfully.
* `tests/test_mp2_wauc.py`, `tests/test_similarity_sandwich.py`, and
  `tests/test_end_to_end_h2.py` were strengthened so they now catch:
  wrong wAUC weighting, sigma-vs-selector-mask confusion, silent MP2
  denominator singularities, vacuous one-channel top-k checks, and
  example regressions.

## 2026-04-20 — Generator-exp oracle/QSP lift over the real sigma oracle

* Audited `block_encoding/generator_exp.py` and `qsp/{chebyshev,phases,qsvt_poly}.py`
  directly against Sec. IV.B Eq. (44)-(46) and Appendix C of the paper.
* The old main path for `e^{sigma}` bypassed the compiled sigma oracle:
  it took a dense anti-Hermitian matrix, normalized it, and applied the
  truncated Chebyshev series by dense matrix Clenshaw evaluation. That
  matched the target function numerically, but it was not the paper's
  oracle/QSP flow.
* `src/composer/block_encoding/generator_exp.py` now builds
  `e^{sigma}` from the actual `SigmaOracle` object. The implementation:
  1. converts the reflection-style sigma block encoding into the Wx
     scalar convention used by the repo's QSP utilities,
  2. synthesizes scalar phase lists for the parity-valid real targets
     `cos(alpha x)` and `sin(alpha x)`,
  3. runs those QSP sequences on the compiled sigma oracle,
  4. extracts the Hermitian target block explicitly via a block-level
     `(U + U^dag)/2` LCU, and
  5. combines the resulting real `cos` / `sin` block encodings into
     `e^{sigma}` by a final LCU.
* The remaining non-literal gap is explicit and documented: the paper
  presents a single complex QSP ladder for `e^{sigma}`, while the repo
  currently implements an equivalent parity-split realization using two
  scalar real-polynomial ladders plus one extra LCU ancilla. The dense
  Chebyshev path is still present, but only as `dense_generator_exp_reference(...)`.
* `tests/test_generator_exp.py` now checks the oracle-built
  `e^{sigma}` block against `scipy.linalg.expm` on small masked
  generators, checks the intermediate `cos` / `sin` blocks separately,
  and verifies that the QSP phase lists are mask-invariant when the
  compiled `alpha_bar` is held fixed.
* `tests/test_qsp.py` now keeps the scalar side scoped explicitly to
  parity-valid utilities, including a Chebyshev-basis wrapper test for
  scalar phase fitting.
* `README.md`, `PAPER_MAPPING.md`, and `ASSUMPTIONS.md` were updated so
  the repo now claims the oracle/QSP generator-exp construction it
  actually implements, and documents the remaining parity-split phase
  synthesis limitation precisely.

## 2026-04-20 — t2 / sigma path and generator-exponential audit corrections

* Audited `factorization/pair_svd.py`, `operators/generator.py`,
  `qsp/{chebyshev,phases,qsvt_poly}.py`, and
  `block_encoding/generator_exp.py` against the paper-facing claims in
  the repo.
* The pair-SVD path is now stated and tested at the right level:
  `pair_svd_decompose()` performs the pair-basis SVD of `t2`, and
  `ClusterGenerator` now exposes the explicit operator reconstruction
  `T2 = sum_mu sigma_mu B^dag[U_mu] B[V_mu]` and
  `sigma_doubles = T2 - T2^dag`. New tests lock in both the tensor
  reconstruction and the channel-to-operator identity.
* The QSP/exponential layer is now described honestly. The scalar Wx-QSP
  primitives and the `x -> x^2` reference schedule remain real QSP
  utilities, but `generator_exp.py` does not synthesize a QSP/QSVT
  circuit for `e^{sigma}`. It evaluates the truncated Chebyshev series
  on the dense normalized matrix directly.
* Tests now distinguish approximation quality from implementation type:
  `tests/test_generator_exp.py` checks both closeness to
  `scipy.linalg.expm` and exact agreement with the truncated Chebyshev
  series, including a case where the surrogate is intentionally
  measurably different from the exact exponential.
* `tests/test_qsp.py` now rejects invalid scalar-QSP targets with the
  wrong parity or `|P(x)| > 1` on `[-1, 1]`.
* `ASSUMPTIONS.md`, `PAPER_MAPPING.md`, `README.md`, and package/module
  docstrings were updated so the repo no longer overclaims a full
  generator-exponential QSP implementation.

## 2026-04-20 — Appendix B.2 literalization pass

* Re-audited Lemma 2 directly against Appendix B.2 in the paper.
* `src/composer/block_encoding/cholesky_channel.py` no longer returns
  only a dense `A^2` identity surrogate on the main Lemma-2 path.
  `cholesky_channel_block_encoding(L)` now builds and returns the
  explicit dense register structure from the proof:
  retained-mode index register `I`, occupation flag `f`, and degree-2
  signal qubit `g`.
* Each retained branch now encodes the rotated occupation operator
  `n_{mu xi}` by conjugating the pivot occupation projector with the
  Appendix-A.1 number-conserving one-electron ladder for the returned
  eigenmode, rather than collapsing the whole channel into one dense
  reflection block encoding up front.
* `src/composer/ladders/one_electron.py` now exposes
  `mode_rotation_unitary(u)`, the dense unitary of the audited
  number-conserving ladder, so Lemma 2 can reuse the ladder itself as
  the rotated-basis primitive.
* `src/composer/qsp/qsvt_poly.py` now contains the exact constant-depth
  ancilla-projector degree-2 transform used by the Lemma-2 builder.
  The scalar 3-phase Wx schedule is still kept as a reference
  derivation, but it is no longer the only concrete degree-2 object in
  the repo.
* `tests/test_cholesky_channel_be.py` now checks the construction at
  three levels:
  returned branch blocks equal the rotated occupations,
  `PREP-SELECT-PREP^dag` returns `O_mu / Gamma_mu`,
  and the final signal-added unitary has top-left block
  `O_mu^2 / Gamma_mu^2`.
* `README.md`, `PAPER_MAPPING.md`, and `ASSUMPTIONS.md` were updated so
  the repo now claims a rotated-mode/occupation-flag Lemma-2
  construction honestly. The remaining non-literal gap is only that the
  exact degree-2 step is synthesized as a dense ancilla-projector
  composition rather than by explicitly compiling the paper's phase-list
  QSVT ladder gate-by-gate.

## 2026-04-20 — Block-encoding audit corrections (Lemma 1 / Lemma 2 / Theorem 1)

* Audited `block_encoding/{bilinear,cholesky_channel,lcu}.py` and the
  corresponding tests against Sec. IV.A and Appendix B of the paper.
* `src/composer/block_encoding/bilinear.py` now states the exact scope
  honestly: the implemented top-left block is a projector sandwich on
  full Fock space and reduces to the dyad `|u><v|` only on `H_{N=1}`.
  The docs no longer describe it as the literal Appendix B.1
  vacuum-projector circuit.
* `src/composer/block_encoding/cholesky_channel.py` now exposes the
  exact Lemma-2 operator identity. That historical audit landed the
  exact `O_mu^2 / Gamma_mu^2` top-left block; the later Appendix-B.2
  literalization pass replaced the main Lemma-2 path with an explicit
  rotated-mode / occupation-flag / signal-qubit construction while
  keeping the compact helpers as compatibility wrappers.
* `src/composer/block_encoding/lcu.py` now inherits that exactness:
  `build_hamiltonian_block_encoding(pool).top_left_block()` satisfies
  `alpha * block == H` on the supported real-integral scope, rather
  than only after taking the Hermitian part.
* `tests/test_bilinear_be.py`, `tests/test_cholesky_channel_be.py`,
  `tests/test_lcu.py`, and `tests/test_end_to_end_h2.py` were
  strengthened so they validate the returned block encodings
  themselves at the top-left-block level.
* `ASSUMPTIONS.md`, `PAPER_MAPPING.md`, `README.md`, and
  `examples/04_lcu_hamiltonian_h2.py` were updated so the repo's truth
  claims now match the audited code.

## 2026-04-20 — Ladder/state-preparation audit (Sec. III, App. A)

* Audited `ladders/{givens,one_electron,two_electron,phased_pair_givens}.py`
  against Sec. III and Appendix A of the paper, with emphasis on the
  distinction between preparation-from-vacuum and the underlying
  number-conserving ladder unitaries.
* `src/composer/ladders/one_electron.py` now exposes both forms
  explicitly:
  `build_number_conserving_ladder(u)` implements `U_u`, while
  `build_ladder(u)` implements the preparation form `U_u X_r`.
  The old semantics were numerically correct but the public surface and
  docstrings blurred these two roles.
* `src/composer/ladders/phased_pair_givens.py` now matches Eq. (32)
  more faithfully: pair-Givens rotations support overlapping pivot and
  target pairs, rejecting only invalid pair labels or identical pairs.
  The earlier implementation incorrectly required all four orbital
  indices to be distinct.
* `src/composer/ladders/two_electron.py` now implements the paper's
  direct App. A.2 ladder over the antisymmetric pair basis:
  one phased pair-Givens block per non-pivot unordered pair, with the
  paper's classical angle recursion and an explicit separation between
  the number-conserving and preparation forms. The old
  rank-2-only orbital-rotation shortcut has been removed from the main
  path; `build_rank2_ladder()` remains as a compatibility alias that
  now dispatches to the full ladder.
* `tests/test_one_electron_ladder.py` and
  `tests/test_two_electron_ladder.py` were strengthened to catch:
  incorrect Givens/pair-Givens sign conventions, conflation of
  preparation with number-conserving basis rotation, wrong default or
  explicit pivot handling, failure to preserve particle number on
  spectator sectors, and incorrect behavior of pair-Givens on
  overlapping-pair subspaces.

## 2026-04-20 — Hamiltonian preprocessing audit (mean-field / Cholesky / pool)

* Audited the Sec. II.B preprocessing path against the paper's
  Eq. (10)-(17) conventions: physicist-order ERIs, mean-field shift,
  pivoted Cholesky, one-body eigendecomposition of `h_tilde`, and dense
  Hamiltonian reconstruction from the stored pool.
* `src/composer/operators/hamiltonian.py` now states the real scope
  explicitly. The low-level `cholesky_eri()` routine remains generic for
  complex Hermitian-PSD ERIs, but `build_pool_from_integrals()` rejects
  complex inputs because the current Eq. (13)-(17) pool representation
  uses the real-case `(1/2) sum_mu O_mu^2` channels.
* `HamiltonianPool.dense_matrix()` now reconstructs the dense reference
  from `h_tilde` plus the ERI implied by the stored Cholesky factors,
  rather than assuming `O_mu^2` is valid for every possible factor.
  This keeps the dense reference mathematically correct while the pool
  builder enforces the supported real-valued regime.
* `mean_field_shifted_h()` now validates the one-body input shape
  explicitly.
* `tests/test_cholesky.py` and `tests/test_end_to_end_h2.py` were
  strengthened to catch:
  physicist-vs-chemist convention mistakes, missing real-symmetric
  Cholesky structure, failure of the `h_tilde` eigendecomposition to
  reconstruct the shifted one-body matrix, and accidental use of the
  unsupported complex-general Hamiltonian-pool path.

## 2026-04-20 — JW / rank-one foundation audit against the paper

* Audited the foundational fermion and rank-one layers against Sec. II.A
  and Eq. (10) of the paper, with the paper treated as source of truth.
* `src/composer/operators/rank_one.py`:
  `ProjectedQuadraticRankOne` now matches Definition 3 / Eq. (8)-(9):
  `O = sum_r C_r n[u^(r)]` with `n[u] = a^dag[u] a[u]`, and the dense
  primitive is `O O^dag`. The previous implementation incorrectly used
  a particle-nonconserving `a^dag[u] + a[v]` channel, which was not the
  paper's operator.
* `PairRankOne` now supports explicit creation and annihilation orbital
  embeddings so the primitive can represent the paper's occupied/virtual
  pair spaces without forcing both antisymmetric tensors onto the same
  orbital index set. When the two local dimensions differ, the embedding
  must be supplied explicitly rather than guessed.
* `src/composer/utils/fermion.py` now validates orbital and basis-index
  bounds and exposes `jw_mode_number(u) = a^dag[u] a[u]`, the dense
  rotated-mode occupation operator used by Definition 3 and Lemma-2
  channel checks.
* `tests/test_foundation.py` and `tests/test_rank_one.py` were
  strengthened to lock in:
  LSB-first JW action on computational-basis states, explicit
  single-excitation ordering, physicist-order two-body signs on the
  two-electron sector, lexicographic antisymmetric-pair indexing,
  pair-rank-one distinct subspace embeddings, and projected-quadratic
  number conservation / PSD / direct dense correctness.

## 2026-04-20 — Foundation audit and truth-claim tightening

* `README.md`, `PAPER_MAPPING.md`, and `ASSUMPTIONS.md` now describe the
  current foundation truthfully: the repository is module-oriented, the
  H2 dataset and demos are repo assets rather than installed package
  data, App E diagnostics are described at their implemented scope, and
  the compile-once claim is stated as a topology-hash proxy rather than
  full mask-independent gate matrices.
* `ASSUMPTIONS.md` no longer claims a PySCF adapter exists. Larger-molecule
  integral ingestion is still only a follow-up.
* `src/composer/__init__.py` now states the intended public convention:
  the top-level package exports only `__version__`; functionality is
  imported from subpackages.
* `tests/test_foundation.py` is extended to check version/metadata
  consistency, the declared Python floor, whole-package importability,
  repo assets claimed by the README, and that `src/composer` only imports
  declared third-party runtime dependencies (`numpy`, `scipy`).
* Full suite after the audit: 105 collected tests green under local
  Python 3.12 (`.venv/bin/pytest -q`).

## 2026-04-20 — Diagnostics, end-to-end H2, and summary

* `diagnostics/mp2.py`: MP2 doubles amplitudes (Eq E9) and the closed-form
  MP2 correlation energy. Returns a `(NV, NV, NO, NO)` complex tensor
  antisymmetric in both index pairs, consumable directly by
  `factorization.pair_svd.pair_svd_decompose`. The Eq. (E9) `1/4`
  convention is absorbed into the canonical antisymmetric tensor
  definition used by the repo's generator path.
* `diagnostics/subspace.py`: `channel_overlap_matrix`, `wauc` and
  `rdm1_drift`. `wauc` now follows Eq. (E7)-(E8)'s rank-cumulative
  overlap, with ASSUMPTION #15 narrowed to the shorter-list
  generalization used for masked subsets.
* `diagnostics/mask_selection.py`: one-shot cumulative-coverage
  selector-mask builder using the App-E.3 MP2 ladder weights, with the
  iterative / residual-greedy variant left as a follow-up.
* `tests/test_mp2_wauc.py` locks in antisymmetry, the energy formula,
  Eq. (E7)-(E8) rank weighting, App-E.3 MP2 ladder weights, and the
  cumulative-coverage selector behavior.
* `tests/test_similarity_sandwich.py` covers the correctness of the
  dense sandwich, the external role of the model-space projector, the
  compile-once structural invariance checks across masks, and mask
  utilities.
* Repository asset `data/h2_sto3g_integrals.npz` is generated by `data/build_h2_integrals.py`
  from textbook H2 / STO-3G integrals at R = 1.4 bohr. Spin-orbital
  ordering: 2 spatial * (alpha, beta) = 4 spin-orbitals.
* `examples/04_lcu_hamiltonian_h2.py` runs the full pipeline end-to-end
  on H2, reports the Theorem-1 alpha, the reconstruction residual
  (1.3e-15 observed) and the N=2 ground state (-1.144 Ha vs. published
  FCI -1.137 Ha; the 7 mHa gap is rounding in the textbook integrals).
* `examples/05_similarity_sandwich.py` demonstrates the compile-once
  claim: two masks, same topology hash, different `H_eff`.
* `tests/test_end_to_end_h2.py` (3 tests) verifies the pipeline on the
  canned integrals.

### Follow-ups (deferred, documented)

* Iterative / residual-greedy mask selection variant (App E.3's
  non-"one-shot" path).
* `ErrorBudget` dataclass bundling the `eps_factor/block/mux/qsp`
  tallies exposed as ASSUMPTION #14.
* Depth-optimal PREP (replace the Möttönen cascade).
* PySCF adapter for larger molecules.

---

## 2026-04-20 — Similarity sandwich + mask

* `operators/mask.py`: `ChannelMask` dataclass + `uniform_mask` /
  `top_k_mask` constructors. Null-branch residual is modeled via
  `with_alpha_bar(alpha_bar)` (ASSUMPTION #10).
* `block_encoding/similarity_sandwich.py`: `ModelSpaceProjector`,
  `effective_hamiltonian_dense` and `build_similarity_sandwich`. The
  dense reference is `P @ expm(-sigma) @ H @ expm(sigma) @ P`, but the
  returned circuit itself encodes only the *unprojected* ancilla-zero
  system block `exp(-sigma) H exp(sigma) / alpha`; `P` is external, in
  direct agreement with Sec. IV.C of the paper. The circuit wraps the
  Theorem-1 block encoding of H with two mask-parametrized
  `GeneratorExp+/-` multi-qubit gates whose `kind` labels and qubit
  support are mask-independent. `Circuit.two_qubit_topology_hash()`
  returns the SHA-256 digest of the ordered `(kind, qubits)` tuples,
  providing only a structural compile-once proxy.

## 2026-04-20 — Similarity-sandwich audit tightening

* The paper-source audit of Sec. IV.C / App. D.3 tightened the central
  truth claim: the returned similarity-sandwich `circuit` does **not**
  encode `P^(m) e^{-sigma^(m)} H e^{sigma^(m)} P^(m)` directly. It
  encodes the unprojected ancilla-zero block
  `e^{-sigma^(m)} H e^{sigma^(m)} / alpha`, while the model-space
  projector `P^(m)` remains external and is applied only in the dense
  reported `H_eff_dense`.
* `SimilaritySandwich` now exposes `encoded_system_block_dense` so the
  returned object states explicitly what the circuit encodes before
  model-space restriction.
* Mask semantics are now strict: `ChannelMask.weights` must cover the
  full pair-rank-one pool length. The previous implicit "missing
  entries mean weight 1" behavior was removed as too loose for the
  paper's full-pool mask definition.
* The null branch is documented honestly: `null_weight` is retained as
  paper-faithful normalization bookkeeping, but the current dense
  generator placeholder does not consume it numerically because the repo
  has not synthesized the generator PREP/SELECT/QSP stack.
* `tests/test_similarity_sandwich.py` now verifies the circuit's
  ancilla-zero block explicitly, the external role of `P^(m)`, strict
  mask-length validation, null-branch bookkeeping semantics, and
  compile-once structure beyond the topology hash alone.

---

## 2026-04-20 — sigma-hat pool + e^sigma block encoding

* `factorization/pair_svd.py` unfolds an antisymmetric `t2` into the
  `(NV 2) x (NO 2)` pair matrix, SVDs it, and returns `PairChannel`
  entries with pair-normalized `U_mu` / `V_mu`.
* `operators/generator.py` builds `T - T^dag` on Fock space for
  verification, plus the `pair_rank_one_pool()` view.
* `block_encoding/generator_exp.py` implements the QSP exponential via
  the Chebyshev truncation of `e^{i kappa x}` applied to
  `A = -i sigma / kappa`; Clenshaw recurrence evaluates the matrix
  polynomial. Verified against `scipy.linalg.expm(sigma)` to `< 1e-8`.

---

## 2026-04-20 — Theorem 1 LCU + Lemma 2 channel

* `block_encoding/cholesky_channel.py`: the current main Lemma-2 path
  is the explicit Appendix-B.2-style rotated-mode construction with an
  index register, occupation flag, and one signal qubit for the exact
  degree-2 transform. The older compact reflection helper and scalar
  3-phase `x -> x^2` schedule are retained only as compatibility /
  reference utilities.
* `block_encoding/lcu.py`: `build_hamiltonian_block_encoding(pool)`
  returns a full Theorem-1 `W = PREP^dag SELECT PREP` plus a
  `top_left_block()` accessor whose exact block * alpha equals
  `h_tilde + (1/2) sum O_mu^2`. Weights are `(|e_k|, 0.5 * alpha_O^2)`
  for each one-body eigenchannel and each Cholesky channel; signs of
  the one-body entries are absorbed into a selector `Z`-phase gate.

---

## 2026-04-20 — Ladder primitives + QSP plumbing

* `qsp/chebyshev.py`: Jacobi-Anger coefficients, cos/sin truncations,
  `recommended_degree(alpha, eps)` using the standard
  `ceil(2 * (e*alpha/2 + log(2/eps)))` bound.
* `qsp/phases.py`: Wx-convention QSP, symmetric-phase optimizer
  (L-BFGS-B on a real-coefficient target polynomial); verified on
  degree-<=20 cosine / sine approximants.
* `qsp/qsvt_poly.py`: exact 3-phase scalar schedule for `x -> x^2`
  derived in the module docstring (`phi_0 + phi_1 + phi_2 = 0`,
  `phi_0 - phi_1 + phi_2 = pi/2`), plus the exact ancilla-projector
  degree-2 transform now used by the returned Lemma-2 block encoding.
* `ladders/{givens,phased_pair_givens,one_electron,two_electron}.py`:
  SO(2) Givens plus direct one- and two-electron ladders. The current
  audited implementation distinguishes the number-conserving forms
  (`U_u`) from the preparation forms (`U_u X_r`, `U_u X_r X_s`) and
  verifies both against explicit statevector targets.
* `block_encoding/bilinear.py`: Lemma 1 is implemented as the exact
  projector sandwich
  `W = (I ⊗ U_u)(X_a ⊗ I)CNOT_{q0→a}(I ⊗ U_v^dag)`.
  Its top-left block reduces to `|u><v|` only on the `N=1` sector.
  `orbital_rotation_first_column(u)` builds a QR-extended unitary
  with first column equal to `u`.

---

## 2026-04-19 — Scaffolding + Jordan-Wigner foundation

* Repo layout matches the plan: `src/composer/{operators,factorization,
  ladders,block_encoding,qsp,circuits,diagnostics,utils}` with
  `__init__.py` in every subpackage.
* `pyproject.toml` is editable-install-ready; runtime deps are
  `numpy>=1.24`, `scipy>=1.10`; dev extras `pytest>=7.0`.
* No qiskit / openfermion / pyscf dependency. All verification uses the
  in-repo statevector simulator.
* `src/composer/utils/fermion.py` implements the Jordan-Wigner
  mapping: creation/annihilation/number operators, one- and two-body
  matrices in physicist-order, Slater-determinant index with
  fermionic phase, and helpers for Lemma 1's `N=1` verification. The
  bit convention is LSB-first (qubit 0 = bit 0 of the basis index).
* Four documentation stubs populated: `README.md`, `ASSUMPTIONS.md`,
  `PAPER_MAPPING.md`, `IMPLEMENTATION_LOG.md` (this file).

---

## Honest summary — done / partial / follow-up

**Fully implemented + numerically verified on small systems:**
- Supported reference workflow on Python `3.10+`: editable install with
  dev dependencies, repo-root imports/examples, full pytest suite, and
  the two shipped examples. Closure validation snapshot:
  `173 passed in 744.76s`, no skips/xfails/warnings on Python `3.12.13`.
- Rank-one operator primitives (Def 1/2/3).
- Pivoted Cholesky + mean-field shift.
- One- and two-electron ladder state prep (residual < 1e-10).
- Pair-SVD factorization of `t2` (exact reconstruction).
- Cluster generator `sigma = T - T^dag` (anti-Hermitian to 1e-10).
- Lemma 1 bilinear block encoding (top-left block = dyad on N=1 sector).
- Lemma 2 projected-quadratic block encoding via explicit
  rotated-mode PREP/SELECT plus an exact degree-2 ancilla-projector
  transform (the 3-phase scalar `x -> x^2` schedule is retained only as
  reference data).
- Theorem 1 LCU block encoding (alpha * top_left == H_dense to 1e-8).
- Generator-exp oracle/QSP path over the real sigma oracle
  (small-system agreement with `scipy.linalg.expm` verified at the
  current test tolerances), plus the separate dense Chebyshev reference
  branch.
- Similarity sandwich and compile-once compiled-signature invariance
  under mask re-dialing.
- Logical compiled-object resource/accounting summaries for the
  Hamiltonian oracle, sigma oracle, generator-exp oracle, and outer
  similarity sandwich.
- ChannelMask with null-branch `with_alpha_bar` utility.
- MP2 amplitudes and the closed-form MP2 energy.
- wAUC + channel-overlap diagnostic.
- One-shot cumulative-coverage mask selection.
- End-to-end H2 / STO-3G pipeline.

**Partial (interpretation documented but narrower than paper scope):**
- `wAUC` follows Eq. (E7)-(E8), but `wauc(reference, truncated)`
  generalizes the equal-rank paper formula to a shorter masked list by
  comparing each reference rank ``r`` against the available truncated
  prefix `min(r, |truncated|)` (ASSUMPTION #15).
- Lemma 1's dyad check is performed on the `N=1` sector only
  (ASSUMPTION #3). Many-electron correctness is folded into the
  Theorem 1 test.
- The `GeneratorExp+/-` gates in the similarity-sandwich circuit are
  materialized as dense multi-qubit unitaries with a mask-independent
  `kind` label — enough to verify the compile-once topology hash
  (ASSUMPTION #13), but not a two-qubit-decomposed implementation.
  Hardware routing is explicitly out of scope.

**Follow-ups (flagged, not blocking):**
- `ErrorBudget` dataclass formalizing ASSUMPTION #14.
- Iterative / residual-greedy mask-selection variant.
- Depth-optimal PREP (replace Möttönen cascade).
- PySCF adapter for bigger molecules behind optional import.
- Fault-tolerant resource counting (paper explicitly defers, Sec V.d).
- Runtime optimization of the generator/QSP verification path. The
  current supported runs are clean and reproducible, but
  `examples/05_similarity_sandwich.py` and the full pytest suite take
  minutes rather than seconds because they perform repeated scalar QSP
  phase fitting and dense small-system verification.
