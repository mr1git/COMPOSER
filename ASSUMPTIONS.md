# ASSUMPTIONS

Every implementation choice made where the paper leaves room for
interpretation, with a pointer to where the choice appears in code.

## 1. Fermion-to-qubit mapping — Jordan-Wigner

The paper says *"one qubit per spin-orbital (e.g., Jordan-Wigner or
parity)"* (Sec II.A). We use **Jordan-Wigner** throughout.

* Qubit ordering: qubit `p` stores the occupation of spin-orbital `p`;
  qubit `0` is the least-significant bit of the computational-basis
  index.
* JW string on annihilation/creation: `Z_0 Z_1 ... Z_{p-1}` on lower
  indices.
* Basis ordering used in dense reference tests: the one-electron basis
  is ordered by increasing orbital label `p`, so the basis vector
  `|p>` is the computational basis state with index `2**p`.
* Occupation operator `n_p = a_p^dag a_p` is the diagonal
  `diag(x_p)_{|x>}` — crucial for the literal interpretation of
  `n_{mu xi}` in Lemma 2 as a single-qubit bit flag after conjugation.
* Code anchor: [`src/composer/utils/fermion.py`](src/composer/utils/fermion.py).

## 2. Dense but ancilla-faithful realization of Lemma 2

Eq. (34) is now implemented with the paper's register structure:

* a retained-mode index register `I` of width `a_I = ceil(log2 R_mu)`;
* one occupation-flag qubit `f`;
* one degree-2 signal qubit `g`.

For each retained eigenmode `u^(mu xi)`, the code builds the rotated
occupation operator `n_{mu xi}` by conjugating the pivot occupation
projector `n_pivot` with the number-conserving one-electron ladder
`U_{u^(mu xi)}` derived from Appendix A.1, so the returned branch block
is literally a rotated-mode occupation flag rather than a dense
surrogate for the full sum `O_mu`. `PREP-SELECT-PREP^dag` over the
weights `sqrt(|lambda^(mu)_xi|)` then yields an exact dense block
encoding of `O_mu / Gamma_mu`.

The remaining abstraction is only in the final degree-2 step: instead
of compiling an explicit phase-parameterized QSVT ladder gate-by-gate,
the repo inserts the ancilla-zero projector between two uses of the
`O_mu / Gamma_mu` block encoding using one signal qubit. This is still
an exact constant-overhead degree-2 transform with the paper's ancilla
count and the same top-left block `O_mu^2 / Gamma_mu^2`.

* Code anchor: `src/composer/block_encoding/cholesky_channel.py`.

## 3. Dyad interpretation of Lemma 1 in tests

Lemma 1's operator-norm statement (Eq 33) is proven on the
single-excitation subspace (paper explicitly so). Our Lemma 1 tests
therefore separate two claims:

* on the full Fock space, the returned top-left block equals the
  implemented projector sandwich `U_u |1><1|_{q_0} U_v^dag`;
* after restricting to `H_{N=1}`, that block equals the dyad
  `|u><v|` up to `alpha`.

The repo no longer phrases this as a full-Fock block encoding of the
one-body operator `a^dag[u] a[v]`.

* Code anchor: `tests/test_bilinear_be.py`.

## 4. Bilinear adaptor normalization — alpha = |lambda|

The paper requires `alpha >= |lambda|` (Eq 33). We pick
`alpha = |lambda_s|`, the smallest admissible, so the top-left block
equals the un-scaled dyad `|u><v|` (up to `sign(lambda)`). Phases are
absorbed into `u`/`v` so `lambda in R_{>=0}` by convention.

* Code anchor: `src/composer/block_encoding/bilinear.py`.

## 5. Singles + pair adaptor choices for sigma-hat anti-Hermitian generators

The paper's generator pool in this repo is represented as

* explicit singles channels `t_ai a_a^dag a_i`, followed by
* doubles channels
  `L_mu = sigma_mu B^dag[U_mu] B[V_mu]`, with `U_mu` on the virtual
  space and `V_mu` on the occupied space.

The current repo exposes the doubles channels as explicit embedded
objects carrying the occupied/virtual orbital labels needed by later
SELECT/PREP logic. The stored occupied factor follows Definition 2's
operator convention, so the amplitude tensor identity is
`t2[a,b,i,j] = sum_mu sigma_mu U_mu[a,b] V_mu[i,j]^*`, and the dense
verification path checks the matching operator identity
`T2 = sum_mu sigma_mu B^dag[U_mu] B[V_mu]`.

For singles, the paper says they are "inherently rank-one and may be
treated analogously." The repo now compiles them into the sigma oracle
as explicit branches, but not via the Lemma-1 dyad gadget: Lemma 1 is
only verified on the `N=1` sector, while the generator oracle needs the
full Fock-space operator `-i(L - L^dag)`. We therefore use the exact
full-Fock rotated-mode one-body block encoding returned by
`build_hermitian_one_body_block_encoding(...)` for each singles branch:
`PREP_O^dag SELECT_O PREP_O` over retained one-body eigenmodes, using
the same occupation-flag gadget as the Lemma-2 channel builder. This is
the strongest literal realization currently available on the repo's
gate/circuit model.

For doubles, the repo now compiles each embedded channel
`L_s = sigma_s B^dag[U_s] B[V_s]` through an explicit channel-local
internal adaptor. Concretely, `generator_exp.py` flattens the pair
coefficients `U_s[a,b] V_s[i,j]^*` onto the canonical pair-pair basis,
prepares their magnitudes with an internal `PREP_pair`, and applies a
pair-index-controlled `SELECT_pair` whose branches are the canonical
Hermitian four-orbital generators
`-i(e^{iphi_abij} a_a^dag a_b^dag a_j a_i - h.c.)`. The returned
branch adaptor therefore matches the full Fock-space doubles sigma term
exactly, and the sigma oracle now reuses that explicit branch object
instead of wrapping the entire channel in one dense full-Fock Hermitian
subencoding.

What is still deferred is the more compressed paper-style Eq. (37)
pair-state adaptor for each doubles channel, where the pair factors
would be absorbed into one ancilla-resolved channel-local state-prep /
dyad gadget rather than flattened into an internal canonical pair-basis
selector. The optional further wedge-factor decomposition of
Eq. (20)-(24) is also still deferred. The explicit doubles channel pool
itself remains at the Eq. (25)-(27) `B^dag[U] B[V]` level.

* Code anchors: `src/composer/factorization/pair_svd.py`,
  `src/composer/operators/generator.py`,
  `src/composer/block_encoding/generator_exp.py`.

## 6. QSP convention — Wx / reflection (Gilyen 2019)

We use the Wx convention of Gilyen, Su, Low, Wiebe (2019), which the
paper cites. `qsp/phases.py` implements scalar Wx signal/unitary
evaluation and a numerical phase fit for bounded real polynomials via
L-BFGS-B on a sample grid. The solver now keeps Chebyshev-basis targets
in Chebyshev form during compilation, and the generator-exponentiation
path now consumes a compiled exponential schedule object rather than
issuing two ad hoc trig-specific fits at the generator layer.

This is still a verification-scale scalar phase-synthesis utility
rather than the paper's production complex QSP phase compiler: the
current scalar circuit model exposes the ancilla-zero top-left
polynomial of one Wx ladder, and that polynomial has definite parity.
Since `exp(-i alpha x)` has both even and odd Chebyshev sectors, the
compiler now records the direct complex target, explicitly marks one
single ladder as infeasible on this model, and only then resolves the
target into real parity-valid fallback branches.

* Code anchor: `src/composer/qsp/phases.py`.

## 7. Chebyshev truncation for exp(-i alpha x)

Degree chosen as `d = ceil(2 (e alpha/2 + log(2/eps)))`, the standard
Jacobi-Anger bound; equivalent up to constants to the paper's Eq (C4).
In the current repo this degree now first compiles the direct complex
Chebyshev target `exp(-i alpha x)`, records its truncation metadata on
the returned exponential phase schedule, and then controls the scalar
`cos` / `sin` branch targets that drive the oracle-level
generator-exponential construction. A dense matrix Chebyshev evaluation
is still retained, but only as a numerical reference branch for direct
dense-matrix input, while the main `build_generator_exp_oracle(...)`
path keeps the oracle/QSP/LCU hierarchy explicit and synthesizes dense
matrices only when verification properties are queried.

What is still not literal is the paper's "single QSP ladder suffices"
presentation: the repo currently realizes `e^{sigma}` as two parity
compatible real QSP ladders (`cos`, `sin`) plus one final LCU. The gap
is therefore narrower but not closed: the compiler now begins from the
paper's direct complex target, but the resolved ladder synthesis still
uses the structured parity split because the current Wx/top-left model
only admits definite-parity scalar ladders. Closing that remaining gap
would require a more general direct complex/Laurent QSP phase
factorization path than the repo currently implements.

* Code anchor: `src/composer/qsp/chebyshev.py`.

## 8. Exact x -> x^2 channel identity for Lemma 2

Paper specifies a fixed degree-2 QSVT/QSP step implementing `x -> x^2`
(App B.2). The raw scalar Wx schedule in `qsvt_poly.py` is kept as a
reference derivation, but the returned Lemma-2 channel now uses an
explicit signal-qubit projector gadget to realize the exact degree-2
map on the already-constructed `O_mu / Gamma_mu` block encoding. In
other words: the repo no longer claims only a dense functional-calculus
shortcut for Lemma 2; it returns an exact ancilla-structured block
encoding whose top-left block is `O_mu^2 / Gamma_mu^2`.

Implementation detail: zero eigenmodes are dropped from the retained
rank `R_mu` using `spectral_tol = 1e-12`, and the remaining eigenmodes
are ordered by descending `|lambda^(mu)_xi|` so the returned branch list
and binary index width are deterministic in tests.

* Code anchors: `src/composer/qsp/qsvt_poly.py`,
  `src/composer/block_encoding/cholesky_channel.py`.

## 9. Binary selector register

Width `a_H = ceil(log2 ell_H)` as stated in Theorem 1 and Table I.
The repo now represents recurring selector-side oracle primitives
structurally:

* `PREP_H`, `PREP_sigma`, `PREP_pair`, and the one-qubit `PREP_*`
  layers used in the generator-exponential LCU path are synthesized as
  state-preparation primitives whose contract is only the prepared
  amplitude vector on `|0...0>`.
* `SELECT_H`, `SELECT_sigma`, and `SELECT_pair` are represented as
  explicit compiled multiplexors over branch subcircuits, with fixed
  selector width and fixed branch ordering.
* oblivious-amplitude-amplification reflections are represented as
  explicit ancilla-zero reflection primitives rather than as dense
  full-width diagonal gates.

The dense simulator still verifies these objects by synthesizing their
matrices lazily, but the main circuit representation no longer stores
the reusable oracle scaffold only as one dense PREP blob plus one dense
SELECT blob. In particular, the Theorem-1 Hamiltonian object now keeps
the widened branch workspace in the compiled circuit and computes its
paper-facing ancilla-zero system block directly from that circuit. The
full dense `W_H` matrix is still available, but only as a lazy
verification property rather than as an eagerly materialized part of
the main scalable path.

What is still deferred is the lower-level gate-set decomposition of
those generic primitives themselves. PREP is still the small-system
stand-in for a Moettoenen-style `R_y` cascade, and the multiplexed
SELECT objects do not yet compile down to an explicit selector-routing
tree of elementary controlled gates.

* Code anchor: `src/composer/block_encoding/lcu.py`.

## 10. Null branch for mask-aware constant alpha-bar

Eq (40) mentions a null selector branch that holds `alpha-bar'`
constant across masks. The current repo now implements this in the
generator-side sigma oracle itself: selector width is
`ceil(log2(ell_sigma^pool + 1))`, `PREP_sigma` is parameterized over the
fixed compiled generator pool (singles first, then doubles), and the
null residual is carried by an explicit zero-block branch in
`SELECT_sigma`. The helper
`ChannelMask.with_compiled_alpha_bar(...)` fills `null_weight` in the
compiled branch-weight units used by the oracle.

* Code anchor: `src/composer/operators/mask.py`,
  `src/composer/block_encoding/generator_exp.py`.

## 11. Model-space projector P^(m)

Paper leaves this user-specified. We implement it as a
computational-basis projector onto a user-supplied list of Slater
determinants (an active-space determinant set, the first concrete
example the paper gives). Following Sec. IV.C exactly, this projector
is **not** implemented coherently inside the returned oracle: the
oracle block is first constructed on the full working Fock sector, and
`P^(m)` is applied only when reporting / comparing the dense effective
Hamiltonian block on the chosen model space. The current sandwich path
now supports the same compiled singles+doubles generator pool as the
Sec. IV.B sigma oracle, and it closes the prior generator-side gap by
applying one round of oblivious amplitude amplification to the
parity-split `e^{sigma} / 2` block encoding produced in
`generator_exp.py`. The returned outer circuit is therefore the literal
nested compiled-object composition `U_sigma^dag W_H U_sigma`, where the
similarity-side `U_sigma` subcircuit has ancilla-zero block
`e^{sigma(m)}` up to the generator/QSP approximation error. The exact
ancilla-zero block of the returned outer circuit is reported as
`encoded_system_block_dense`, and the intended paper claim is now that
`alpha_H * encoded_system_block_dense` matches the unprojected operator
`e^{-sigma(m)} H e^{sigma(m)}` on the supported verification-scale
systems. The returned `H_eff_dense` remains the dense paper target
`P^(m) e^{-sigma(m)} H e^{sigma(m)} P^(m)` with the projector kept
external.

* Code anchor: `src/composer/block_encoding/similarity_sandwich.py`.

## 12. Integrals

No PySCF is required. The repository includes a hand-derived
`data/h2_sto3g_integrals.npz` (2 spatial = 4 spin orbitals, values
cross-checked against published HF/FCI energies) for the end-to-end
test. Random Gaussian integrals are used for scaling unit tests. A
PySCF adapter is **not** implemented in the current repo; bigger-molecule
integral generation remains a documented follow-up.

## 13. Compile-once verification via circuit structure fingerprints

The compile-once claim (Sec V) is a design principle: changing the
mask must not change the fixed compiled selector widths, gate ordering,
or oracle/QSP schedule used by the current circuit model. We now track
this with a stronger ordered full-gate signature
`(kind, qubits, matrix.shape, implementation_tag, nested hashes)` via
`Circuit.compiled_signature_hash()`, alongside the older multi-qubit
topology hash. This distinguishes dense primitives from hierarchical
subcircuit calls / branch selects and lets tests reject regressions
back to a single dense placeholder gate. `tests/test_similarity_sandwich.py`
asserts invariance of the outer sandwich circuit, the sigma oracle, and
the QSP subcircuits across masks, while separately checking that the
allowed mask-dependent changes are confined to PREP data and the
resulting `U_sigma` matrices. For the similarity sandwich specifically,
the outer circuit must remain a hierarchy of nested oracle calls rather
than collapsing back to a dense `U_sigma` system-register placeholder.
The same compiled objects now also expose
logical resource summaries derived from those gate lists, so the
reported selector widths, branch counts, QSP query counts, and
composite-gate counts are tied to the real compiled circuits rather
than to separate bookkeeping.
This is still a structural compile-once claim rather than a statement
that every dense gate matrix is mask-independent.

* Code anchor: `src/composer/circuits/circuit.py`.

## 14. Error budget

App D.3's `eps_tot = eps_factor + eps_block + eps_mux + eps_qsp` is a
reporting decomposition. In this implementation the non-generator pieces
are exact for the verification-scale systems we test, while the
generator-exponential piece is an oracle/QSP construction whose scalar
targets are chosen from Jacobi-Anger truncations:

* `eps_factor = ||eri - sum_mu L^mu L^mu^T||_F` falls below the
  pivoted-Cholesky threshold (default `1e-10`).
* `eps_block` is zero for Lemma 1 (direct dyad) and for the x^2 QSVT
  of Lemma 2 (derived analytically in `qsp/qsvt_poly.py`).
* `eps_mux` is zero for our Möttönen PREP.
* `eps_qsp` is the combination of:
  the Jacobi-Anger truncation error for the scalar `cos` / `sin`
  targets, bounded by `recommended_degree(alpha, eps)` in
  `qsp/chebyshev.py`, and the verification-scale numerical phase-fit
  residual from `solve_phases_real_chebyshev(...)`.

A dedicated `ErrorBudget` dataclass is a documented follow-up
(`IMPLEMENTATION_LOG.md`).

## 15. wAUC definition

App E.2 defines

    ov(r) = (1 / r) || B_r^dagger \tilde{B}_r ||_F^2,
    wAUC(R) = sum_{r=1..R} w_r ov(r),
    w_r = s_r^2 / sum_{k=1..R} s_k^2

for singular-value ordered manifolds of equal rank. The repo's
assumption is narrower: ``wauc(reference, truncated)`` compares a full
reference manifold against a possibly shorter masked/truncated list by
evaluating Eq. (E7) with the available truncated prefix at each rank,

    ov_repo(r) = (1 / r) || B_r^dagger \tilde{B}_{min(r, |truncated|)} ||_F^2.

This keeps the paper's cumulative-rank weighting, makes
`wauc(reference, reference) == 1`, and penalizes missing higher-rank
channels in a shorter selected list. The implementation assumes the
input channel lists are already singular-value ordered, as produced by
`pair_svd_decompose`.

* Code anchor: `src/composer/diagnostics/subspace.py`.

## 16. Hamiltonian preprocessing scope — real-valued integrals only

Sec. II.B's Eq. (13)-(17) Hamiltonian pool is currently implemented for
the standard **real-orbital** electronic-structure case only: `h` and
`<pq|rs>` must be real-valued, and the pivoted-Cholesky factors used by
`HamiltonianPool` must be real symmetric matrices `L^mu`.

The low-level primitive `factorization.cholesky.cholesky_eri` is more
general and accepts complex Hermitian-PSD matricized ERIs, but the
higher-level pool / Theorem-1 path rejects that case explicitly. The
reason is structural: the current pool stores channels in the paper's
real-case form `(1/2) sum_mu O_mu^2`; for generic complex factors, the
same tensor identity is not represented by that object without an
extended channel decomposition.

* Code anchors: `src/composer/factorization/cholesky.py`,
  `src/composer/operators/hamiltonian.py`,
  `tests/test_cholesky.py`.

## 17. Resource reporting is layered: logical summary, compiled synthesis view, optional backend view

Appendix D discusses selector-controlled depth scaling and ancilla
budgets at the logical-oracle level. The repo now exposes `resources`
summaries on the returned `LCUBlockEncoding`, `SigmaOracle`,
`GeneratorExpOracle`, and `SimilaritySandwich` objects, plus a generic
`Circuit.resource_summary()`. On top of that, the circuit layer now also
exposes `Circuit.resource_report(...)` /
`composer.circuits.resource_report(...)`.

The reporting contract is intentionally split:

* the existing object-level `resources` fields and
  `Circuit.resource_summary()` remain the concise logical compiled-object
  summaries, and
* `Circuit.resource_report(system_width=...)` adds a recursive compiled
  synthesis view over the actual structural circuit: ancilla count,
  recursive gate-family inventory, selector/control-state overhead, and
  dense-leaf counts.
  It also returns `dense_leaf_gate_count_by_kind`, so the remaining
  low-level dense leaves are named explicitly rather than hidden behind
  one aggregate count.

When the optional backend/export layer is installed, the same report API
can also add a third view,
`Circuit.resource_report(..., backend="qiskit", basis_gates=(...))`,
which reports SDK/export-side instruction families, depth, and
two-qubit counts when the chosen export/transpilation basis makes those
quantities concrete.

The directly-supported quantities are therefore:

* selector widths and total ancilla counts,
* active vs compiled branch counts on returned oracle objects,
* top-level and recursive gate inventories grouped by compiled gate kind,
* recursive selector/control-state overhead on compiled circuits,
* dense-leaf counts that make remaining verification-only leaves visible,
* QSP query counts obtained from the built `cos` / `sin` circuits, and
* outer-sandwich `U_sigma` call counts and projector rank,
* plus optional SDK/export depth and two-qubit counts when representable.

What they do **not** report is equally important:

* no Clifford+T / Toffoli / surface-code estimates,
* no hardware-routing or pulse-level recompilation costs,
* no claim that optional backend/export counts are paper semantics, and
* no exact selector-controlled two-qubit depth constants for the dense
  multi-qubit stand-ins still present in the current repo.

So the repo now matches the paper more closely at the logical
compile-once/accounting layer, while fault-tolerant resource estimation
remains a documented follow-up.

* Code anchors: `src/composer/circuits/circuit.py`,
  `src/composer/circuits/resources.py`,
  `src/composer/block_encoding/lcu.py`,
  `src/composer/block_encoding/generator_exp.py`,
  `src/composer/block_encoding/similarity_sandwich.py`.

## 18. Supported runtime floor and checkout-local import path

The repository's supported runtime floor is the one declared in
`pyproject.toml`: **Python >= 3.10**. Some modules may still import on
older interpreters, but that is not treated as supported behavior and
the foundation suite now reports such environments explicitly instead of
failing due to version-specific stdlib introspection details.

The supported reference workflow is one activated Python `3.10+`
virtualenv with an editable install plus dev dependencies:

* `python -m pip install -e '.[dev]'`
* `python -m pytest`
* `python examples/04_lcu_hamiltonian_h2.py`
* `python examples/05_similarity_sandwich.py`

The optional backend/export workflow extends that same environment with:

* `python -m pip install -e '.[qiskit]'`
* `python examples/06_qiskit_export_h2.py`

The checkout-local import conveniences below are intentionally kept, but
they are not a second independent support contract.

For checkout-local use, the repo intentionally supports the `src`
layout without a manual `PYTHONPATH` export:

* `composer/__init__.py` is a repo-root shim that forwards
  `import composer` to `src/composer` when Python starts from the
  repository root.
* `pyproject.toml` configures pytest with `pythonpath = ["src"]`.
* `examples/04_lcu_hamiltonian_h2.py` and
  `examples/05_similarity_sandwich.py` and
  `examples/06_qiskit_export_h2.py` insert `src/` on `sys.path`
  when run by path from the checkout.

This is a usability choice for research reproduction; installed
environments still use the normal package metadata under `src/`.

Normal supported runs are expected to be warning-clean. The only
intentional warning containment left in the codebase is a small set of
localized `np.errstate(all="ignore")` guards in verification-scale
dense algebra helpers (`qsvt_poly.py` and `cholesky_channel.py`),
where earlier stacks could emit benign low-level floating-point
warnings during exact dense unitary assembly. Those guards are not used
to hide test failures or paper-facing mismatches; the supported
`pytest` and example workflows now run without warning spam.

* Code anchors: `pyproject.toml`, `composer/__init__.py`,
  `tests/test_foundation.py`, `examples/04_lcu_hamiltonian_h2.py`,
  `examples/05_similarity_sandwich.py`,
  `src/composer/qsp/qsvt_poly.py`,
  `src/composer/block_encoding/cholesky_channel.py`.

## 19. Optional SDK export is an adapter layer, not the semantic core

The paper's source-of-truth semantics in this repo remain the compiled
COMPOSER `Circuit` object model plus the dense verifier. The optional
Qiskit layer is an export adapter for downstream SDK workflows
(transpilation, simulation, resource inspection); it does **not** define
the oracle semantics.

Structural COMPOSER operations are lowered recursively into SDK-side
subcircuits/controlled operations. However, the exporter does not
reinterpret opaque dense leaf primitives as new scalable synthesis
algorithms. If a compiled COMPOSER leaf is still represented as a dense
matrix, the adapter preserves that contract by emitting an exact SDK
unitary instruction for that leaf. After the latest structural cleanup,
the remaining leaves of that form are lower-level primitives such as
full-register fermionic ladder gates, pair-reflection branches, and the
exact full-unitary form of `StatePreparationGate`, not the old one-body
or Cholesky branch wrappers.

One subtle consequence is state preparation: COMPOSER's
`StatePreparationGate` fixes a particular full verification unitary via
Gram-Schmidt completion, not only the prepared `|0...0>` column.
Therefore the optional Qiskit export currently preserves semantics by
emitting that exact unitary, rather than by switching to Qiskit's
native prepared-state-only primitive, whose off-reference action is not
the same object.

The same distinction applies to resource reporting: the compiled
COMPOSER circuit report is the repo-facing source of truth for ancilla
and selector/control overhead, while backend/export depth and two-qubit
counts are explicitly labeled SDK-side views that depend on the chosen
export/transpilation basis.

* Code anchors: `src/composer/circuits/export.py`,
  `src/composer/circuits/backends/qiskit.py`,
  `src/composer/circuits/resources.py`.

## Smaller choices (documented inline in source)

* Sign/phase convention for `sgn(omega_s)` embedding: single Z on the
  selector branch `|s>` before SELECT, verified by diffing the
  top-left block for signed coefficients.
  See `src/composer/block_encoding/lcu.py`.
* Pivot selection for the ladders: `r = argmax |u_p|` (1e), and
  `(r,s) = argmax |u_{pq}|` (2e). Conditioning only; correctness is
  independent. See `src/composer/ladders/one_electron.py`,
  `src/composer/ladders/two_electron.py`.
* Pair-Givens convention `G_{pq,rs}(theta, phi)`: Eq. (32) is
  implemented directly via its JW generator and supports overlapping
  pairs (`{p, q} != {r, s}` but not necessarily four distinct
  orbitals). The direct dense form is used for verification.
  See `src/composer/ladders/phased_pair_givens.py`.
* Antisymmetric-pair SVD unpacking (Eq 19-23): matrix unfolding of
  `t_{ab,ij}` into `(NV 2) x (NO 2)`, followed by a single matrix SVD
  and reshaping of the singular vectors back into antisymmetric pair
  matrices. The repo does not perform any further per-channel
  decomposition beyond that pair-basis factorization.
  See `src/composer/factorization/pair_svd.py`.
