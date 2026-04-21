# Paper to code mapping

Peng, Liu, Kowalski (2026). *COMPOSER: Compile-Once Modular Parametric
Oracle for Similarity-Encoded Effective Reduction*.

| Paper section / equation / result | Code |
|-----------------------------------|------|
| Def 1, Eq (1)-(3) bilinear rank-one operator | [`src/composer/operators/rank_one.py`](src/composer/operators/rank_one.py) `BilinearRankOne` |
| Def 2, Eq (4)-(5) pair rank-one operator | `rank_one.py :: PairRankOne` |
| Def 3, Eq (8)-(9) projected quadratic rank-one (`O = sum_r C_r n[u^(r)]`, `L = O O^dag`) | `rank_one.py :: ProjectedQuadraticRankOne`, `fermion.py :: jw_mode_number` |
| Eq (10) two-body Hamiltonian, physicist order | [`src/composer/utils/fermion.py`](src/composer/utils/fermion.py) `two_body_matrix` |
| Eq (11) pivoted Cholesky of (pq|rs) | [`src/composer/factorization/cholesky.py`](src/composer/factorization/cholesky.py) (generic Hermitian-PSD primitive; the Hamiltonian pool currently uses the real-valued subset only) |
| Eq (12) mean-field shift of h_pq | [`src/composer/factorization/mean_field_shift.py`](src/composer/factorization/mean_field_shift.py) |
| Eq (13)-(17) Hamiltonian rank-one pool | [`src/composer/operators/hamiltonian.py`](src/composer/operators/hamiltonian.py) (implemented for real physicist-order electronic integrals with real-symmetric Cholesky factors; see ASSUMPTION #16) |
| Eq (18)-(27) sigma-hat generator representation: explicit singles channels `t_ai a_a^dag a_i`, explicit embedded doubles channel pool `L_s = omega_s B^dag[U^(s)] B[V^(s)]`, and dense generator reference reconstruction | [`src/composer/operators/generator.py`](src/composer/operators/generator.py), [`src/composer/factorization/pair_svd.py`](src/composer/factorization/pair_svd.py), `rank_one.py :: PairRankOne` |
| Sec III.A + App A.1 one-electron ladder, with separate preparation and number-conserving forms, plus dense `mode_rotation_unitary(u)` extracted from that ladder for rotated occupation gadgets | [`src/composer/ladders/one_electron.py`](src/composer/ladders/one_electron.py) |
| Sec III.B + App A.2 direct two-electron pair ladder over non-pivot unordered pairs | [`src/composer/ladders/two_electron.py`](src/composer/ladders/two_electron.py) |
| Eq (29) SO(2) Givens G_{pr}(theta) | [`src/composer/ladders/givens.py`](src/composer/ladders/givens.py) |
| Eq (32) phased pair-Givens, including overlapping-pair rotations | [`src/composer/ladders/phased_pair_givens.py`](src/composer/ladders/phased_pair_givens.py) |
| Lemma 1, Eq (33) bilinear block encoding | [`src/composer/block_encoding/bilinear.py`](src/composer/block_encoding/bilinear.py) |
| Lemma 2, Eq (34) projected quadratic block encoding | [`src/composer/block_encoding/cholesky_channel.py`](src/composer/block_encoding/cholesky_channel.py) (explicit dense realization of the rotated-mode PREP/SELECT/occupation-flag construction, followed by a constant-overhead exact degree-2 transform) |
| Theorem 1, Eq (35) binary-multiplexed LCU | [`src/composer/block_encoding/lcu.py`](src/composer/block_encoding/lcu.py) (exact top-left-block identity on the supported real-integral scope) |
| Eq (36)-(43) masked sigma-pool oracle `U_sigma = PREP_sigma^dag SELECT_sigma PREP_sigma` with fixed compiled selector width, explicit null branch, and ancilla-zero block reporting for `-i sigma_pool / alpha_bar` over the compiled singles+doubles generator pool | [`src/composer/block_encoding/generator_exp.py`](src/composer/block_encoding/generator_exp.py), [`src/composer/operators/mask.py`](src/composer/operators/mask.py) |
| Eq (44)-(46) QSP exponential target, implemented as a paper-facing oracle flow over `U_sigma`: parity-split scalar phase synthesis for `cos` / `sin`, repeated oracle-call QSP sequences over a fixed compiled Wx-form sigma oracle, Hermitian-part extraction via explicit branch-select composition, a final LCU for `e^{sigma} / 2`, and one round of oblivious amplitude amplification to obtain the similarity-side unitary whose ancilla-zero block approximates `e^{sigma}`. The repo still does not synthesize one direct complex single-ladder phase list, but the returned `unitary_circuit` closes the paper-facing `U_sigma` identity on the supported small-system scope | [`src/composer/block_encoding/generator_exp.py`](src/composer/block_encoding/generator_exp.py), [`src/composer/qsp/chebyshev.py`](src/composer/qsp/chebyshev.py), [`src/composer/qsp/phases.py`](src/composer/qsp/phases.py) |
| Eq (47)-(53) similarity sandwich | [`src/composer/block_encoding/similarity_sandwich.py`](src/composer/block_encoding/similarity_sandwich.py) (the returned outer circuit is the literal nested compiled-object composition `U_sigma^dag W_H U_sigma`, where `U_sigma` is the amplified similarity-side unitary derived from the real singles+doubles generator oracle/QSP construction and `W_H` is the real Hamiltonian oracle. The reported `encoded_system_block_dense` is the exact ancilla-zero block of that nested circuit, and on the supported verification-scale systems the repo now claims `alpha_H * encoded_system_block_dense ~= e^{-sigma} H e^{sigma}` before the external model-space projector is applied. `H_eff_dense` remains the projected dense paper target `P e^{-sigma} H e^{sigma} P`, with `P` kept external exactly as in Sec. IV.C.) |
| App B.1 Lemma 1 proof | inline docstring in `bilinear.py` |
| App B.2 Lemma 2 proof + x -> x^2 polynomial | `cholesky_channel.py`, [`src/composer/qsp/qsvt_poly.py`](src/composer/qsp/qsvt_poly.py) (scalar Wx reference phases plus the exact ancilla-projector degree-2 transform used by the returned block encoding) |
| App B.3 Theorem 1 proof | inline docstring in `lcu.py` |
| App C.1-C.3 scalar QSP utilities and Chebyshev truncation/error bounds used by the generator-exp oracle construction's scalar phase targets, plus the retained dense reference branch | [`src/composer/qsp/chebyshev.py`](src/composer/qsp/chebyshev.py), [`src/composer/qsp/phases.py`](src/composer/qsp/phases.py) |
| App D.1 selector-controlled overhead and logical compiled-object accounting | [`src/composer/block_encoding/lcu.py`](src/composer/block_encoding/lcu.py), [`src/composer/block_encoding/generator_exp.py`](src/composer/block_encoding/generator_exp.py), [`src/composer/block_encoding/similarity_sandwich.py`](src/composer/block_encoding/similarity_sandwich.py), [`src/composer/circuits/circuit.py`](src/composer/circuits/circuit.py) (current repo scope: returned objects now carry logical resource summaries derived from the actual compiled circuits/oracles: selector widths, active/compiled branch counts, gate inventories, and QSP query counts. These summaries do **not** claim fault-tolerant synthesis costs or selector-controlled two-qubit depth constants beyond the paper's asymptotic discussion.) |
| App E.2 channel-overlap / rank-cumulative wAUC subspace diagnostic, with shorter-list generalization documented in ASSUMPTION #15 | [`src/composer/diagnostics/subspace.py`](src/composer/diagnostics/subspace.py) |
| App E.3 MP2 amplitudes + MP2-weighted one-shot cumulative-coverage selector mask | [`src/composer/diagnostics/mp2.py`](src/composer/diagnostics/mp2.py), [`src/composer/diagnostics/mask_selection.py`](src/composer/diagnostics/mask_selection.py) |

This table maps paper anchors to the current modules/functions. It is
intentionally module-level rather than line-level because line numbers
are not a stable public interface.

Repo-facing infrastructure that is not itself a paper claim lives
outside the table: `pyproject.toml`, `composer/__init__.py`,
`tests/test_foundation.py`, and the two `examples/` entry points define
the supported Python floor (`>=3.10`), the editable-install dev
workflow, checkout-local import behavior for the `src` layout, and the
packaging/foundation checks used to keep those claims honest.
