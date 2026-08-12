# Chapter 15 — SLAM as Least Squares: Factor Graphs

> Part V — Mapping and SLAM · Estimated length: 11 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

This is the book's modernization keystone: the chapter where the 2005 baseline's future arrives.
Ch. 14 ended with a filter that could not revise its past; the fix is radical in concept, mundane
in mechanics: *keep the past*. Write the **full** SLAM posterior, take its log, and watch it
shatter into a sum of small quadratic penalties — one per control, one per measurement — a
**factor graph**. MAP inference becomes sparse nonlinear least squares; Gauss-Newton
re-linearizes everything every iteration (the medicine for Ch. 14's frozen Jacobians); and the
sparsity the draft's EIF/SEIF chapters glimpsed finally pays off, because smoothing never
marginalizes and so never densifies. The reader's "aha" is a triple equivalence they can *touch*:
probability (factorization) = optimization (springs relaxing) = linear algebra (a sparse matrix
mirroring the graph). The ancestors are honored: EIF's construct/reduce/solve pipeline *is*
Gauss-Newton avant la lettre; SEIF is the heroic wrong turn whose sparsity instinct was right.

Story line:
1. **Hook** — Ch. 14's inconsistent map on screen. "What if we could go back and re-linearize
   everything?" We can: write down the whole problem.
2. **Play** — w15.1 autoplays: a drifty odometry chain relaxes into the Apartment's true shape.
3. **Formalism** — the posterior factorizes; $-\log$ gives sums of squared Mahalanobis residuals;
   MAP = NLLS; Gauss-Newton and Levenberg-Marquardt derived on the manifold ($\boxplus$, Ch. 3/7).
4. **Structure** — $\Omega$ mirrors the graph; elimination = Schur complement = `EIF_reduce`
   reborn; ordering and fill-in (w15.2).
5. **History** — EIF as one GN iteration; SEIF as forced sparsity; why smoothing won.
6. **Robustness** — one bad loop factor bends the whole map; robust kernels via IRLS.
7. **Implementation & lab** — dense GN/LM → sparse faer → the same graph in `factrs` 0.3; the lab
   heals Ch. 14's run. Bridge: Ch. 16 wraps this optimizer into RustSLAM-2D.

## 2. Prerequisites & Position

- **Builds on:** Ch. 3 (SE(2), $\exp/\log$, $\boxplus/\boxminus$ — retractions are load-bearing);
  Ch. 6 (information form $\xi, \Omega$ — the duality returns at trajectory scale); Ch. 7
  (Jacobian/on-manifold discipline); Ch. 9/10 (models become factors verbatim); Ch. 14 (the two
  fatal flaws this chapter cures; its exported dataset); Appendix B (Schur complement, Gaussian
  marginalization in information form — cited, not re-proved).
- **Feeds into:** Ch. 16 (pose-graph SLAM runs on this optimizer); Ch. 17 (contrast: sampling the
  same full posterior); Ch. 18 (bundle adjustment, preintegration factors, marginalization
  fill-in — SEIF's lesson resurfacing); Ch. 24 (active SLAM plans over this graph); Ch. 26.
- **Baseline sources:** Thrun et al. (1999–2000 draft) Ch. 11 §11.1–11.7 — Tables 11.1
  (`EIF_initialize`), 11.2 (`EIF_construct`), 11.3 (`EIF_reduce`), 11.4 (`EIF_solve`), 11.5
  (`EIF_SLAM_known_correspondence`) — the restructured ancestor; Ch. 12 §12.1–12.7, 12.11 (SEIF),
  condensed to honest history; Ch. 3 §3.4 (information filter). Modernization set: Dellaert &
  Kaess, *Factor Graphs for Robot Perception* (2017) as primary modern source; GTSAM/g2o/Ceres
  context; robust-estimation lineage (Huber; switchable constraints, Sünderhauf & Protzel 2012;
  DCS; adaptive kernels, Chebrolu et al.; graduated non-convexity, Yang et al.; SE-Sync pointer).

## 3. Foundation (F) — Mathematical Core

### 3.1 Notation introduced (chapter-scoped table)

| Symbol | Meaning |
|---|---|
| $y = (x_{0:t}, m)$ | stacked vector of *all* unknowns (Thrun's full-SLAM state) |
| $\phi_k$ | factor $k$: local function of the few variables it touches |
| $r_k(y)$, $\Sigma_k$ | residual of factor $k$ and its noise covariance |
| $\lVert r \rVert_{\Sigma}^2 = r^\top \Sigma^{-1} r$ | squared Mahalanobis norm |
| $J(y) = \tfrac12 \sum_k \lVert r_k(y) \rVert_{\Sigma_k}^2$ | MAP objective ($-\log$ posterior + const) |
| $A$, $\bar r$ | whitened stacked Jacobian / residual at the linearization point |
| $\Omega = A^\top A$, $b = A^\top \bar r$ | normal equations $\Omega\,\Delta = -b$; $\Omega$ **is** Ch. 6's information matrix at trajectory scale |
| $\Delta$, $y \boxplus \Delta$ | tangent-space update and its retraction |
| $\lambda$ | Levenberg-Marquardt damping |
| $\rho(\cdot)$, $w = \rho'(e)/e$ | robust kernel and its IRLS weight |
| $\Omega / \Omega_{mm}$ | Schur complement of the map block (elimination of $m$) |
| $\operatorname{nnz}$, fill-in | sparsity accounting for the Cholesky factor $L$ |

### 3.2 Definitions

- **Factor graph**: bipartite graph of variable nodes (poses, landmarks) and factor nodes (prior
  on $x_0$; one odometry factor per control; one measurement factor per observation).
- **MAP estimate**: $\hat y = \arg\max_y p(y \mid z_{1:t}, u_{1:t}) = \arg\min_y J(y)$.
- **The triple equivalence** (a displayed, color-coded box — the chapter's thesis): *factorized
  posterior* ↔ *sum of quadratic constraints (springs)* ↔ *sparse $\Omega$ mirroring the graph's
  adjacency*. Every later section is one face of this box.
- **Smoothing vs. filtering**: a filter marginalizes the past every step at a frozen
  linearization; a smoother keeps all variables and may re-linearize any of them, any time.

### 3.3 Key derivations

**D1 — The posterior is a graph.**
*Statement:* $p(x_{0:t}, m \mid z_{1:t}, u_{1:t}) \propto p(x_0) \prod_{\tau} p(x_\tau \mid x_{\tau-1}, u_\tau) \prod_{\tau,i} p(z_\tau^i \mid x_\tau, m_{c_\tau^i})$,
and under Gaussian noise its $-\log$ is $J(y)$ with $r_{u_\tau} = x_\tau \boxminus f(x_{\tau-1}, u_\tau)$
and $r_{z_\tau^i} = z_\tau^i - h(x_\tau, m_{c_\tau^i})$.
*Sketch (4 steps):* (1) Bayes-net factorization (Markov + known correspondence for now);
(2) substitute Gaussian noise models; (3) $-\log$: products → sums, densities → half-squared
Mahalanobis residuals; (4) read each summand as a graph edge — the formula *is* the picture.
*Collapsible:* normalizer bookkeeping (why $\eta$ vanishes under $\arg\max$); the $\boxminus$
residual of the SE(2) odometry factor.

**D2 — MAP = NLLS; Gauss-Newton on the manifold.**
*Statement:* linearizing $r_k(y^0 \boxplus \Delta) \approx \bar r_k + J_k \Delta$ (whitened)
makes $J$ quadratic; its minimizer solves $\Omega\,\Delta = -b$; iterate linearize → solve →
retract.
*Sketch (5 steps):* (1) tangent-space expansion; (2) stack and whiten; (3) normal equations;
(4) $y \leftarrow y \boxplus \Delta$; (5) the payoff: *every* Jacobian — including ten-minutes-old
ones — is re-evaluated at the newest estimate; Ch. 14's frozen-linearization sin is structurally
impossible. *Collapsible:* dropped second-order term (GN vs. Newton); at convergence $\Omega$ is
the posterior Laplace approximation's information matrix — Ch. 6's $\Omega$ grown up — with
marginal covariances as selected entries of $\Omega^{-1}$.

**D3 — Levenberg-Marquardt as adaptive trust.**
*Statement:* solve $(\Omega + \lambda \operatorname{diag}\Omega)\,\Delta = -b$; $\lambda \to 0$
is GN, $\lambda \to \infty$ scaled gradient descent; adapt $\lambda$ by gain ratio.
*Sketch (3 steps):* damping as a trust region; the accept/reject schedule; where GN overshoots
(strong nonlinearity, bad init — SLAM with big loops). w15.3 carries the intuition.

**D4 — Sparsity: $\Omega$ mirrors the graph.**
*Statement:* block $\Omega_{[j][k]} \ne 0$ iff variables $j,k$ co-appear in some factor. In SLAM,
poses chain to neighbors and star to observed landmarks; **landmark–landmark blocks are zero**:
$\Omega$ has $O(\text{edges})$ nonzeros, not $O(n^2)$.
*Sketch (3 steps):* (1) $\Omega = \sum_k J_k^\top \Sigma_k^{-1} J_k$ — each factor touches only
its own variables' blocks; (2) enumerate SLAM's factor types → the bordered-block pattern;
(3) contrast Ch. 14: the EKF's dense $\Sigma$ is what marginalizing $x_{0:t-1}$ costs — its
glorious correlations are the smoother's *implicit* by-product, not stored state. *Collapsible:*
proof of the pattern claim; the near-sparsity observation that motivated SEIF.

**D5 — Variable elimination = Schur complement = `EIF_reduce`, reborn.**
*Statement:* eliminating the map block gives the reduced pose system
$(\Omega_{xx} - \Omega_{xm}\Omega_{mm}^{-1}\Omega_{mx})\,\Delta_x = -(b_x - \Omega_{xm}\Omega_{mm}^{-1} b_m)$
— algebraically Thrun's Table 11.3. Each eliminated variable clique-connects its neighbors
(**fill-in**); ordering governs cost (optimal ordering NP-hard; min-degree/COLAMD excellent).
*Sketch (5 steps):* (1) block-partition $\Omega, b$; (2) solve for $\Delta_m$, substitute;
(3) recognize Gaussian marginalization in information form (Appendix B); (4) graph reading: a
landmark seen from poses $i,j,k$ leaves a pose-clique behind — fill-in *is* induced correlation;
(5) sparse Cholesky = repeated single-variable elimination, $\operatorname{nnz}(L)$
ordering-dependent by orders of magnitude. *Collapsible:* full Schur algebra; landmarks-first as
the classical "Schur trick"; `EIF_solve` (T11.4) as exactly the back-substitution pass.

**D6 — Robust kernels via IRLS.**
*Statement:* replacing $\tfrac12 e_k^2$ by $\rho(e_k)$, $e_k = \lVert r_k \rVert_{\Sigma_k}$,
yields at stationarity a *weighted* least squares with $w_k = \rho'(e_k)/e_k$ — Huber caps an
outlier's influence, Cauchy/Geman-McClure suppress it.
*Sketch (4 steps):* (1) quadratic loss gives outliers unbounded leverage; (2) differentiate the
robust objective, factor out $w_k$; (3) IRLS folds into GN as one multiply per factor; (4) the
bill: redescending kernels are non-convex — local minima return; escalation ladder: switchable
constraints/DCS → graduated non-convexity → certifiable SE-Sync. *Collapsible:* Huber weight
derivation; equivalence with covariance inflation.

**D7 — Why filtering lost (the honest history).**
*Statement:* (i) Ch. 14's EKF is this graph with every past pose marginalized immediately at a
once-only linearization — dense $\Sigma$, frozen Jacobians. (ii) The draft-Ch. 11 EIF pipeline,
iterated, *is* Gauss-Newton — construct = linearize+assemble, reduce = Schur elimination, solve =
back-substitution, outer loop = re-linearization; it lacked only the vocabulary and the sparse
solvers. (iii) SEIF (draft Ch. 12) forced the *online* information matrix sparse by deleting
fill-in links, buying constant-time updates at the price of overconfidence — right instinct,
wrong mechanism; smoothing gets exact sparsity free by never marginalizing. (iv) The debt is
live: Ch. 18's sliding-window marginalization re-creates SEIF's dilemma. *Sketch:* the a15.5
timeline; the Rosetta table below. *Collapsible:* SEIF's sparsification rule and where it
violates consistency.

### 3.4 Named algorithms

| Algorithm | Signature | Complexity |
|---|---|---|
| `linearize_graph` | $(\text{graph}, y^0) \to (A, \bar r)$ sparse blocks | $O(K)$ factors, $O(1)$ each |
| `gauss_newton` | $(\text{graph}, y^0; \text{iters}, \text{tol}) \to \hat y$ | per iter: assemble $O(K)$ + sparse Cholesky (fill-in-dependent: near-linear chain-like, $\sim O(n^{1.5})$ planar with good ordering, $O(n^3)$ dense worst case) |
| `levenberg_marquardt` | $(\text{graph}, y^0; \lambda_0) \to \hat y$ | GN + $O(n)$ damping per trial step |
| `schur_marginalize` | $(\Omega, b, \text{blocks } M) \to (\tilde\Omega, \tilde b)$ | $O(\sum_{v \in M} \deg(v)^2)$ — **= `EIF_reduce` (T11.3)** |
| `irls_weight` | $(\rho, e_k) \to w_k$ | $O(1)$ per factor per iteration |

**Rosetta table (printed in the chapter):** `EIF_initialize` (T11.1) = odometry initial guess ·
`EIF_construct` (T11.2) = linearize + assemble at fixed $\mu$ · `EIF_reduce` (T11.3) = Schur-
eliminate landmarks · `EIF_solve` (T11.4) = solve + back-substitute · the outer loop of
`EIF_SLAM_known_correspondence` (T11.5) = Gauss-Newton. One table, 25 years of history.

Numeric micro-example (hand-checkable, unit-tested): 1D chain $(x_0, x_1, x_2)$, unit-information
factors: prior $x_0 = 0$; odometry $x_1 - x_0 = 1$, $x_2 - x_1 = 1$; absolute fix $x_2 = 1.5$.
From init $(0,0,0)$: $\Omega = \begin{pmatrix} 2 & -1 & 0 \\ -1 & 2 & -1 \\ 0 & -1 & 2 \end{pmatrix}$,
$b = (1, 0, -2.5)^\top$; solving $\Omega\Delta = -b$ gives $\hat y = (-0.125, 0.75, 1.625)$ in one
step (linear problem). The prior and the fix disagree; least squares shrinks *both* odometry
intervals to $0.875$ — tension shared, exactly like springs.

## 4. Conceptual (C) — Intuition & Visual Design

One metaphor end-to-end: **springs and their ledger** — factors are springs (stiffness =
information), optimization is relaxation to minimum energy, and $\Omega$ is the ledger of who is
connected to whom. The Sparsity Scope is the ledger made visible.

- **Widget w15.1: Spring-Graph Optimizer** *(flagship, TOC name)* — type: wasm-sim.
  Apartment loop: pose chain + landmark stars; springs colored by residual energy; energy meter
  with per-iteration ticks. Autoplay: LM relaxation from the drifty odometric init (orange) to
  the converged map (purple) against gray-dashed truth — the snap-to-shape moment. Manipulates:
  **drag any node** (graph re-relaxes live — direct manipulation of MAP); **"add bad loop
  factor"** injects an outlier closure and the whole map bends (L2's unbounded leverage); **one
  headline parameter:** kernel selector + scale $k$ — flip to Huber and the bad spring visibly
  slackens (drawn thin, IRLS weight printed), the map recovers; step mode exposes single GN
  iterations. Misconceptions killed: "optimization is a black box" (it's springs) and "least
  squares averages outliers away" (it amplifies them; robustness is a *modeling* decision).
  Static fallback: before / relaxed / outlier / rescued four-panel strip.
- **Widget w15.2: Sparsity Scope** *(flagship, TOC name)* — type: wasm-sim, linked views.
  Left: the factor graph; right: live spy plot of $\Omega$; bottom: spy plot of $L$ with fill-in
  cells flashed in orange. Reader picks the elimination ordering — {chronological,
  landmarks-first (Schur trick), poses-first, min-degree} — then **scrubs elimination
  variable-by-variable**: the eliminated node's neighbors clique-connect simultaneously in graph
  and matrix (one event, two faces). $\operatorname{nnz}(L)$ counter, FLOPs estimate, ordering
  leaderboard; preset **"reduce like it's 1999"** runs landmarks-first and labels the reduced
  system "`EIF_reduce` (T11.3) → the pose graph of Ch. 16." Misconceptions killed: "elimination
  order is an implementation detail" and "marginalizing variables simplifies the problem" (it
  relocates complexity as fill-in). Static fallback: spy-plot triptych for two orderings.
- **Widget w15.3: Descent Duel** — type: wasm-sim (supporting). Contours of the true nonlinear
  cost around one pose; overlaid step paths for GN, gradient descent, and LM with a $\lambda$
  slider (autoplay sweeps it once). Misconception killed: "GN always converges" / "LM is just GN
  with insurance" — watch GN overshoot a curved valley while LM interpolates down.
- **Widget w15.4: Kernel Gallery** — type: wasm-sim (supporting, small). $\rho(e)$ and $w(e)$
  curves for L2/Huber/Cauchy/Geman-McClure; below, a 10-inlier 1D fit where the reader drags one
  outlier's magnitude: the estimate follows under L2, resists under Huber, ignores under GM — a
  bad-init preset shows GM trapped in a local minimum. Misconception killed: "robust kernels are
  a free lunch."
- **Animation a15.5: From Filter to Graph** — type: animation (scrub-only; storyboard fallback).
  Three synchronized timelines on the same data: EKF (past poses collapse, $\Sigma$ densifies),
  smoother (chain stays sparse, re-linearization ripples backward), SEIF (links snipped by force,
  an "approximation debt" meter rising). D7 as thirty seconds of pictures.

Dashboard: the chapter **integration lab** loads Ch. 14's exported dataset into w15.1 and
batch-optimizes it — the smoother heals the EKF's inconsistent map, final RMSE printed next to
the EKF's; Part V's arc closes on that number. All widgets autoplay, seeded, one headline
parameter, static fallbacks rendered at build time from the same Rust code.

## 5. Practical (P) — Rust Implementation

Crates:
- `nalgebra` 0.35 — dense block algebra; `DMatrix` reference solver (oracle to ~300 variables).
- `nalgebra-sparse` 0.12 — COO triplet assembly of $\Omega$: sparsity *taught* through the API
  before performance enters.
- `faer` 0.24 (+ `faer-ext`) — sparse Cholesky for the real solve; the same engine `factrs` uses,
  so graduating changes the API, not the math. WASM: scalar fallback with `rayon` off; dense path
  is the documented widget-scale fallback.
- `factrs` 0.3 — production path: GTSAM-style typed variables/factors, GN/LM, Huber, serde;
  native-side (WASM unverified per the stack research — widgets run our hand-rolled solver).
- `levenberg-marquardt` 0.15 — dense LM cross-check in tests; `tiny-solver` 0.18 — Ceres-flavored
  alternate (exercise 6).
- `rand` 0.9 / `rand_distr` 0.6 (seeded `SmallRng`); `egui`/`eframe` 0.35 + `egui_plot` 0.34;
  `plotters`; `pr-core`'s `geom` module (Ch. 3) for SE(2) $\boxplus$ and Jacobians.

Module plan:

```text
crates/ch15_graph/
  src/key.rs        — VarKey { Pose(usize) | Landmark(usize) }, BlockIndex
  src/values.rs     — Values container + retract (⊞ per block type)
  src/factor.rs     — Factor trait; PriorFactor, OdomFactor, RangeBearingFactor
  src/kernel.rs     — Kernel enum + IRLS weight (D6)
  src/dense.rs      — DenseCholesky (nalgebra) — the oracle
  src/sparse.rs     — COO assembly → faer CSC → sparse LLT (D4/D5)
  src/schur.rs      — schur_marginalize = EIF_reduce reborn
  src/order.rs      — orderings: chronological, landmarks_first, min_degree
  src/optimize.rs   — gauss_newton, levenberg_marquardt (solver-generic)
  examples/spring_relax.rs   examples/rescue_the_loop.rs
  examples/heal_ch14.rs      examples/factrs_same_graph.rs
  tests/micro_1d.rs          tests/dense_vs_sparse.rs   tests/lm_crosscheck.rs
demos/ch15-spring-graph/   — w15.1, w15.3, w15.4
demos/ch15-sparsity/       — w15.2, a15.5
```

Key types & signatures (compiles-in-spirit):

```rust
use nalgebra::{DMatrix, DVector, Matrix2, Matrix3, Vector2};
use pr_core::geom::SE2;   // Ch. 3: exp/log, ⊞ (compose ∘ exp), ⊟, right Jacobians

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum VarKey { Pose(usize), Landmark(usize) }

pub struct Values { poses: Vec<SE2>, landmarks: Vec<Vector2<f64>> }
impl Values {
    /// y ← y ⊞ Δ: SE2 retraction for poses, vector add for landmarks
    pub fn retract(&mut self, delta: &DVector<f64>, index: &BlockIndex);
}

pub struct Linearization { pub residual: DVector<f64>,        // whitened, √w folded in
                           pub jacobians: Vec<DMatrix<f64>> } // one block per key
pub trait Factor {
    fn keys(&self) -> &[VarKey];
    fn dim(&self) -> usize;
    fn linearize(&self, v: &Values) -> Linearization;   // text cross-links: // eq. (15.9)
    fn kernel(&self) -> Kernel { Kernel::L2 }
}
pub struct PriorFactor        { pub key: VarKey, pub prior: SE2, pub sqrt_info: Matrix3<f64> }
pub struct OdomFactor         { pub i: usize, pub j: usize, pub delta: SE2,
                                pub sqrt_info: Matrix3<f64> }   // r = (x_i⁻¹ x_j) ⊟ δ
pub struct RangeBearingFactor { pub pose: usize, pub lm: usize, pub z: Vector2<f64>,
                                pub sqrt_info: Matrix2<f64>, pub kernel: Kernel }

pub enum Kernel { L2, Huber { k: f64 }, Cauchy { c: f64 }, GemanMcClure { c: f64 } }
impl Kernel { pub fn weight(&self, e_sq: f64) -> f64 }          // w = ρ′(e)/e

pub struct FactorGraph { pub factors: Vec<Box<dyn Factor>> }

pub trait LinearSolver { fn solve(&self, omega: &SparseSym, b: &DVector<f64>) -> DVector<f64>; }
pub struct DenseCholesky;                              // nalgebra, O(n³) oracle
pub struct FaerLlt { pub ordering: Ordering }          // faer sparse LLT
pub enum Ordering { Chronological, LandmarksFirst, MinDegree }

pub fn gauss_newton<S: LinearSolver>(g: &FactorGraph, init: Values, cfg: &GnConfig, s: &S)
    -> (Values, Report);            // Report: iterations, cost trace, nnz(L)
pub fn levenberg_marquardt<S: LinearSolver>(g: &FactorGraph, init: Values, cfg: &LmConfig, s: &S)
    -> (Values, Report);

/// EIF_reduce (Thrun T11.3), reborn: eliminate the given blocks via Schur complement.
pub fn schur_marginalize(omega: &SparseSym, b: &DVector<f64>, victims: &[VarKey])
    -> (SparseSym, DVector<f64>);
```

The `factrs` 0.3 section rebuilds the *same* Apartment graph with factrs's typed `SE2` variables,
prior/between/range-bearing factors, Huber wrapper, and its GN/LM optimizers — ~40 lines —
comparing: identical solution to $10^{-8}$, comparable sparse-solve time (same faer underneath),
far less code; plus what factrs adds (typed keys, serde graph I/O, rerun integration) and what no
Rust crate has yet (Bayes tree / iSAM2 incremental smoothing — honest ecosystem note, → GTSAM).

Worked end-to-end example — `cargo run --example spring_relax`: Apartment loop, 60 poses, 14
landmarks (208 tangent dims), odometry drift seeded with `42`. Expected output: `J: 8.4e3 → 61.2
in 6 GN iterations`; dense-vs-sparse timing table (~ms vs. ~100 µs here; gap widening to ~10²× on
the 500-pose synthetic); before/after/truth plot in book colors. `rescue_the_loop --outlier`
shows L2 warping the map (RMSE ×8); adding `--huber` recovers it (within 5 % of clean).
`heal_ch14` loads Ch. 14's dataset, optimizes, prints EKF vs. smoother RMSE side by side. Tests:
`micro_1d` asserts $(-0.125, 0.75, 1.625)$ exactly; `dense_vs_sparse` asserts agreement to
$10^{-9}$; `lm_crosscheck` matches the `levenberg-marquardt` crate on the dense toy. Runnable
artifact: the WASM demo is the Spring-Graph Optimizer running this chapter's hand-rolled sparse
GN/LM live in the browser; the native examples produce the printed figures and the factrs table.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w15.1 | Spring-Graph Optimizer | wasm-sim | ch15_graph + sim + eframe 0.35 | drag nodes, step GN, add bad loop, kernel selector+scale, energy meter | MAP = energy minimization; re-linearization; outlier leverage & robust rescue |
| w15.2 | Sparsity Scope | wasm-sim | ch15_graph (order, schur) + eframe | ordering dropdown, elimination scrubber, nnz/FLOPs leaderboard, 1999 preset | Ω mirrors the graph; elimination = Schur; fill-in & ordering cost |
| w15.3 | Descent Duel | wasm-sim | ch15_graph + egui_plot | λ slider, autoplay sweep | GN overshoot vs. LM interpolation |
| w15.4 | Kernel Gallery | wasm-sim | ch15_graph (kernel) + egui_plot | drag outlier, kernel tabs, bad-init preset | influence functions; the non-convexity bill |
| a15.5 | From Filter to Graph | animation | scripted + storyboard fallback | scrub only | EKF/EIF/SEIF/smoothing history in one timeline |

## 7. Exercises & Extensions

1. **(F)** Derive the normal equations from the linearized residuals (D2), then verify the §3.4
   1D micro-example by hand: assemble $\Omega$ and $b$, solve, and explain in one sentence why
   both odometry intervals shrank equally.
2. **(F)** Prove D4's sparsity-pattern claim; draw $\Omega$'s block pattern for a 5-pose/
   2-landmark graph; then show Schur-eliminating a landmark seen from poses $i, j, k$ creates
   the pose-clique fill-in, citing the Appendix B identity used.
3. **(F)** Derive the Huber IRLS weight $w = \rho'(e)/e$ from stationarity, and show L2's
   influence is unbounded while Huber's saturates at $k$.
4. **(C)** *Predict-then-verify with w15.2:* for the M-shaped preset, predict which ordering
   (chronological vs. landmarks-first) yields smaller $\operatorname{nnz}(L)$ and why; verify;
   find a graph where your rule of thumb *fails* and explain via induced cliques.
5. **(C)** *Predict-then-verify with w15.1:* find an outlier magnitude where Huber no longer
   rescues the map but Geman-McClure (good init) does; break GM with the bad-init preset; write
   the one-paragraph pitch for graduated non-convexity.
6. **(P)** Implement a `BearingOnlyFactor` and the Cauchy kernel; implement min-degree ordering
   in `order.rs` and compare $\operatorname{nnz}(L)$ on the 500-pose synthetic; port the same
   graph to `tiny-solver` 0.18 and report agreement and timing vs. our solver and `factrs`.

## 8. Modernization Notes

- **This chapter is the book's main restructure.** The 1999–2000 draft has no factor-graph
  chapter; its Ch. 11 (EIF, ancestor of published GraphSLAM) is presented as exactly that — the
  Rosetta table lets a reader of the old book watch their chapter become this one. Primary modern
  source: Dellaert & Kaess (2017); GTSAM/g2o/Ceres named as the lineage.
- **What the baseline lacked, now added:** GN/LM as first-class derived algorithms (the EIF
  iterated but never named its loop); sparsity/fill-in/ordering theory (the elimination game,
  min-degree/COLAMD) that makes smoothing *fast*, not merely correct; robust estimation entirely
  — kernels, IRLS, switchable-constraints/DCS/GNC pointers, SE-Sync mention — absent in 2005;
  the manifold-correct $\boxplus$ formulation; marginal covariances via $\Omega^{-1}$; a working
  Rust path (hand-rolled → faer → factrs 0.3, tiny-solver alternate).
- **Condensed to honest history:** draft Ch. 12 (SEIF) — now essentially historical per the
  modernization findings — survives as D7's ~2-page cautionary tale: right insight (information
  sparsity), wrong mechanism (forced link deletion → inconsistency), one living legacy (Ch. 18's
  marginalization/fill-in dilemma). Dropped outright from Ch. 11–12: EIF/SEIF correspondence
  tests (T11.8/T11.9), tree-based branch-and-bound association, equivalency constraints,
  multi-vehicle SLAM, amortized map recovery — superseded by front-end verification (Ch. 16)
  plus robust back-ends (this chapter), which is how 2026 systems handle bad associations.
- **Deliberately deferred, with signposts:** Bayes tree / iSAM2 incremental smoothing (concept
  box only; factrs is batch-only and the text says so); fixed-lag smoothing as the
  filter–smoother bridge (one paragraph, → Ch. 18); pose-graph SLAM and loop-closure front-ends
  (→ Ch. 16); continuous-time trajectories (pointer to Barfoot).
