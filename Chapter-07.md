# Chapter 7 — Beyond Linearity: EKF, UKF, and Estimation on Manifolds

> Part II — The Bayes Filter Family · Estimated length: 13 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Real robots rotate, and rotation is nonlinear twice over: the models are nonlinear functions, and the
state itself doesn't live in a vector space. This chapter is where the book's 2005 baseline ends and
its 2026 payload begins. First the classical move: linearize with Taylor, get the EKF, and be honest
about the lie (a tangent line is only as good as the curvature × spread it ignores). Then the better
approximation: the unscented transform — a few well-chosen scouts instead of one tangent line. Then
the deeper fix, which no amount of better linearization provides: put the *state* on the manifold
where it lives. The "aha" chain: EKF error is a property you can *see and measure* (w7.1); the
heading wrap-around bug is not a bug but a type error — averaging angles as reals is meaningless
(w7.2); and $\boxplus/\boxminus$ let every Kalman equation survive intact with $+/-$ replaced,
covariance living in the tangent space. Invariant-EKF intuition closes the arc: choose the error
definition well and the filter becomes consistent almost by construction.

Story line:
1. **Problem** — Rusty turns. The Ch. 6 `Kf` on $(x, y, \theta)$ with a compass measurement near
   $\pm\pi$ produces a posterior heading pointing *backwards* (autoplay hook: the 179°/−179° mean
   catastrophe).
2. **Play** — w7.1 Linearization Lens autoplaying a Gaussian pushed through a curve: gray truth
   banana vs. orange EKF ellipse vs. purple UKF ellipse.
3. **Intuition** — tangent-line lies; sigma-point scouts; "the error is small even when the state
   is wild" (error-state idea, w7.3).
4. **Formalism** — EKF via Taylor (Thrun §3.3, full derivation); UT/UKF; $\boxplus/\boxminus$
   retraction axioms; error-state EKF; on-manifold EKF/UKF on $SE(2)$; invariant-EKF intuition.
5. **Implementation** — `Manifold` trait; generic `Ekf`/`Ukf` over it; `VectorSpace` impl recovers
   Ch. 6 exactly; `SE2` impl fixes the hook; a deliberate compile error as pedagogy.
6. **Experiment** — figure-eight lab: vector EKF vs. error-state SE(2) EKF vs. UKF-M on identical
   logs; failure gallery (operating-point sensitivity, overconfident divergence, wrap-around).

## 2. Prerequisites & Position

- **Builds on:** Ch. 3 (SO(2)/SE(2), exp/log, $\boxplus/\boxminus$ preview, banana-distribution
  teaser), Ch. 5 (`BayesFilter` trait), Ch. 6 (every KF equation, Joseph form, NEES/NIS), Ch. 2
  (Gaussian transformations).
- **Feeds into:** Ch. 9 (on-manifold motion noise), Ch. 11 (EKF localization runs on this chapter's
  error-state EKF), Ch. 14 (EKF SLAM + its inconsistency autopsy, which invariant filtering
  reframes), Ch. 15 (iterated linearization → Gauss-Newton), Ch. 16/18 (iterated ESKF in
  FAST-LIO-style odometry; MSCKF), Ch. 25 (differentiable filters).
- **Baseline sources:** Thrun et al. (draft) Ch. 3 §3.3 (Taylor linearization §3.3.1, EKF algorithm
  Table 3.3, derivation §3.3.3, practical considerations §3.3.4). **The UKF is absent from the
  draft PDF** (it entered the published 2005 edition): sourced here from the modernization set —
  Julier & Uhlmann's UT as presented in Barfoot 2nd ed. Ch. 4 (sigmapoint methods). All manifold
  material is post-2005: Solà et al., "A micro Lie theory" (arXiv:1812.01537); Solà's ESKF
  quaternion-kinematics notes; Barfoot 2nd ed. (matrix Lie groups); Brossard et al. UKF-M;
  Barrau & Bonnabel (IEKF) for the invariant intuition.

## 3. Foundation (F) — Mathematical Core

**Notation introduced:** nonlinear models $g(u_t, x_{t-1})$, $h(x_t)$; Jacobians $G_t, H_t$;
sigma points $\mathcal{X}^{[i]}$, UT weights $w_m^{[i]}, w_c^{[i]}$, parameters
$(\alpha_{\mathrm{UT}}, \beta, \kappa)$, $\lambda = \alpha_{\mathrm{UT}}^2(n+\kappa) - n$
(**call-out:** $\alpha_{\mathrm{UT}}$ is subscripted book-wide to avoid colliding with the motion
noise parameters $\alpha_1..\alpha_6$ of Ch. 9); manifold $\mathcal{M}$, tangent dimension $d$,
retractions $\boxplus: \mathcal{M} \times \mathbb{R}^d \to \mathcal{M}$,
$\boxminus: \mathcal{M} \times \mathcal{M} \to \mathbb{R}^d$; error state
$\varepsilon_t = x_t \boxminus \mu_t$; $T \in SE(2)$, $\exp/\log$ per Ch. 3.

**Definitions:**
- *Nonlinear Gaussian system*: $x_t = g(u_t, x_{t-1}) + \varepsilon_t$, $z_t = h(x_t) + \delta_t$,
  noises as in Ch. 6. Closure is lost: the true $bel(x_t)$ is non-Gaussian from step one; every
  filter in this chapter chooses *which* Gaussian to pretend with.
- *On-manifold Gaussian*: pair $(\mu \in \mathcal{M},\ \Sigma \in \mathbb{R}^{d\times d})$
  representing the distribution of $\mu \boxplus \varepsilon$, $\varepsilon \sim \mathcal{N}(0,\Sigma)$
  — mean on the manifold, covariance in its tangent space. (Concentrated-Gaussian caveat stated.)
- *Retraction axioms*: $x \boxplus 0 = x$; $x \boxplus (y \boxminus x) = y$;
  $(x \boxplus \delta) \boxminus x = \delta$ locally; smoothness. For $SE(2)$:
  $x \boxplus \delta = x \cdot \exp(\delta)$, $y \boxminus x = \log(x^{-1} y)$ (right/local
  convention fixed book-wide; left convention noted with a table in Appendix C).

**Derivations:**

1. **EKF via first-order Taylor** (Thrun §3.3.1/3.3.3). *Statement:* replace $g, h$ by their
   tangent maps at the current mean — $g(u_t, x_{t-1}) \approx g(u_t, \mu_{t-1}) +
   G_t (x_{t-1} - \mu_{t-1})$, $h(x_t) \approx h(\bar\mu_t) + H_t (x_t - \bar\mu_t)$ — then run
   Ch. 6 verbatim: $\bar\mu_t = g(u_t, \mu_{t-1})$, $\bar\Sigma_t = G_t\Sigma_{t-1}G_t^\top + R_t$,
   $K_t = \bar\Sigma_t H_t^\top S_t^{-1}$, etc. *Sketch (4 steps):* Taylor-expand inside the Ch. 6
   convolution/product derivations; observe every step goes through with $A_t \to G_t$,
   $C_t \to H_t$; note the means propagate through the *true* $g, h$ (only covariances use the
   linearization). *Collapsible:* the full derivation mirroring Ch. 6's completing-the-square, plus
   Thrun §3.3.4's practical considerations recast as measurable claims the widgets demonstrate.
2. **The size of the lie.** *Statement:* the neglected term is second-order; the induced mean error
   scales like $\tfrac{1}{2}\sum_i e_i\, \mathrm{tr}(\nabla^2 g_i \, \Sigma)$ — curvature × spread.
   *Sketch (3 steps):* second-order Taylor with Gaussian input; take expectations (odd terms
   vanish); read the bias term. *Collapsible:* full second-order analysis. This equation is w7.1's
   live error readout.
3. **Unscented transform** (modernization payload; not in the draft). *Statement:* the $2n{+}1$
   points $\mathcal{X}^{[0]} = \mu$, $\mathcal{X}^{[\pm i]} = \mu \pm \big(\sqrt{(n+\lambda)\Sigma}\big)_i$
   with weights $w_m^{[0]} = \lambda/(n{+}\lambda)$, $w_c^{[0]} = w_m^{[0]} + (1 - \alpha_{\mathrm{UT}}^2
   + \beta)$, $w^{[i]} = 1/(2(n{+}\lambda))$, propagated through $g$ and re-collected, match the true
   mean/covariance to 3rd order for Gaussian inputs (vs. EKF's 1st). *Sketch (5 steps):* demand a
   point set matching $\mu, \Sigma$ exactly; exploit symmetry to kill odd moments; propagate;
   recombine; state the accuracy order (proof of the moment-matching claims in the collapsible;
   matrix square root via Cholesky, which the Rust shows). No Jacobians anywhere — $g$ is a black
   box.
4. **UKF.** *Statement:* UT for predict (augment with $R_t$ or additive-noise shortcut), UT for
   correct with cross-covariance $\bar\Sigma^{xz}_t$ and gain $K_t = \bar\Sigma^{xz}_t S_t^{-1}$.
   *Sketch:* transcribe the Bayes-filter steps with UT as the Gaussian pushforward engine.
   *Collapsible:* augmented-state variant; parameter-choice guidance
   ($\alpha_{\mathrm{UT}} \in [10^{-3}, 1]$, $\beta{=}2$, $\kappa = 3{-}n$ heuristics) and when UKF
   is *not* better (near-linear regimes: same answer, 2n+1× cost).
5. **Error-state formulation.** *Statement:* maintain nominal $\mu_t \in \mathcal{M}$ plus a
   zero-mean error $\varepsilon_t \in \mathbb{R}^d$ with covariance $\Sigma_t$; filter the *error*
   (nearly linear, always small), inject $\mu \leftarrow \mu \boxplus \hat\varepsilon$, reset
   $\hat\varepsilon \to 0$. *Sketch (4 steps):* define $\varepsilon = x \boxminus \mu$; derive its
   dynamics to first order; note curvature of $g$ over the *error's* small support is negligible even
   when over the state's it isn't; give the inject/reset cycle (with the reset Jacobian relegated to
   the collapsible). Source: Solà ESKF notes.
6. **EKF/UKF on the manifold.** *Statement:* replace every vector $+/-$ in the EKF with
   $\boxplus/\boxminus$: predict $\bar\mu_t = g(u_t, \mu_{t-1})$,
   $\bar\Sigma_t = G_t \Sigma_{t-1} G_t^\top + R_t$ with $G_t$ the Jacobian *in tangent
   coordinates* ($G_t = \partial\, (g(u, \mu \boxplus \varepsilon) \boxminus g(u,\mu)) /
   \partial \varepsilon |_0$); correct $\delta = K_t (z_t - h(\bar\mu_t))$,
   $\mu_t = \bar\mu_t \boxplus \delta$. UKF-M: sigma points $\mu \boxplus (\pm\sqrt{(n{+}\lambda)\Sigma})_i$,
   recombination via $\boxminus$-mean iteration. *Sketch:* show the vector-space case reduces to
   Ch. 6 exactly ($\boxplus = +$); work the $SE(2)$ Jacobian for the unicycle model concretely.
   *Collapsible:* the iterative mean-on-manifold computation; left-vs-right Jacobian bookkeeping
   (pointer to Appendix C).
7. **Invariant-EKF intuition** (no full derivation; qualitative + one worked fact). *Statement:*
   defining error by group operation, $\eta_t = \mu_t^{-1} x_t$ (right-invariant analog stated),
   makes error dynamics for group-affine models *independent of the state estimate* — Jacobians
   constant, no linearization-point drift, consistency guarantees follow. *Sketch (3 steps):* show
   for $SE(2)$ odometry that $\eta$ evolves autonomously; contrast with the standard EKF whose
   $G_t$ depends on $\mu_{t-1}$; state (cite, don't prove) the Barrau–Bonnabel convergence result.
   Payoff deferred: Ch. 14's EKF-SLAM inconsistency autopsy re-reads this section.

**Named algorithms (signatures + complexity):**
- `Extended_Kalman_filter(µ_{t-1}, Σ_{t-1}, u_t, z_t) → (µ_t, Σ_t)` — Thrun Table 3.3;
  $O(d^3)$ per step, $d$ = tangent/state dim.
- `Unscented_transform(µ, Σ, f) → (µ', Σ')` — $2d{+}1$ evaluations of $f$ + one Cholesky; $O(d^3)$.
- `Unscented_Kalman_filter(µ_{t-1}, Σ_{t-1}, u_t, z_t) → (µ_t, Σ_t)` — modernization-set algorithm
  (absent from the draft baseline); $O(d^3)$, constant factor ≈ 2d+1 model evaluations.
- `error_state_ekf_predict / _correct / _inject_and_reset` — the production pattern (names ours;
  no Thrun table exists).
- `ekf_on_manifold` / `ukf_m` — generic over `Manifold`; identical asymptotics.

**Numeric micro-example** (unit-test contract): polar→Cartesian pushforward, $r \sim
\mathcal{N}(1, 0.02^2)$, $\theta \sim \mathcal{N}(\pi/2, (15°)^2)$, $f(r,\theta) = (r\cos\theta,
r\sin\theta)$. Analytic truth: $\mathbb{E}[y] = e^{-\sigma_\theta^2/2} = 0.96631$. EKF says
$1.00000$ (bias $3.4$ cm at 1 m). UT ($\alpha_{\mathrm{UT}}{=}1, \beta{=}0, \kappa{=}1$):
$\tfrac{1}{3}(1) + \tfrac{1}{6}(1.03464 + 0.96536) + \tfrac{1}{6}(0.89894 + 0.89894) = 0.96631$ —
matches the analytic value to 5 decimals. Three-line table the reader can verify by hand; the test
also Monte-Carlo-validates with $10^6$ seeded samples.

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphors: **the tangent-line lie** (one curve, one zoom) and **the scouts** (sigma points).
Color mapping for this chapter, stated in a legend on every widget: input/prior belief **blue**;
true pushforward / ground truth **gray**; EKF approximation **orange** (it is a prediction);
sigma points **green** (they are measurements *of the function*); UKF/posterior results **purple**.

- **Widget w7.1: Linearization Lens** — *flagship, interactive wasm-sim.* A nonlinear function
  (selectable: $r\sin\theta$ slice, range-to-landmark, $x^2/20$) with a blue input Gaussian on the
  abscissa. **Manipulates:** headline slider — the operating point $\mu$ (slides along the curve);
  second slider — input spread $\sigma$; toggles for EKF tangent line, UKF scouts, Monte Carlo
  truth cloud; zoom control ("keep zooming until the curve looks straight — that's the EKF's whole
  worldview"). **Observes:** the gray truth histogram going banana-shaped as $\sigma$ grows or the
  operating point hits curvature; the orange EKF ellipse staying stubbornly symmetric; green scouts
  bending with the curve and the purple UKF ellipse landing near truth; a live error readout (mean
  bias + KL estimate) implementing derivation 2's formula. **Misconceptions killed:** "the EKF
  linearizes once, globally" (it re-linearizes at the mean each step — drag and watch the tangent
  move); "UKF is a small particle filter" (scouts are deterministic, weighted, and 2d+1 of them);
  "linearization error is a constant nuisance" (it's curvature × spread — two sliders prove it).
- **Widget w7.2: Manifold vs. Vector** — *flagship, interactive wasm-sim.* Left: Rusty with true
  heading near $\pm\pi$ and a compass; right: the belief over $\theta$ drawn both on a line
  (vector treatment) and on a circle (manifold treatment). **Manipulates:** drag the measured
  heading around the circle; a "fuse" button; toggle *vector update* ↔ *on-manifold update*
  ($\boxplus$); a "drive the loop" autoplay that repeatedly crosses the seam. **Observes:** in
  vector mode, fusing $179°$ and $-179°$ yields $\approx 0°$ — the purple posterior arrow points
  *backwards* out of Rusty; in manifold mode the same fusion lands at $180°$; on the loop drive,
  vector-mode covariance and estimate glitch at every seam crossing while manifold mode is
  seam-free. **Misconception killed:** "wrap-around is a coding edge case to patch with `mod 2π`"
  — it is a representation error; the state was never a real number.
- **Widget w7.3: Error-State Ledger** — *animation.* A wild $SE(2)$ trajectory (nominal state, drawn
  in the world) beside a ledger panel showing the tangent-space error ellipse, which stays small and
  near-Gaussian throughout; inject/reset events flash as the ledger empties into the nominal state.
  **Manipulates:** play/pause; noise scale. **Observes:** the error's world is boring even when the
  state's world is violent — *that* is why linearizing the error works. **Misconception killed:**
  "the EKF must linearize the whole state trajectory" — linearize the small thing instead.
- **Widget w7.4: Filter Chooser** — *static-svg with hover.* Decision chart: curvature × uncertainty
  plane with regions "KF (linear)", "EKF fine", "UKF earns its cost", "iterate (IEKF→Ch. 15/16)",
  "go invariant / re-derive error", annotated with the chapter's demos as data points.
  **Misconception killed:** "UKF > EKF always" (near-linear: same answer, more flops).

Dashboard: w7.1 heads the chapter; w7.2 sits between the EKF and manifold F sections as the pivot
("where 2005 stops and 2026 begins" callout links the exact scroll position); w7.3 accompanies the
error-state derivation; w7.4 closes the chapter.

## 5. Practical (P) — Rust Implementation

**Crates:** `nalgebra` 0.35 (const-generic `SMatrix`; Cholesky for sigma points), `rand` 0.9 /
`rand_distr` 0.6 (seeded `Pcg64` Monte-Carlo truth), `statrs` 0.19 (densities for KL readouts),
`sophus` (pinned minor version; 3D groups referenced in an aside — this chapter's $SE(2)$ is the
hand-rolled Ch. 3 type on purpose), `egui`/`eframe` 0.35 + `egui_plot` 0.34, `plotters` (fallbacks).
`adskalman` deliberately absent: it is linear-only, which is itself a teaching point.

**Module plan:** `crates/ch07_nonlinear/` (library: `Ekf`, `Ukf`, `Eskf`, models — all generic
over Ch. 3's `Manifold` trait), reusing `Manifold` and `SE2` from Ch. 3's `pr-core` `geom`
module; demos in `demos/ch07-demo/`. Depends on `pr-core`, `bayes_core`, `ch06_kalman` (for the
reduction test), `sim`.

**Key types & signatures:**

```rust
use nalgebra::{SMatrix, SVector};

/// Recap — Ch. 3's Manifold trait (it and these impls live in pr-core's geom module);
/// D = tangent dimension.
pub trait Manifold<const D: usize>: Clone {
    fn boxplus(&self, delta: &SVector<f64, D>) -> Self;
    fn boxminus(&self, rhs: &Self) -> SVector<f64, D>;
}

/// Vector spaces are manifolds where ⊞ is +. Ch. 6's Kf is the special case.
impl<const D: usize> Manifold<D> for SVector<f64, D> { /* + and - */ }

/// Ch. 3's hand-rolled SE(2): ⊞ via x·exp(δ), ⊟ via log(x⁻¹y).
impl Manifold<3> for pr_core::geom::SE2 { /* exp/log from pr-core's geom (Ch. 3) */ }

pub struct OnManifoldGaussian<X: Manifold<D>, const D: usize> {
    pub mean: X,
    pub cov: SMatrix<f64, D, D>,   // tangent-space covariance at `mean`
}

pub trait MotionModel<X: Manifold<D>, const D: usize> {
    type Control;
    fn g(&self, x: &X, u: &Self::Control) -> X;
    /// G_t: Jacobian of (g(x ⊞ ε, u) ⊟ g(x, u)) w.r.t. ε at 0 — tangent coordinates.
    fn jac(&self, x: &X, u: &Self::Control) -> SMatrix<f64, D, D>;
    fn noise(&self, u: &Self::Control) -> SMatrix<f64, D, D>;      // R_t
}

pub trait MeasurementModel<X: Manifold<D>, const D: usize, const M: usize> {
    fn h(&self, x: &X) -> SVector<f64, M>;
    fn jac(&self, x: &X) -> SMatrix<f64, M, D>;                    // H_t
    fn noise(&self) -> SMatrix<f64, M, M>;                          // Q_t
}

/// EKF generic over the manifold; with X = SVector this *is* Ch. 6's filter.
pub struct Ekf<X: Manifold<D>, G, H, const D: usize, const M: usize> {
    pub belief: OnManifoldGaussian<X, D>,
    pub motion: G,
    pub meas: H,
}
impl<...> bayes_core::BayesFilter for Ekf<...> { /* predict/correct via ⊞/⊟ */ }

/// UKF-M: sigma points drawn in the tangent space, retracted via ⊞.
pub struct Ukf<X: Manifold<D>, G, H, const D: usize, const M: usize> {
    pub alpha_ut: f64, pub beta: f64, pub kappa: f64,
    /* belief, models as above */
}

/// Error-state EKF: nominal state + tangent error filter + inject/reset.
pub struct Eskf<X: Manifold<D>, const D: usize> { /* nominal: X, err: Gaussian<D> */ }
```

**The deliberate compile error** (TOC promise, shown verbatim in the text): declaring a compass
model `impl MeasurementModel<SE2, 3, 1>` and then calling `correct` with an `SVector<f64, 2>`
measurement — the book prints the actual `rustc` mismatched-types diagnostic and the caption "the
dimension check Thrun does by careful bookkeeping, the type system does at compile time."
A second boxed example: a `jac` returning `SMatrix<f64, 1, 2>` where `SMatrix<f64, 1, 3>` is
required — caught before any filter runs.

**Worked end-to-end example:** `cargo run --example figure_eight -p ch07_nonlinear` drives Rusty on
a figure-eight through the Apartment with unicycle dynamics and compass + range-beacon
measurements, running four filters on the *identical seeded log*: (a) Ch. 6 `Kf` on naive
$(x,y,\theta)$, (b) vector EKF, (c) error-state EKF on `SE2`, (d) UKF-M on `SE2`. Output: RMSE +
final-heading-error table and a plotters SVG of heading error over time — (a) diverges at the first
seam crossing, (b) glitches at each $\pm\pi$ pass, (c)/(d) are seam-free with (d) marginally better
through the high-curvature lobes. Unit tests: the polar→Cartesian micro-example to $10^{-5}$;
`ekf_reduces_to_kf` (on a linear model, `Ekf<SVector<..>>` matches `ch06::Kf` to $10^{-12}$).

**Runnable artifact:** WASM demos w7.1/w7.2 compile from this crate — the `Ukf` computing the purple
ellipse in the Lens is the `Ukf` in the listing. The Integration lab replays the figure-eight in the
browser with a filter-select dropdown.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w7.1 | Linearization Lens | wasm-sim | ch07_nonlinear + eframe/egui_plot, Pcg64 MC truth | drag operating point & σ, toggle EKF/UKF/truth, zoom | linearization error = curvature × spread; UT accuracy; when EKF suffices |
| w7.2 | Manifold vs. Vector | wasm-sim | ch07_nonlinear + geom (SE2) + sim + eframe | drag measured heading, fuse, vector↔manifold toggle, loop autoplay | wrap-around as representation error; ⊞ update; seam-free filtering |
| w7.3 | Error-State Ledger | animation | ch07_nonlinear + eframe | play/pause, noise scale | error stays small & near-linear; inject/reset cycle |
| w7.4 | Filter Chooser | static-svg (hover) | plotters build-time | hover regions | matching filter to curvature/uncertainty regime |
| — | Backwards-posterior hook | animation (autoplay) | ch06_kalman + sim | none (replay) | the 179°/−179° catastrophe motivating the chapter |

## 7. Exercises & Extensions

1. **(F)** Derive the second-order bias term of derivation 2 for $h(x) = x^2/20$ with
   $x \sim \mathcal{N}(\mu, \sigma^2)$, and check it against w7.1's error readout at three
   operating points.
2. **(F)** Prove the unscented transform is exact for affine $f$ (any $\alpha_{\mathrm{UT}}, \kappa$),
   and that for the polar micro-example it reproduces $\mathbb{E}[y]$ to the stated order. Where
   does the $\beta$ term enter and why doesn't it affect the mean?
3. **(F)** Show that for the unicycle motion model on $SE(2)$, the invariant error
   $\eta = \mu^{-1} x$ evolves independently of $\mu$ (group-affine property), and exhibit the
   state-dependent term that appears in the standard EKF's $G_t$ instead.
4. **(C)** Predict-then-verify in w7.2: with the vector update, what posterior heading results from
   prior $170°$ ($\sigma = 20°$) and measurement $-170°$ ($\sigma = 20°$)? Verify, then find the
   prior/measurement pair that maximizes the vector-mode error.
5. **(P)** Implement the iterated EKF measurement update (re-linearize at the posterior mean until
   convergence) as `IekfCorrect` on the same traits; show on the range-beacon model that it matches
   a one-step Gauss-Newton solve — and read Ch. 15's opening with that in mind.
6. **(P)** Add `Manifold<1>` for the circle $S^1$ and reproduce the w7.2 comparison numerically in
   a test: vector fusion error vs. manifold fusion error over 1,000 seeded seam crossings.

## 8. Modernization Notes

- **Where 2005 stops:** the draft baseline contributes exactly one thing here — the Taylor-EKF and
  its honest derivation (§3.3), which we keep in full as the canonical "first fix." Its practical
  considerations (§3.3.4) are preserved but converted from prose warnings into measurable widget
  demonstrations.
- **Not in the draft, sourced from the modernization set:** the unscented transform and UKF (in the
  published 2005 edition but absent from our draft PDF — rebuilt from Barfoot 2nd ed. Ch. 4 and the
  original Julier–Uhlmann papers); everything from §"error-state" onward is post-2005 practice:
  $\boxplus/\boxminus$ parameterization (Solà micro-Lie; Hertzberg et al.'s original boxplus),
  ESKF (Solà notes), on-manifold EKF/UKF incl. UKF-M (Brossard et al.), invariant-EKF intuition
  (Barrau & Bonnabel). This is the book's single largest upgrade over the baseline, per the
  research findings ("the Euler-angle/Euclidean treatment is the largest technical gap").
- **Dropped:** the draft's extended information filter (§3.4.4–3.4.6) — its nonlinear/SLAM role is
  historical and is told in Ch. 15; carrying it here would dilute the manifold arc. Moment-matching
  alternatives beyond UT (cubature/Gauss–Hermite) get one pointer box, not sections — UKF carries
  the pedagogical load alone.
- **Deliberate sequencing choice:** invariant EKF appears as *intuition only* (one worked fact, no
  general derivation) — the full machinery would demand left/right Jacobian calculus this early;
  its payoff is staged for Ch. 14's consistency autopsy and Ch. 16/18's iterated-ESKF systems.
