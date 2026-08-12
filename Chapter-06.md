# Chapter 6 — Kalman Filters: The Linear-Gaussian World

> Part II — The Bayes Filter Family · Estimated length: 11 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

The Bayes filter of Ch. 5 is exact but uncomputable over continuous state. This chapter buys
computability with two assumptions — linear dynamics, Gaussian everything — and gets in return the
most consequential algorithm in engineering. The "aha" is double: (i) the Kalman filter is *nothing
but* Ch. 5's two steps executed in closed form on $(\mu, \Sigma)$, with the Kalman gain emerging from
the algebra as **precision-weighted trust** — not a tuning knob; (ii) the same Gaussian has a second
coordinate system (information form) in which the *other* filter step is the cheap one, a duality
whose sparsity insight quietly sets up factor graphs (Ch. 15). The chapter ends with the RTS
smoother: the first hint that waiting for future data beats recursion — the book's longest-range
foreshadowing.

Story line:
1. **Problem** — track Rusty's 1D cart on a rail from noisy position pings; the histogram filter
   from Ch. 5 needs $10^4$ cells for centimeter accuracy. Surely two numbers ($\mu, \sigma^2$) suffice?
2. **Play** — w6.1 Kalman Tuning Bench autoplaying: blue prior → orange prediction → green
   measurement tick → purple posterior, every step.
3. **Intuition** — the update as a product of two blobs (w2.2 returns); the gain as a lever whose
   position is set by relative precisions (w6.3).
4. **Formalism** — linear-Gaussian systems; KF derived by completing the square; gain algebra;
   information form and the duality table; matrix inversion lemma; RTS smoother.
5. **Implementation** — `Kf<N, U, M>` on `SMatrix`, `impl BayesFilter`; information filter;
   RTS; cross-validation against `adskalman`.
6. **Experiment** — 1D cart lab (numeric example), 2D constant-velocity tracking in the Apartment
   (velocity inferred though never measured), divergence-on-purpose failure gallery.

## 2. Prerequisites & Position

- **Builds on:** Ch. 2 (Gaussians, moments vs. canonical form, product of Gaussians — w2.1/w2.2),
  Ch. 5 (the recursion and the `BayesFilter` trait this chapter `impl`s), Ch. 4 (cart-on-rail rig
  in the `sim` crate).
- **Feeds into:** Ch. 7 (EKF/UKF generalize every equation here), Ch. 11 (EKF localization),
  Ch. 14 (EKF SLAM inherits the gain machinery), Ch. 15 (information-form duality → sparse
  smoothing; RTS is the first whole-trajectory estimate), Ch. 18 (MSCKF), Ch. 25 (differentiating
  through this exact `Kf`).
- **Baseline sources:** Thrun et al. (draft) Ch. 3 §3.1 (family overview), §3.2 (linear-Gaussian
  systems, KF algorithm Table 3.1, full derivation §3.2.4), §3.4 (information filter §3.4.1–3.4.3,
  inversion lemma Table 3.2). RTS smoother is **not in the draft**: sourced from Barfoot, *State
  Estimation for Robotics* 2nd ed. Ch. 3 (linear-Gaussian batch/smoothing) — modern presentation.
  Pedagogy: bzarg color-coded blobs; kalmanfilter.net numeric discipline; Labbe's Q/R
  experimentation.

## 3. Foundation (F) — Mathematical Core

**Notation introduced:** $\mu_t, \Sigma_t$ (moments), $\xi_t = \Sigma_t^{-1}\mu_t,\ \Omega_t =
\Sigma_t^{-1}$ (canonical/information), $A_t, B_t, C_t$ (system matrices), $R_t$ (motion noise
covariance), $Q_t$ (measurement noise covariance), $K_t$ (Kalman gain), $S_t$ (innovation
covariance). **Call-out box:** in this book — following Thrun — $R$ is *motion* noise and $Q$ is
*measurement* noise, the reverse of most control-theory texts. The widgets label sliders with both
symbol and meaning to defuse the trap permanently.

**Definitions:**
- *Linear-Gaussian system*:
  $x_t = A_t x_{t-1} + B_t u_t + \varepsilon_t,\ \varepsilon_t \sim \mathcal{N}(0, R_t)$;
  $z_t = C_t x_t + \delta_t,\ \delta_t \sim \mathcal{N}(0, Q_t)$; Gaussian prior. Closure claim:
  under these, $bel(x_t)$ stays Gaussian forever — so the filter need only propagate $(\mu, \Sigma)$.
- *Innovation* $z_t - C_t\bar\mu_t$ and *innovation covariance* $S_t = C_t\bar\Sigma_t C_t^\top + Q_t$
  (named here; load-bearing for gating in Ch. 11 and NIS below).
- *Canonical form*: $\Omega = \Sigma^{-1}$, $\xi = \Sigma^{-1}\mu$; log-density is quadratic with
  $\Omega$ as its Hessian — "information = curvature of the log-belief."

**Derivations:**

1. **KF prediction from the convolution integral.** *Statement:*
   $\bar\mu_t = A_t\mu_{t-1} + B_t u_t$, $\bar\Sigma_t = A_t\Sigma_{t-1}A_t^\top + R_t$.
   *Sketch (4 steps):* (i) write Ch. 5's prediction integral with Gaussian motion model and prior;
   (ii) combine exponents into one quadratic in $(x_t, x_{t-1})$; (iii) complete the square in
   $x_{t-1}$ and integrate it out (a Gaussian integral — contributes only a constant);
   (iv) read off the remaining quadratic in $x_t$. *Collapsible:* Thrun §3.2.4's full decomposition,
   plus the slick alternative via linearity of expectation and $\operatorname{Cov}(Ax+b)$ — shown
   second, honestly labeled "why the long way still matters" (it's the template for Ch. 7).
2. **KF measurement update by completing the square.** *Statement:* $K_t =
   \bar\Sigma_t C_t^\top S_t^{-1}$, $\mu_t = \bar\mu_t + K_t(z_t - C_t\bar\mu_t)$,
   $\Sigma_t = (I - K_tC_t)\bar\Sigma_t$. *Sketch (5 steps):* (i) $bel = \eta\, p(z_t|x_t)\,
   \overline{bel}$, both Gaussian → exponent is a sum of two quadratics in $x_t$;
   (ii) collect: posterior information $\Omega_t = \bar\Sigma_t^{-1} + C_t^\top Q_t^{-1}C_t$,
   posterior mean from the linear term; (iii) that *is* the answer — in information form, one line;
   (iv) apply the inversion lemma to return to moments form; (v) simplify to the gain form.
   *Collapsible:* every inversion-lemma manipulation, plus the **Joseph form**
   $\Sigma_t = (I{-}K_tC_t)\bar\Sigma_t(I{-}K_tC_t)^\top + K_tQ_tK_t^\top$ and why it is the
   numerically honest variant (symmetric, PSD-preserving — what our Rust uses).
3. **The gain as precision-weighted trust.** *Statement (scalar case):*
   $K = \bar\sigma^2/(\bar\sigma^2 + \sigma_z^2)$, $\mu = (1{-}K)\bar\mu + K z$, and in precisions:
   $\omega_{\text{post}} = \bar\omega + \omega_z$ (precisions add). *Sketch (3 steps):* specialize
   derivation 2 to $n=m=1$; rewrite as convex combination; invert to precision form. This is the
   section w6.3 animates; the color-coded equation is the chapter's poster.
4. **Matrix inversion lemma** (Sherman–Morrison–Woodbury; Thrun Table 3.2). *Statement* + 2-step
   verification proof (multiply, cancel). Full proof and the Schur-complement view live in
   Appendix B; this chapter only certifies and uses it.
5. **Information filter & duality.** *Statement:* correct: $\Omega_t = \bar\Omega_t + C_t^\top
   Q_t^{-1}C_t$, $\xi_t = \bar\xi_t + C_t^\top Q_t^{-1}z_t$ (additive, cheap, multi-sensor fusion =
   literal addition); predict: $\bar\Omega_t = (A_t\Omega_{t-1}^{-1}A_t^\top + R_t)^{-1}$ (requires
   inversion, expensive). *Sketch:* transcribe derivations 1–2 in $(\xi, \Omega)$; observe the cost
   mirror-flip. Deliverable: the **duality table** (operation × form × cost) that returns verbatim
   in Ch. 15 when sparsity makes information form win. *Collapsible:* full `Information_filter`
   derivation (Thrun §3.4.3).
6. **RTS smoother** (first taste of whole-trajectory estimation; Barfoot). *Statement:* backward
   pass with smoother gain $L_t = \Sigma_t A_{t+1}^\top \bar\Sigma_{t+1}^{-1}$:
   $\mu_{t|T} = \mu_t + L_t(\mu_{t+1|T} - \bar\mu_{t+1})$,
   $\Sigma_{t|T} = \Sigma_t + L_t(\Sigma_{t+1|T} - \bar\Sigma_{t+1})L_t^\top$.
   *Sketch (4 steps):* (i) form the joint Gaussian of $(x_t, x_{t+1})$ given $z_{1:t}$;
   (ii) condition on $x_{t+1}$ (Appendix B conditional-Gaussian identity); (iii) marry with the
   smoothed $x_{t+1}$ marginal (tower rule); (iv) read off the recursion. *Collapsible:* full
   conditional algebra; remark that the same posterior is what Ch. 15 computes by sparse solve.

**Named algorithms (signatures + complexity):**
- `Kalman_filter(µ_{t-1}, Σ_{t-1}, u_t, z_t) → (µ_t, Σ_t)` — Thrun Table 3.1. Cost
  $O(n^2 + m^3 + mn^2)$ per step ($n$ state, $m$ measurement dims); the $m^3$ is $S_t^{-1}$.
- `Information_filter(ξ_{t-1}, Ω_{t-1}, u_t, z_t) → (ξ_t, Ω_t)` — Thrun Table 3.4. Correct $O(n^2)$
  additive; predict $O(n^3)$.
- `RTS_smoother({µ_t, Σ_t}, {µ̄_t, Σ̄_t}, {A_t}) → {µ_{t|T}, Σ_{t|T}}` — backward $O(Tn^3)$ after a
  stored forward pass.
- Consistency metrics (modern practice, used by the bench): NEES
  $(x_t{-}\mu_t)^\top\Sigma_t^{-1}(x_t{-}\mu_t)$ against ground truth (sim only) and NIS
  $(z_t{-}C_t\bar\mu_t)^\top S_t^{-1}(z_t{-}C_t\bar\mu_t)$ (available on real data); $\chi^2$
  envelopes stated, derivation deferred to a collapsible.

**Numeric micro-example** (unit-test contract): 1D cart, $A=B=1$, $C=1$, $R=0.25$, $Q=0.5$, prior
$\mathcal{N}(0,1)$. Step 1: $u_1{=}1 \Rightarrow \overline{bel} = \mathcal{N}(1, 1.25)$;
$z_1{=}1.7 \Rightarrow K = 0.714286$, $bel = \mathcal{N}(1.5, 0.357143)$. Step 2: $u_2{=}1
\Rightarrow \mathcal{N}(2.5, 0.607143)$; $z_2{=}2.1 \Rightarrow K = 0.548387$,
$bel = \mathcal{N}(2.280645, 0.274194)$. Full table printed in the text; the reader can check every
entry with a calculator; the same numbers gate `cargo test`.

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor: **everything is a fuzzy blob; predict slides & stretches it, correct multiplies it
tighter** (bzarg's device, given sliders). Color code strictly: prior blue, prediction orange,
measurement green, posterior purple, ground truth gray dashed — the equations in F use the same
colors per term (this chapter is where the book's color-coded-equation device pays off hardest).

- **Widget w6.1: Kalman Tuning Bench** — *flagship, interactive wasm-sim.* 1D cart on a rail,
  noisy pings; strip-chart of the last 100 steps. **Manipulates:** headline sliders $R$ and $Q$
  (log-scale, labeled "motion noise $R$ — how much I trust my model" / "measurement noise $Q$ — how
  much I trust my sensor"); play/pause + scrub; seed re-roll; preset buttons — *balanced*,
  *sluggish* ($Q$ huge: filter ignores sensors, lags turns), *jittery* ($R$ huge: filter chases
  noise), **divergent-on-purpose** ($R{=}0$ claimed while the simulated cart actually slips:
  covariance collapses, innovations grow, NIS pegs above its $\chi^2$ envelope — the filter is
  confidently wrong). **Observes:** the four color-coded curves; live gain readout $K_t$; NEES/NIS
  meters with 95% envelopes; in **sweep mode** (ladder of abstraction) the bench runs a $(R,Q)$
  grid offline and renders the RMSE surface with the current setting as a draggable dot.
  **Misconceptions killed:** "$K$ is a constant you tune" (watch it settle from the Riccati
  recursion); "smaller noise settings are always better" (divergence preset); "RMSE is the only
  health metric" (NIS catches the overconfident filter that RMSE flatters early).
- **Widget w6.2: Moments vs. Information** — *flagship, interactive.* One 2D belief shown twice:
  left, $(\mu, \Sigma)$ as a purple ellipse; right, $(\xi, \Omega)$ as an information-matrix
  heatmap + vector. **Manipulates:** `predict` / `correct` step buttons; "add second sensor"
  toggle; drag the belief's correlation. **Observes:** a flop-counter animation showing which side
  does real work per operation (predict: left cheap, right inverts; correct: right adds two
  matrices — literally — left runs the gain gauntlet); with two sensors, the right side's updates
  visibly *sum*. **Misconception killed:** "the information filter is a different filter" — same
  posterior, transposed costs; plants the Ch. 15 seed ("what if $\Omega$ were sparse?").
- **Widget w6.3: Gain Lever** — *interactive animation* (Ch. 2's Blob Multiplier, specialized).
  Orange prediction blob and green measurement blob on one axis; the purple posterior between them;
  the gain drawn as a physical lever position between the two means. **Manipulates:** drag either
  blob's width. **Observes:** the lever slides toward whichever blob is *tighter*; precision
  readouts add. **Misconception killed:** "the KF averages prediction and measurement" — it
  precision-weights them, and the weights are derived, not chosen.
- **Widget w6.4: Smoother Rewind** — *animation.* A filtered trajectory (purple, causal) over
  ground truth; press "smooth" and the RTS backward pass sweeps right-to-left, tightening the
  covariance band, with a draggable "knowledge horizon" showing how estimates *in the past* improve
  as future data is admitted. **Misconception killed:** "filtering is the best possible use of the
  data" — it's the best *causal* use; interior states do better with hindsight (Ch. 15's opening
  argument).

Dashboard: w6.1 is the chapter dashboard; w6.3 docks beside the update equations in F; w6.2 sits at
the head of the information-filter section; w6.4 closes the chapter next to the RTS derivation.

## 5. Practical (P) — Rust Implementation

**Crates:** `nalgebra` 0.35 (`SMatrix`/`SVector`, const-generic dims — dimension errors caught at
compile time; Cholesky for sampling sim noise), `rand` 0.9 + `rand_distr` 0.6 (seeded `Pcg64`),
`statrs` 0.19 (Gaussian densities for NIS/NEES $\chi^2$ checks), `adskalman` 0.18 (dev-dependency:
cross-validate our KF *and* RTS numerically), `egui`/`eframe` 0.35 + `egui_plot` 0.34 (bench),
`plotters` (static fallbacks).

**Module plan:** `crates/ch06_kalman/` (library: `Kf`, `InfoFilter`, `rts_smooth`, consistency
metrics; the belief type is Ch. 2's `pr-core` `Gaussian`, reused) and `demos/ch06-demo/`
(w6.1–w6.4). Depends on `pr-core` (Ch. 2), `bayes_core` (Ch. 5) and `sim` (Ch. 4).

**Key types & signatures:**

```rust
use nalgebra::{SMatrix, SVector};
use pr_core::prob::Gaussian; // Ch. 2's moments-form Gaussian — reused, not re-declared

impl<const N: usize> bayes_core::BeliefLike<SVector<f64, N>> for Gaussian<N> { /* mode, entropy */ }

/// Linear Kalman filter: N = state dim, U = control dim, M = measurement dim.
pub struct Kf<const N: usize, const U: usize, const M: usize> {
    pub a: SMatrix<f64, N, N>,
    pub b: SMatrix<f64, N, U>,
    pub c: SMatrix<f64, M, N>,
    pub r: SMatrix<f64, N, N>,   // motion noise      (Thrun's R_t)
    pub q: SMatrix<f64, M, M>,   // measurement noise (Thrun's Q_t)
    pub belief: Gaussian<N>,
}

impl<const N: usize, const U: usize, const M: usize> bayes_core::BayesFilter for Kf<N, U, M> {
    type State = SVector<f64, N>;
    type Control = SVector<f64, U>;
    type Measurement = SVector<f64, M>;
    type Belief = Gaussian<N>;
    fn predict(&mut self, u: &Self::Control) { /* µ̄ = Aµ + Bu; Σ̄ = AΣAᵀ + R */ }
    fn correct(&mut self, z: &Self::Measurement) { /* S, K, Joseph-form Σ update */ }
    fn belief(&self) -> &Gaussian<N> { &self.belief }
}

pub struct InfoFilter<const N: usize, const U: usize, const M: usize> {
    pub xi: SVector<f64, N>,          // ξ = Σ⁻¹µ
    pub omega: SMatrix<f64, N, N>,    // Ω = Σ⁻¹
    /* a, b, c, r, q as above */
}
impl<const N: usize, const U: usize, const M: usize> InfoFilter<N, U, M> {
    pub fn to_moments(&self) -> Gaussian<N>;
    pub fn correct_additive(&mut self, z: &SVector<f64, M>); // Ω += CᵀQ⁻¹C; ξ += CᵀQ⁻¹z
}

/// Backward pass over a stored forward run. `filtered[t]`, `predicted[t+1]` per Ch. 6 F §6.
pub fn rts_smooth<const N: usize>(
    filtered: &[Gaussian<N>],
    predicted: &[Gaussian<N>],
    a: &SMatrix<f64, N, N>,
) -> Vec<Gaussian<N>>;

pub fn nees<const N: usize>(truth: &SVector<f64, N>, bel: &Gaussian<N>) -> f64;
pub fn nis<const N: usize, const M: usize>(/* innovation, S_t */) -> f64;
```

**Worked end-to-end example:** `cargo run --example cart_1d -p ch06_kalman` runs the F-section
numeric table (prints $\bar\mu, \bar\Sigma, K, \mu, \Sigma$ for both steps; `#[test]` asserts to
$10^{-6}$ and cross-checks the same sequence through `adskalman`). `cargo run --example
apartment_track` runs the 2D lab: constant-velocity model ($N{=}4$: $x, y, \dot x, \dot y$;
$M{=}2$: position pings from the Apartment beacon), plotting position error *and* the inferred
velocity — a state never measured, recovered through $\Sigma$'s cross-correlations — followed by an
RTS pass showing interior covariance shrink (plotters SVG artifact).

**Runnable artifact:** WASM demo = w6.1, compiled from this crate's `Kf` — the reader tunes the very
struct they just read. The failure gallery presets (sluggish/jittery/divergent) are `RunConfig`
constants in the library so the text, the widget, and the tests reference identical numbers.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w6.1 | Kalman Tuning Bench | wasm-sim | ch06_kalman + sim + eframe/egui_plot, Pcg64 | R/Q log sliders, presets incl. deliberate divergence, scrub, seed re-roll, RMSE sweep mode | gain dynamics; Q/R trade-off; NEES/NIS; overconfidence |
| w6.2 | Moments vs. Information | wasm-sim | ch06_kalman + eframe | predict/correct steps, add-sensor toggle, drag correlation | canonical-form duality; additive fusion; Ch. 15 seed |
| w6.3 | Gain Lever | interactive animation | ch06_kalman + eframe | drag blob widths | gain = precision-weighted trust |
| w6.4 | Smoother Rewind | animation | ch06_kalman + eframe | play smooth pass, drag knowledge horizon | smoothing beats filtering in the interior |
| — | 1D cart hook | animation (autoplay) | sim + eframe | none | histogram-filter cost motivates parametrization |

## 7. Exercises & Extensions

1. **(F)** Derive the scalar gain $K = \bar\sigma^2/(\bar\sigma^2+\sigma_z^2)$ directly by
   completing the square in the 1D exponent, then show precisions add. Verify with two steps of the
   chapter's numeric table.
2. **(F)** Prove $\Sigma_t$ (and hence $K_t$) never depends on the measurements $z_{1:t}$ —
   only on the matrices. What does this let you precompute offline, and why will this break
   in Ch. 7?
3. **(F)** Using the inversion lemma, show the information-filter correct step and the KF correct
   step produce identical posteriors; count flops for $n{=}12, m{=}2$ and decide which form wins.
4. **(C)** In w6.1, before touching sliders: predict what happens to $K_t$, RMSE, and NIS if $Q$ is
   multiplied by 100. Verify, then find the $(R, Q)$ region of the sweep surface where NIS is
   in-envelope but RMSE is poor, and explain it.
5. **(P)** Implement the naive $\Sigma = (I-KC)\bar\Sigma$ update alongside Joseph form; run the
   cart 10⁶ steps at $f32$ precision, plot minimum eigenvalue of $\Sigma$ over time, and observe
   which one loses positive-definiteness.
6. **(P)** Log an Apartment run to disk, then `rts_smooth` it; report filtered vs. smoothed RMSE
   and the covariance ratio at mid-trajectory. Cross-check against `adskalman`'s smoother in a test.

## 8. Modernization Notes

- **Kept from the baseline, whole:** the completing-the-square derivation (Thrun §3.2.4) — modern
  texts often replace it with the expectation shortcut; we keep the long form because Ch. 7 reuses
  its skeleton for EKF, and show the shortcut second.
- **Added beyond the 1999–2000 draft (and the 2005 edition):** the RTS smoother (from Barfoot 2nd
  ed.) as the deliberate bridge to smoothing-based estimation — its Ch. 15 payoff is the book's
  restructuring thesis; Joseph-form updates and a numerical-hygiene discussion; NEES/NIS
  consistency testing as standard modern practice (the 2005 book tunes by eye); the explicit
  "$R$/$Q$ naming trap" box; cross-validation against a production crate (`adskalman`) as an
  engineering habit.
- **Condensed:** the draft's extended-information-filter material (§3.4.4–3.4.6) — the EIF as a
  *nonlinear* SLAM engine belongs to history and is told honestly in Ch. 15's information-form
  lineage section; here the information filter appears only in its linear form, sized to carry the
  duality insight.
- **Deferred:** the draft's EKF sections (§3.3) move to Ch. 7, where they get the full
  modernization treatment (manifolds, error-state) the 2005 baseline lacks entirely.
