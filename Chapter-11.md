# Chapter 11 — Localization I: Tracking with Gaussians

> Part IV — Localization · Estimated length: 9 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Part III built the two generative models; this chapter finally *localizes*: given a map, where is
Rusty? The hook is a betrayal. We hand a well-tuned EKF localizer a landmark world and it tracks
beautifully — ellipses breathe in and out as landmarks come into view — until two landmarks stand
close together, one wrong association slips in, and the filter confidently glides off the map,
its ellipse tiny and its estimate wrong. The "aha" is twofold: (1) with *known* correspondence,
EKF localization is just Ch. 7's EKF with Ch. 9/10's models plugged in — a satisfying assembly,
not a new filter; (2) the actual hard problem is the discrete variable $c_t$ — deciding *what you
are seeing* — and distance must be measured in the metric of your uncertainty (Mahalanobis, not
Euclidean). Multi-hypothesis tracking appears as the honest Gaussian answer to ambiguity, and its
cost previews why Part IV needs particles.

Story line:

1. **Taxonomy first** — what "localization" even means: tracking / global / kidnapped, static /
   dynamic, passive / active — and which cell of that grid Gaussians can inhabit.
2. **Markov localization** — the Bayes filter with a map; the Hallway run once more, now formally.
3. **Assembly** — EKF localization with known correspondence: predict with Ch. 9 Jacobians,
   correct per-landmark with Ch. 10's model.
4. **The villain** — unknown correspondence: ML association, Mahalanobis gating, validation
   regions; one wrong gate poisons the filter (w11.1).
5. **Hedging** — MHT: a mixture of EKFs, weights as association likelihoods, pruning.
6. **Implementation & experiment** — `EkfLocalizer` in the Apartment's landmark layer; NEES
   consistency plots; the failure gallery that motivates Ch. 12.

## 2. Prerequisites & Position

- **Builds on:** Ch. 5 (Bayes filter), Ch. 6 (KF machinery, innovation, gain), Ch. 7 (EKF,
  error-state on-manifold update — the substrate `EkfLocalizer` runs on), Ch. 9 (velocity model;
  its Jacobians $G_t, V_t$ and control-noise covariance $M_t$ are derived *here*, in D1),
  Ch. 10 (landmark model, $Q_t$, correspondence variable $c_t$).
- **Feeds into:** Ch. 12 (everything Gaussians cannot do: global localization, kidnapping),
  Ch. 14 (EKF SLAM = this chapter with the map promoted into the state; ML association and gating
  reappear verbatim), Ch. 15–16 (robust back-ends as the modern answer to data association),
  Ch. 22 (active localization as decision making).
- **Baseline sources:** Thrun et al. Ch. 7 (§7.2 taxonomy; §7.3–7.4 Markov localization;
  §7.5 EKF localization incl. §7.5.3 derivation; §7.6 correspondence estimation incl. §7.6.2
  derivation; §7.7 MHT; §7.8 practical considerations; Tables 7.1–7.3). Modernization: on-manifold
  error-state formulation (Solà; Barfoot); χ² gating tables; JCBB (Neira & Tardós) as pointer.

## 3. Foundation (F) — Mathematical Core

**Notation introduced**: taxonomy axes (tracking/global/kidnapped; static/dynamic;
passive/active); control-noise covariance $M_t$ and control Jacobian $V_t$ (velocity model);
per-landmark predicted measurement $\hat z_t^j$; innovation covariance
$S_t^j = H_t^j \bar\Sigma_t (H_t^j)^\top + Q_t$; Mahalanobis distance
$d_M^2(z, \hat z^j) = (z - \hat z^j)^\top (S^j)^{-1} (z - \hat z^j)$; gate threshold
$\gamma_g = \chi^2_{d,1-\epsilon}$; hypothesis set $\{(\mu_t^{(h)}, \Sigma_t^{(h)}, w_t^{(h)})\}$.

**Definitions & key equations.**

- *The localization taxonomy* (three binary-ish axes, presented as a table the rest of Part IV
  fills in): local tracking (unimodal prior) vs. global (uniform prior) vs. kidnapped (wrong
  confident prior); static vs. dynamic environment; passive estimation vs. active control.
  Position each algorithm of Chs. 11–12 in this grid — EKF lives strictly in
  (tracking, static-ish, passive).
- *Markov localization* — the Bayes filter conditioned on a map:
  $$\overline{bel}(x_t) = \int p(x_t \mid u_t, x_{t-1}, m)\, bel(x_{t-1})\, dx_{t-1},\qquad
    bel(x_t) = \eta\, p(z_t \mid x_t, m)\, \overline{bel}(x_t)$$
  with initial belief = point mass (tracking), uniform (global), or wrong point mass (kidnapped).
- *EKF localization, known correspondence* — prediction (velocity model, $\omega \neq 0$ branch):
  $$\bar\mu_t = g(u_t, \mu_{t-1}),\qquad
    \bar\Sigma_t = G_t \Sigma_{t-1} G_t^\top + V_t M_t V_t^\top,\qquad
    M_t = \begin{pmatrix} \alpha_1 v^2 + \alpha_2 \omega^2 & 0 \\ 0 & \alpha_3 v^2 + \alpha_4 \omega^2 \end{pmatrix}$$
  where $G_t = \partial g / \partial x_{t-1}$, $V_t = \partial g / \partial u_t$ evaluated at
  $\mu_{t-1}$. Per observed feature $f_t^i$ with $c_t^i = j$:
  $$q = (m_{j,x} - \bar\mu_{t,x})^2 + (m_{j,y} - \bar\mu_{t,y})^2,\qquad
    \hat z_t^i = \begin{pmatrix} \sqrt{q} \\ \operatorname{atan2}(m_{j,y} - \bar\mu_{t,y},\ m_{j,x} - \bar\mu_{t,x}) - \bar\mu_{t,\theta} \\ m_{j,s} \end{pmatrix}$$
  $$H_t^i = \frac{\partial h}{\partial x}\Big|_{\bar\mu_t},\qquad
    S_t^i = H_t^i \bar\Sigma_t (H_t^i)^\top + Q_t,\qquad
    K_t^i = \bar\Sigma_t (H_t^i)^\top (S_t^i)^{-1}$$
  $$\bar\mu_t \leftarrow \bar\mu_t + K_t^i (z_t^i - \hat z_t^i),\qquad
    \bar\Sigma_t \leftarrow (I - K_t^i H_t^i)\, \bar\Sigma_t$$
  (angle residuals wrapped via $\boxminus$ — the Ch. 7 manifold discipline, stated as *the* fix
  for the heading wrap bug the baseline never mentions).
- *ML correspondence*:
  $\hat c_t^i = \arg\max_j\ \mathcal{N}\!\big(z_t^i;\ \hat z_t^j,\ S_t^j\big)
   = \arg\min_j\ \big[ d_M^2(z_t^i, \hat z_t^j) + \log\det S_t^j \big]$,
  accepted only if $d_M^2 \le \gamma_g$ with $\gamma_g = \chi^2_{2,0.95} \approx 5.99$
  (range-bearing) or $\chi^2_{3,0.95} \approx 7.81$ (with signature); otherwise the feature is
  discarded as an outlier. The *validation region* is the ellipse
  $\{z : d_M^2(z, \hat z^j) \le \gamma_g\}$ — the geometric object w11.1 draws.
- *MHT* — belief as a Gaussian mixture
  $bel(x_t) = \tfrac{1}{\sum_h w_t^{(h)}} \sum_h w_t^{(h)}\, \mathcal{N}(x_t; \mu_t^{(h)}, \Sigma_t^{(h)})$;
  each hypothesis branches over admissible associations; weight recursion
  $w_t^{(h,j)} = w_{t-1}^{(h)} \cdot \mathcal{N}(z_t; \hat z_t^{j,(h)}, S_t^{j,(h)})$;
  prune $w < w_{min}$ relative to the best, cap at $H_{max}$, merge near-duplicates.

**Derivations** (name — statement — sketch — collapsible):

1. **EKF localization from the Bayes filter** (Thrun §7.5.3) — *substituting the linearized
   Ch. 9/10 models into the Gaussian Bayes filter yields Table 7.2.* Sketch (5 steps):
   (i) start from Markov localization; (ii) linearize $g$ about $\mu_{t-1}$
   (Jacobians $G_t, V_t$; map control noise via $V_t M_t V_t^\top$ — noise lives in control
   space, rank 2, hence the $\hat\gamma$ discussion in Ch. 9); (iii) closed Gaussian prediction;
   (iv) linearize $h$ about $\bar\mu_t$ per landmark; (v) standard KF correction with $S_t, K_t$.
   Collapsible: full Jacobian entries for $G_t, V_t, H_t$ (worked, since every implementer
   transcribes them), and the sequential-update caveat (re-linearize between features or not).
2. **ML correspondence as marginal maximization** (Thrun §7.6.2) — *choosing
   $\hat c_t = \arg\max p(z_t \mid c_t, z_{1:t-1}, u_{1:t}, m)$ reduces, under the EKF posterior,
   to minimizing Mahalanobis distance plus a log-determinant term.* Sketch (4 steps):
   (i) marginalize the pose out of the measurement likelihood → predictive
   $\mathcal{N}(\hat z^j, S^j)$; (ii) take logs; (iii) note the often-dropped $\log\det S^j$
   tie-breaker (we keep it — the widget shows a case where dropping it flips the winner);
   (iv) per-feature independent maximization and when that factorization is wrong (shared pose →
   joint association; JCBB pointer). Collapsible: full marginalization algebra; χ² gate
   probability calculation; expected number of false gates vs. clutter density.
3. **Why one wrong association is catastrophic** — *an accepted wrong match injects a biased
   innovation with a confident $S$; the posterior mean moves toward the wrong landmark and the
   covariance shrinks, making the next wrong gate more likely.* Sketch (3 steps): bias
   propagation through $K_t$; positive feedback via shrinking gates; contrast with an outlier
   *test* that inflates uncertainty. Collapsible: a two-step worked numeric example (the same
   numbers w11.1's "poison" preset replays).
4. **MHT weight recursion and pruning bound** — *exact Bayesian bookkeeping over association
   histories is a tree of Gaussians; pruning trades exactness for tractability.* Sketch (3
   steps): branch–weight–normalize; exponential growth $O(J^{T})$; ratio + cap pruning with the
   guarantee that pruned mass is bounded by the ratio threshold per step. Collapsible: weight
   normalization details, merge criterion (Mahalanobis between hypothesis means).

**Named algorithms** ($n$ map landmarks, $N$ observed features, $H$ hypotheses):

| Algorithm | Signature | Complexity |
|---|---|---|
| `Markov_localization` | $(bel(x_{t-1}), u_t, z_t, m) \to bel(x_t)$ | representation-dependent (Table 7.1) |
| `EKF_localization_known_correspondences` | $(\mu_{t-1}, \Sigma_{t-1}, u_t, z_t, c_t, m) \to (\mu_t, \Sigma_t, p_{z})$ | $O(N)$ 3×3 updates (Table 7.2) |
| `EKF_localization` | $(\mu_{t-1}, \Sigma_{t-1}, u_t, z_t, m) \to (\mu_t, \Sigma_t)$ | $O(N \cdot n)$ gate evaluations (Table 7.3) |
| `mht_localization` | $(\{\mu, \Sigma, w\}^{(h)}, u_t, z_t, m) \to$ pruned set | $O(H \cdot N \cdot n)$ + prune/merge (no baseline table; ours) |

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor: **the gate**. Uncertainty defines a private metric; everything the reader drags
is judged inside ellipses. Color code throughout: prior ellipse **blue**, predicted ellipse and
predicted measurements **orange**, incoming measurements **green**, posterior ellipse **purple**,
ground truth **gray dashed**.

- **Widget w11.1: Association Gate** *(flagship — wasm-sim)*. A landmark field (5 landmarks, two
  deliberately close); the predicted measurement $\hat z^j$ of each landmark shown in orange with
  its validation ellipse (the $S^j$ gate). The reader drags a green measurement dot anywhere in
  measurement space (a range-bearing polar pane with the map pane linked beside it).
  Manipulables: drag $z$; toggle *Euclidean ↔ Mahalanobis* nearest neighbor (the winning landmark
  highlights — and *changes* in a marked region of the pane); gate-confidence slider
  ($\epsilon$: 0.90/0.95/0.99, ellipses swell); clutter-rate slider spawning false detections;
  **Poison** button — runs a scripted 20-step tracking sequence where one forced wrong
  association at the close pair sends the purple ellipse confidently off the gray-dashed truth,
  with an error-vs-time strip chart below. Observes: direction-dependent distance; gates
  admitting/rejecting; the poisoning cascade (shrinking ellipse + growing error = the definition
  of inconsistency). Autoplay: slow orbit of $z$ around the close pair, association flickering at
  the boundary. Misconception killed: *"nearest landmark = nearest in meters."* Static fallback:
  two-panel figure (metric flip region; poison run error plot).
- **Widget w11.2: EKF Localization Lab** *(flagship — wasm-sim)*. The Apartment's landmark layer;
  Rusty drives a preset loop (or arrow keys). Purple posterior ellipse breathing: swelling
  through the landmark-free corridor, snapping tight when a landmark enters the sensor cone
  (cone drawn; per-landmark innovation vectors flash green→purple on update). Manipulables:
  sensor range/FOV sliders, $\sigma_r/\sigma_\phi$ knobs, "known correspondence" toggle (the
  safety rails), landmark-density preset (sparse/dense/pathological-pair). A NEES strip chart
  runs underneath with the 95% consistency band shaded. Misconception killed: *"a small ellipse
  means a good estimate"* — the NEES chart exposes confident-and-wrong. 
- **Widget w11.3: Taxonomy Grid** *(animation)*. The 3-axis taxonomy as a clickable grid; each
  cell plays a 5-second canned Hallway/Apartment clip (tracking OK / global impossible for one
  Gaussian / kidnap disaster) and stamps which chapter solves it. Orients the whole Part.
- **Widget w11.4: Hypothesis Forest** *(wasm-sim)*. A symmetric corridor with twin landmark
  pairs: MHT belief drawn as up to 8 translucent purple ellipses with weight-proportional
  opacity; a tree diagram of association histories grows beside the map, pruned branches graying
  out. Manipulables: prune-ratio slider, $H_{max}$, an "observe disambiguating landmark" button
  that collapses the mixture. Observes: hypotheses born at ambiguity, killed by evidence; cost
  counter (Gaussians maintained) ticking. Misconception killed: *"the filter must decide
  immediately"* — deferring the decision is an option with a price.
- **Dashboard layout**: w11.3 opens the chapter (small, full-width strip); w11.2 is the section
  centerpiece for EKF localization; w11.1 anchors the correspondence section with the derivation
  beside it; w11.4 closes before the practical section. Shared chrome: seed, pause, fallback.

## 5. Practical (P) — Rust Implementation

**Crates**: `nalgebra` 0.35 (`SMatrix`/`SVector`; 3×3 covariance blocks, const-generic
measurement dim); `rand` 0.9 (simulation only — the localizer itself is deterministic);
`statrs` 0.19 (χ² quantiles for gates); `sim` (Apartment landmark layer + sensor cone);
`motion`/`sensor` (Ch. 9/10 crates — imported, not reimplemented); `eframe` 0.35 +
`egui_plot` 0.34 (widgets, NEES charts); `plotters` (fallbacks); `adskalman` (dev-dependency
cross-check on the linear part, per book policy).

**Module plan**: library `crates/localize/` (this chapter populates `ekf/`; Ch. 12 adds
`grid/`, `mcl/`) + demo crate `demos/ch11-widgets/`.

```
crates/localize/src/
  lib.rs            // Localizer trait: predict(u) / correct(z) / belief()
  ekf/mod.rs        // EkfLocalizer
  ekf/jacobians.rs  // G_t, V_t, M_t, H_t — each fn maps to an equation number
  ekf/associate.rs  // Mahalanobis, gates, ml_associate
  ekf/mht.rs        // feature "mht": hypothesis set, branch/prune/merge
```

```rust
use nalgebra::{Matrix3, SMatrix, SVector};
use statrs::distribution::{ChiSquared, ContinuousCDF};
use pr_core::geom::se2::SE2;
use motion::{VelocityModel, VelocityCmd};
use sensor::{Feature, Landmark, LandmarkModel};

pub struct GaussianBelief { pub mean: SE2, pub cov: Matrix3<f64> } // cov in tangent space

pub struct EkfLocalizer {
    pub bel: GaussianBelief,
    pub motion: VelocityModel,          // Ch. 9 model; G_t, V_t, M_t built in ekf/jacobians.rs
    pub sensor: LandmarkModel,          // supplies h, Q_t (Ch. 10); H_t built in ekf/jacobians.rs
    pub landmarks: Vec<Landmark>,       // the known map
    pub gate: Gate,                     // chi^2 threshold, dof = Z
}

pub enum Association { Matched { landmark: usize, d2: f64 }, Outlier }

impl EkfLocalizer {
    /// Table 7.2 lines 2–4: on-manifold prediction (boxplus mean, tangent covariance).
    pub fn predict(&mut self, u: &VelocityCmd);

    /// Table 7.2 lines 5–14 (known c) / Table 7.3 (ML association + gating).
    pub fn correct_known(&mut self, f: &[(Feature, usize /* c_t^i */)]);
    pub fn correct(&mut self, f: &[Feature]) -> Vec<Association>;
}

/// arg-min over landmarks of d_M^2 + ln det S; None if best d2 exceeds the gate.
pub fn ml_associate<const Z: usize>(
    z: &SVector<f64, Z>,
    predictions: &[(SVector<f64, Z>, SMatrix<f64, Z, Z>)], // (z_hat^j, S^j)
    gate2: f64,
) -> Association;

#[cfg(feature = "mht")]
pub struct MhtLocalizer {
    pub hyps: Vec<(GaussianBelief, f64 /* w */)>,
    pub prune_ratio: f64, pub max_hyps: usize, pub merge_d2: f64,
}
```

Const generics carry the measurement dimension ($Z = 2$ range-bearing, $Z = 3$ with signature),
so mixing a 2D gate with a 3D innovation is a compile error — the chapter shows that error
verbatim, continuing the Ch. 7 tradition.

**Worked end-to-end example** (`cargo run --example ekf_loc_lab`): seeded 120 s Apartment loop,
12 landmarks, known-correspondence off. Prints: RMSE (position, heading), mean NEES with 95%
bounds, association precision/recall against ground-truth correspondences, outliers rejected.
Expected (fixed seed, reproduced by unit test): NEES within bounds for the default run; then the
same command with `--pathological-pair` reproduces the poison run — RMSE explodes while NEES
*leaves* the band, and the final printed line points the reader to Ch. 12. Emits
`ekf_loc.svg` (trajectory + ellipses every 20 steps + NEES strip).

**Runnable artifact**: WASM demo = w11.1–w11.4 (w11.2 and the example share the exact
`EkfLocalizer`); `--features mht` adds the Hypothesis Forest backend. The `Localizer` trait
defined here is the interface Ch. 12 implements twice more, enabling the side-by-side benchmark.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w11.1 | Association Gate | wasm-sim | localize, sensor, eframe, egui_plot | drag z; metric toggle; ε + clutter sliders; Poison button | Mahalanobis vs. Euclidean; validation gates; association poisoning |
| w11.2 | EKF Localization Lab | wasm-sim | localize, motion, sensor, sim, eframe | drive Rusty; FOV/noise sliders; correspondence toggle; NEES strip | ellipse breathing; correction anatomy; consistency vs. confidence |
| w11.3 | Taxonomy Grid | animation (wasm) | sim, eframe | click cells; play clips | the localization problem space; where Gaussians live |
| w11.4 | Hypothesis Forest | wasm-sim | localize (mht), sim, eframe | prune/cap sliders; disambiguate button | MHT branching, pruning, deferred decisions |
| f11.5 | Poison run + NEES figure | static-svg | localize, plotters | — (build-time) | fallback for w11.1/w11.2; the inconsistency exhibit |

## 7. Exercises & Extensions

1. **(F)** Derive $H_t$ for the range-bearing model by hand (all six nonzero entries) and verify
   against `ekf::jacobians::tests::h_matches_finite_difference`. Why is the bearing row's
   dependence on $\sqrt{q}$ the reason distant landmarks constrain heading better than position?
2. **(F)** Compute the probability that at least one of $n = 10$ clutter measurements falls
   inside a 95% gate of area $A$ under uniform clutter density $\lambda$; derive the clutter
   rate at which ML-NN association is wrong more often than right (the w11.1 clutter slider's
   red zone).
3. **(C, w11.1)** Set the gate to 0.99 and clutter high: predict whether widening the gate makes
   poisoning more or less likely, then verify. Now answer the same question for a *tighter* gate
   during the landmark-free corridor stretch of w11.2. Reconcile the two answers.
4. **(C, w11.4)** Find the smallest $H_{max}$ that still recovers the correct hypothesis in the
   symmetric corridor. What information ultimately breaks the symmetry, and what does that
   suggest about *active* localization (Ch. 22 preview)?
5. **(P)** Add per-feature *sequential* re-linearization to `correct` (re-evaluate $H$ and the
   gates after each accepted feature) and measure its effect on association precision in
   `ekf_loc_lab`. When does update order start to matter?
6. **(P, harder)** Implement joint compatibility for feature *pairs* (a 2-feature mini-JCBB):
   gate on the stacked $4{\times}4$ innovation covariance including cross-correlation. Show one
   seeded scenario in the Lab where individual gates accept a wrong pair but the joint gate
   rejects it.

## 8. Modernization Notes

- **Added vs. baseline:** the EKF runs on Ch. 7's error-state/on-manifold substrate — tangent
  covariance and $\boxminus$ residuals fix the heading-wrap and near-$\pi$ bearing bugs the 2005
  presentation silently suffers; NEES-based consistency evaluation (Bar-Shalom-style, now
  standard practice) is threaded through every widget and example rather than an afterthought;
  the $\log\det S$ term in ML association is kept and demonstrated (commonly dropped); explicit
  χ² gate calibration with `statrs`; a JCBB-flavored joint-compatibility exercise pointing at the
  modern data-association literature.
- **Kept:** the full taxonomy (it still organizes the field), Markov localization as the formal
  umbrella, Tables 7.2/7.3 essentially verbatim (they remain the cleanest teaching form), and MHT
  — but reframed as "a Gaussian-mixture stopgap whose real successors are Ch. 12's particles
  (representation) and Ch. 15's robust factor graphs (association)".
- **Dropped/condensed:** the baseline's long UKF-localization variant is omitted (Ch. 7 already
  covers UKF; a margin note says "swap the `Ekf` for the `Ukf`, the trait makes it a one-liner");
  baseline Fig.-heavy Markov-localization illustration is replaced by the already-built Hallway
  widget (Ch. 5) with a map; MHT implementation detail is compressed to one feature-flagged
  module because modern practice solves ambiguity with particles or robust back-ends, and the
  chapter says so plainly.
