# Chapter 9 — Probabilistic Motion Models

> Part III — Probabilistic Models · Estimated length: 9 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Every filter built in Part II consumed a motion model $p(x_t \mid u_t, x_{t-1})$ as a black box.
This chapter opens the box. The hook: Rusty is commanded to drive a gentle 1-second arc — and we
run that same command 1,000 times in the simulator. The resulting cloud of final poses is not a
tidy ellipse; it is a **banana**, curved and asymmetric, because rotational noise early in the
motion is amplified by the translation that follows. The "aha": motion uncertainty on a pose
manifold is *structured* — the banana is the truth, the Gaussian ellipse an EKF draws is a
caricature, and (the modern punchline) the banana becomes a Gaussian again if you describe the
noise in exponential coordinates. Every sampler written here is the beating heart of every
localization and SLAM demo in the rest of the book.

Story line:

1. **Problem** — dead-reckoning replay: identical commands, wildly different outcomes (autoplay).
2. **Intuition** — play with the Banana Machine (w9.1): each $\alpha$ slider bends, fattens, or
   smears the banana in a distinct, nameable way.
3. **Formalism** — probabilistic kinematics; the velocity model derived from arc geometry; the
   odometry model derived from the rot–trans–rot decomposition; closed-form density vs. sampler.
4. **Algorithm** — Thrun's six algorithm tables, restated and implemented verbatim.
5. **Modern reframe** — noise as a Gaussian in $\mathfrak{se}(2)$ pushed through $\exp$: the
   banana *is* Gaussian, just not in $(x, y, \theta)$ coordinates.
6. **Implementation & experiment** — the `motion` crate; KS-test property tests proving the
   sampler and the closed form agree; map-conditioned motion in the Apartment.

## 2. Prerequisites & Position

- **Builds on:** Ch. 3 (SE(2), $\exp/\log$, $\boxplus/\boxminus$), Ch. 4 (Rusty's differential
  drive, encoders, the `sim` crate), Ch. 5 (the Bayes filter needs $p(x_t \mid u_t, x_{t-1})$),
  Ch. 7 (EKF linearization — the Gaussian caricature we critique), Ch. 8 (samplers feed the
  particle machinery).
- **Feeds into:** Ch. 10 (the other half of the generative story), Ch. 11 (EKF localization uses
  the velocity model's Jacobians), Ch. 12 (MCL uses `sample_motion_model_odometry` for every
  particle), Ch. 14/17 (SLAM prediction steps), Ch. 23 (MPPI rollouts reuse the sampler).
- **Baseline sources:** Thrun et al. Ch. 5 (§5.2 probabilistic kinematics; §5.3 velocity model
  incl. §5.3.3 mathematical derivation; §5.4 odometry model; §5.5 motion and maps; algorithm
  Tables 5.1–5.7). Lynch & Park Ch. 13.3–13.4 (deterministic diff-drive substrate and odometry).
  Modernization set: Solà et al. micro-Lie (arXiv:1812.01537) for exponential-coordinate noise;
  Barfoot Ch. on compounding pose uncertainty ("banana is Gaussian in exponential coordinates").

## 3. Foundation (F) — Mathematical Core

**Notation introduced** (chapter table): pose $x_t = (x\ y\ \theta)^\top$; velocity control
$u_t = (v\ \omega)^\top$ over interval $\Delta t$; odometry control
$u_t = (\bar{x}_{t-1}\ \bar{x}_t)$ with relative decomposition
$(\delta_{rot1}, \delta_{trans}, \delta_{rot2})$; noise parameters $\alpha_1 \ldots \alpha_6$
(velocity) and $\alpha_1 \ldots \alpha_4$ (odometry); final-rotation slack $\hat{\gamma}$;
twist/tangent perturbation $\boldsymbol{\tau} \in \mathfrak{se}(2)$, $R_u$ its covariance
(an $R$, not a $Q$: motion noise, per the book-wide $R$/$Q$ convention of Ch. 6).

**Definitions.**

- *Probabilistic kinematics*: the motion model is the conditional density
  $p(x_t \mid u_t, x_{t-1})$ — the Bayes-filter prediction integrand (Ch. 5).
- *Velocity motion model*: noise enters through perturbed controls
  $\hat{v} = v + \varepsilon_{\alpha_1 v^2 + \alpha_2 \omega^2}$,
  $\hat{\omega} = \omega + \varepsilon_{\alpha_3 v^2 + \alpha_4 \omega^2}$, plus a final rotation
  $\hat{\gamma} = \varepsilon_{\alpha_5 v^2 + \alpha_6 \omega^2}$ (needed so the density has full
  rank on 3-dof poses despite 2-dof controls). $\varepsilon_{b^2}$ is a zero-mean variate of
  variance $b^2$ (normal or triangular).
- *Exact arc update* (noise-free): with $r = v/\omega$,
  $$x' = x - \tfrac{v}{\omega}\sin\theta + \tfrac{v}{\omega}\sin(\theta + \omega\Delta t),\quad
    y' = y + \tfrac{v}{\omega}\cos\theta - \tfrac{v}{\omega}\cos(\theta + \omega\Delta t),\quad
    \theta' = \theta + \omega\Delta t$$
  with the $\omega \to 0$ straight-line limit stated explicitly (and unit-tested).
- *Odometry decomposition*: any relative odometry reading is expressed as rotate–translate–rotate:
  $$\delta_{rot1} = \operatorname{atan2}(\bar{y}' - \bar{y},\ \bar{x}' - \bar{x}) - \bar{\theta},\qquad
    \delta_{trans} = \sqrt{(\bar{x}' - \bar{x})^2 + (\bar{y}' - \bar{y})^2},\qquad
    \delta_{rot2} = \bar{\theta}' - \bar{\theta} - \delta_{rot1}$$
  with noise variances $\alpha_1\delta_{rot1}^2 + \alpha_2\delta_{trans}^2$ (rot1),
  $\alpha_3\delta_{trans}^2 + \alpha_4(\delta_{rot1}^2 + \delta_{rot2}^2)$ (trans),
  $\alpha_1\delta_{rot2}^2 + \alpha_2\delta_{trans}^2$ (rot2).
- *Map-conditioned motion model*:
  $p(x_t \mid u_t, x_{t-1}, m) = \eta\, p(x_t \mid m)\, p(x_t \mid u_t, x_{t-1})$, with
  $p(x_t \mid m) \propto$ indicator of free space — an approximation that checks only the endpoint,
  not the path (state the wall-clipping failure case).
- *On-manifold motion model* (modern form):
  $$x_t = x_{t-1} \boxplus (\boldsymbol{\tau}_t + \mathbf{w}_t),\qquad
    \mathbf{w}_t \sim \mathcal{N}(0, R_u),\qquad \boldsymbol{\tau}_t = \Delta t\,(v,\ 0,\ \omega)^\top$$
  i.e., Gaussian noise in $\mathfrak{se}(2)$, pushed through $\exp$ onto the group.

**Derivations** (each: name — statement — sketch — collapsible content):

1. **Center-of-rotation geometry** — *the exact arc update follows from the ICC (instantaneous
   center of curvature).* Sketch (4 steps): (i) constant $(v,\omega)$ ⇒ circular arc of radius
   $v/\omega$; (ii) ICC sits at $(x - \tfrac{v}{\omega}\sin\theta,\ y + \tfrac{v}{\omega}\cos\theta)$,
   perpendicular to the heading; (iii) rotate the pose about the ICC by $\omega\Delta t$;
   (iv) take $\omega \to 0$ for the degenerate line. Collapsible: full trigonometric expansion,
   the small-$\omega$ Taylor treatment used in code ($|\omega| < 10^{-6}$ branch).
2. **Closed-form velocity density** (Thrun §5.3.3) — *given $(x_{t-1}, x_t)$, the density
   $p(x_t \mid u_t, x_{t-1})$ is computed by inverting the motion*: find the unique arc through
   both positions. Sketch (6 steps): (i) the ICC lies on the perpendicular bisector of the segment
   $x_{t-1}x_t$ *and* on the line perpendicular to the initial heading — intersect them via
   $\mu = \tfrac{1}{2}\tfrac{(x-x')\cos\theta + (y-y')\sin\theta}{(y-y')\cos\theta - (x-x')\sin\theta}$;
   (ii) ICC $x^* = \tfrac{x+x'}{2} + \mu(y-y')$, $y^* = \tfrac{y+y'}{2} + \mu(x'-x)$;
   (iii) $r^* = \|x_{t-1} - (x^*,y^*)\|$, swept angle $\Delta\theta$ via two atan2's;
   (iv) recover $\hat{v} = \tfrac{\Delta\theta}{\Delta t}r^*$, $\hat{\omega} = \tfrac{\Delta\theta}{\Delta t}$,
   $\hat{\gamma} = \tfrac{\theta'-\theta}{\Delta t} - \hat{\omega}$;
   (v) evaluate three independent error densities at $(v-\hat{v}, \omega-\hat{\omega}, \hat{\gamma})$;
   (vi) product = density up to normalization. Collapsible: why the map $(v,\omega,\gamma) \mapsto x_t$
   is a change of variables and where the Jacobian is swept into $\eta$; degeneracy when
   $x_t = x_{t-1}$.
3. **Odometry decomposition sufficiency** — *rot–trans–rot reaches any $\Delta$pose, and the three
   steps are treated as independently noisy.* Sketch (3 steps): construct the decomposition,
   perturb each leg, recompose; note the deliberate fiction (real slip correlates the legs) and
   why it still works. Collapsible: closed-form `motion_model_odometry` density with the same
   invert-then-evaluate pattern.
4. **Sampler ↔ density consistency** — *`sample_*` draws from exactly the density `motion_model_*`
   evaluates.* Sketch: sampling is ancestral (draw noise, push through deterministic kinematics);
   the density is the pushforward measure. Collapsible: formal pushforward argument + the KS-test
   protocol used in the Rust property tests.
5. **The banana is Gaussian in exponential coordinates** — *if $\mathbf{w} \sim \mathcal{N}(0,R_u)$
   in $\mathfrak{se}(2)$, the induced density on SE(2) is exactly Gaussian in the tangent space at
   the predicted mean; the $(x,y)$-marginal crescent is a coordinate artifact.* Sketch (4 steps):
   (i) define the density via $\boxminus$: $p(x_t) \propto \exp(-\tfrac{1}{2}\|x_t \boxminus \bar{x}_t\|^2_{R_u^{-1}})$;
   (ii) show sampling from it reproduces the banana; (iii) compare covariance fitted in
   $(x,y,\theta)$ vs. in tangent coordinates; (iv) preview: this is why Ch. 7's error-state EKF and
   Ch. 15's factor graphs keep bananas honest. Collapsible: left-vs-right perturbation convention,
   compounding over $k$ steps ($R_u$ propagated by the adjoint), Barfoot/Solà references.

**Named algorithms** (Thrun table names, signatures, complexity — all $O(1)$ per call):

| Algorithm | Signature | Notes |
|---|---|---|
| `motion_model_velocity` | $(x_t, u_t, x_{t-1}) \to p$ | Table 5.1; closed-form density |
| `sample_motion_model_velocity` | $(u_t, x_{t-1}) \to x_t$ | Table 5.3; ancestral sampler |
| `motion_model_odometry` | $(x_t, u_t, x_{t-1}) \to p$ | Table 5.5 |
| `sample_motion_model_odometry` | $(u_t, x_{t-1}) \to x_t$ | Table 5.6; MCL's workhorse |
| `prob_normal_distribution` / `prob_triangular_distribution` | $(a, b^2) \to p$ | Table 5.2 |
| `sample_normal_distribution` / `sample_triangular_distribution` | $(b^2) \to \varepsilon$ | Table 5.4; incl. 12-uniform trick as an aside |
| `motion_model_with_map` / `sample_motion_model_with_map` | adds $m$; sampler is rejection-based | Table 5.7; expected $O(1/p_{acc})$ draws |

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor: **the banana**. One growing visual carried end-to-end; sample clouds are always
**orange** (prediction), the previous pose **blue**, ground-truth command arc **gray dashed**,
Gaussian caricatures drawn as outlined ellipses in the same orange (labeled "the lie").

- **Widget w9.1: Banana Machine** *(flagship — interactive sim, wasm)*. The reader drives one
  command (drag an arc handle or pick presets: straight / gentle arc / pivot) and the machine
  instantly samples 1,000 futures from `sample_motion_model_velocity`, rendered as an orange point
  cloud with a faint heading tick per sample. Manipulables: six $\alpha$ sliders (one highlighted
  at a time — hovering a slider flashes the term it controls in the on-screen equation), model
  toggle velocity ↔ odometry (sliders re-skin to $\alpha_1..\alpha_4$), overlay toggles:
  (a) moment-matched Gaussian ellipse in $(x,y)$, (b) exponential-coordinate Gaussian pushed
  through $\exp$ (hugs the banana). Observes: how $\alpha_1$ fattens along-track, $\alpha_4$ fans
  the arc, $\alpha_5/\alpha_6$ blurs headings without moving positions; how overlay (a) leaks
  probability off the banana into unreachable space while (b) does not. Autoplay default: gentle
  arc, medium noise, overlay (a) on. Misconception killed: *"motion noise is an ellipse around the
  predicted pose."* Static fallback: 3-panel banana montage (low/medium/high $\alpha$'s).
- **Widget w9.2: Arc Anatomy** *(animation)*. The exact velocity update as a construction: ICC dot
  appears, radius line sweeps the pose along the arc; a $\Delta t$ scrubber. In closed-form mode
  the reader drags $x_t$ anywhere and watches the inferred $(\hat{v}, \hat{\omega}, \hat{\gamma})$
  and the resulting density value update live (the invert-the-motion derivation made tactile).
  Misconception killed: *"the closed form and the sampler are different models."*
- **Widget w9.3: Odometry Decomposer** *(interactive sim)*. Two draggable poses; the rot–trans–rot
  legs render as a hinged linkage that replays; noise sliders perturb each leg independently and
  ghost-replay 200 samples. Observes: why long translations with sloppy initial rotation produce
  wide arcs of uncertainty. Misconception killed: *"odometry error is additive in $(x,y,\theta)$."*
- **Widget w9.4: Map Squeeze** *(interactive sim)*. The Banana Machine dropped beside a wall and a
  doorway in the Apartment; toggle `motion_model_with_map`. Observes: banana truncated at walls,
  probability renormalized into free space; the doorway case where the endpoint-only approximation
  wrongly keeps samples that clipped a wall (highlighted red). Misconception killed: *"conditioning
  on the map is just clipping."*
- **Dashboard layout**: w9.1 full-width; below it a two-column row (w9.2 | w9.3); w9.4 appears in
  the integration-lab section. A shared seed control + re-roll die button in the widget chrome.

## 5. Practical (P) — Rust Implementation

**Crates**: `nalgebra` 0.35 (poses, covariances, tangent vectors); `rand` 0.9 + `rand_distr` 0.6
(seeded `SmallRng`, `Normal`, custom triangular); `statrs` 0.19 (normal CDF for the KS test);
`sim` (Ch. 4 world + Rusty for data collection); `eframe`/`egui` 0.35 + `egui_plot` 0.34
(widgets); `plotters` (build-time static fallbacks). No new heavyweight dependencies — this
chapter is deliberately close to the metal.

**Module plan**: library `crates/motion/` (used by every later chapter) + demo crate
`demos/ch09-widgets/` (trunk-built eframe app hosting w9.1–w9.4).

```
crates/motion/src/
  lib.rs          // MotionModel trait, re-exports
  velocity.rs     // VelocityModel: prob + sample (Tables 5.1/5.3)
  odometry.rs     // OdometryModel: prob + sample (Tables 5.5/5.6)
  noise.rs        // NoiseKind::{Normal, Triangular}: prob_/sample_ (Tables 5.2/5.4)
  map_cond.rs     // MapConditioned<M> wrapper (Table 5.7)
  se2_noise.rs    // TangentNoiseModel: boxplus-Gaussian, adjoint propagation
```

```rust
use nalgebra::{Vector2, Vector3, Matrix3};
use rand::rngs::SmallRng;
use pr_core::geom::se2::SE2; // Ch. 3: exp/log/boxplus/boxminus

pub struct VelocityCmd { pub v: f64, pub w: f64, pub dt: f64 }
pub struct OdomDelta  { pub rot1: f64, pub trans: f64, pub rot2: f64 }

#[derive(Clone, Copy)]
pub struct VelocityAlphas(pub [f64; 6]);
#[derive(Clone, Copy)]
pub struct OdomAlphas(pub [f64; 4]);

/// Every motion model in the book: a density and a sampler that must agree.
pub trait MotionModel {
    type Control;
    fn prob(&self, x1: &SE2, u: &Self::Control, x0: &SE2) -> f64;
    fn sample(&self, u: &Self::Control, x0: &SE2, rng: &mut SmallRng) -> SE2;
}

pub struct VelocityModel { pub alphas: VelocityAlphas, pub noise: NoiseKind }
pub struct OdometryModel { pub alphas: OdomAlphas,     pub noise: NoiseKind }

impl OdomDelta {
    /// Decompose a relative odometry reading (Table 5.5, lines 2–4).
    pub fn from_poses(prev: &SE2, cur: &SE2) -> Self { /* atan2, hypot */ }
}

/// Modern form: Gaussian in the tangent space, sampled via boxplus.
pub struct TangentNoiseModel { pub r: Matrix3<f64> /* R_u — tangent motion-noise covariance */ }
impl TangentNoiseModel {
    pub fn sample(&self, u: &VelocityCmd, x0: &SE2, rng: &mut SmallRng) -> SE2 {
        let tau = Vector3::new(u.v * u.dt, 0.0, u.w * u.dt);
        x0.boxplus(tau + self.r_cholesky_sample(rng))
    }
    /// log-density via boxminus — no arc inversion needed.
    pub fn log_prob(&self, x1: &SE2, u: &VelocityCmd, x0: &SE2) -> f64 { /* ... */ }
}

pub struct MapConditioned<'m, M> { pub inner: M, pub map: &'m sim::World }
// impl MotionModel: rejection sampling against the world's free space (World::collides),
// density × free-space prior. (OccGrid maps don't exist until Ch. 13.)
```

**Property tests** (`crates/motion/tests/`): (1) sampler-vs-closed-form agreement — draw 50k
samples, evaluate the closed form on a $60\times60\times36$ pose grid, compare marginal CDFs with
a two-sample KS statistic (threshold calibrated, seed fixed); (2) $\omega \to 0$ continuity;
(3) odometry decompose∘recompose is identity; (4) tangent model reduces to the velocity model for
small noise. These tests *are* the chapter's claim that the code is the math.

**Worked end-to-end example** (`cargo run --example banana`): from $x_0 = (0,0,0)$, command
$v = 1\,\text{m/s}, \omega = 0.5\,\text{rad/s}, \Delta t = 1\,\text{s}$, $\alpha = $ the chapter's
canonical set; print sample mean/covariance in $(x,y,\theta)$ *and* in tangent coordinates
(the latter visibly diagonal-ish), emit `banana.svg` via plotters with the two Gaussian overlays.
Expected output (reproduced by a unit test): tangent-space covariance within 3% of $R_u$;
$(x,y)$-covariance skewed with correlation $\rho_{x\theta} \approx$ the value printed in the text.

**Runnable artifact**: the WASM demo is w9.1–w9.4 in one page; the same `motion` crate compiled
natively powers the SVG figures. Ch. 12's MCL Theater imports this crate unchanged.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w9.1 | Banana Machine | wasm-sim | motion, sim, eframe, egui_plot | drag command arc; 6 α sliders; model toggle; overlay toggles; re-roll seed | structured motion noise; banana vs. Gaussian caricature; α semantics |
| w9.2 | Arc Anatomy | animation (wasm) | motion, eframe | Δt scrubber; drag endpoint in closed-form mode | exact arc geometry; invert-the-motion closed form |
| w9.3 | Odometry Decomposer | wasm-sim | motion, eframe | drag two poses; 4 α sliders; ghost replays | rot–trans–rot decomposition; compounding rotation error |
| w9.4 | Map Squeeze | wasm-sim | motion, sim, parry2d, eframe | toggle map conditioning; drag start pose near wall | map-conditioned motion; endpoint-approximation failure |
| f9.5 | Banana montage | static-svg | motion, plotters | — (build-time) | fallback for w9.1; print figure |

## 7. Exercises & Extensions

1. **(F)** Derive the $\omega \to 0$ limit of the exact arc update via Taylor expansion and show
   the closed-form density remains well defined; state the numerical branch condition you would
   use in code and why.
2. **(F)** Show that in the velocity model, marginal position uncertainty after a straight-line
   command grows linearly in $\Delta t$ for translation noise but quadratically for the rotation
   noise's effect on lateral error — the algebraic reason bananas curve.
3. **(C, w9.1)** Predict-then-verify: set all $\alpha$'s to zero except $\alpha_4$ (velocity
   model). Sketch the sample cloud you expect for a straight 2 m command, then check. Repeat for
   only-$\alpha_6$. Explain the difference in one sentence each.
4. **(C, w9.4)** Find a start pose and command near the Apartment doorway where map-conditioned
   sampling accepts a pose whose *path* was infeasible. Propose a fix and its cost.
5. **(P)** Implement `sample_triangular_distribution` and swap it into `VelocityModel`; re-run the
   KS property test and the Banana Machine. Where does the triangular banana visibly differ?
6. **(P, harder)** Implement compounding: propagate `TangentNoiseModel` covariance over a 20-step
   trajectory using the adjoint, and compare against 50k Monte Carlo rollouts. Plot tangent-space
   NEES over time; report where the second-order approximation starts to fail.

## 8. Modernization Notes

- **Added vs. 1999/2005 baseline:** the entire on-manifold treatment — $\boxplus$ noise in
  $\mathfrak{se}(2)$, the "banana is Gaussian in exponential coordinates" result, adjoint
  covariance compounding — is post-2005 practice (Solà, Barfoot) and is what Ch. 7's error-state
  EKF and Part V's factor graphs consume. The baseline draws the banana but has no language for
  why it is *tame*.
- **Added:** property-based sampler↔density testing (KS), reproducible seeds, and the explicit
  statement of the closed form's change-of-variables caveat, which the baseline buries.
- **Kept verbatim in spirit:** all seven algorithm tables and both derivations — they remain the
  cleanest pedagogical path and MCL still runs on `sample_motion_model_odometry` in 2026 (AMCL in
  Nav2). Alpha-parameter treatment is kept complete (all six), since tuning them is a live skill.
- **Dropped/condensed:** the sum-of-12-uniforms sampling trick is a one-line historical aside
  (`rand_distr::Normal` is exact via Ziggurat); the baseline's extended discussion of triangular
  vs. normal noise is compressed into one exercise; Thrun's §5.5 map-conditioned model keeps its
  approximation warning but the "overlap of banana and free space" figure becomes a widget (w9.4)
  instead of prose. Encoder-tick-level odometry modeling stays in Ch. 4 where the device lives.
