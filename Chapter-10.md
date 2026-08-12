# Chapter 10 — Probabilistic Sensor Models

> Part III — Probabilistic Models · Estimated length: 10 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

The Bayes filter's other input is $p(z_t \mid x_t, m)$ — and unlike motion, sensing fails in
*categorically different* ways: a LiDAR beam can hit what the map predicts, hit a person the map
doesn't know, fly off to max range on black felt, or return electronics garbage. The hook: we
place Rusty at a *known* pose in the Apartment, record 10,000 real simulated beams down one
corridor, and histogram them — the histogram has a Gaussian hump, an exponential shoulder, a
spike at $z_{max}$, and a uniform floor. The chapter's "aha" is double: (1) a *mixture of four
simple causes* reproduces that messy histogram, and its parameters can be *learned* from data by
EM rather than hand-tuned; (2) the likelihood you compute is almost always **too confident**,
because multiplying hundreds of "independent" beams overcounts evidence — and managing that
overconfidence is as important as the model itself. Four model families are built and then raced
against each other, because Chs. 11–12 will consume them all.

Story line:

1. **Problem** — one pose, thousands of beams, a histogram no single density fits (autoplay).
2. **Intuition** — Beam Mixture Mixer (w10.1): four sliders, four causes, one shape.
3. **Device physics** — why each cause exists, grounded in real sensors (Niku): specular
   ultrasonic bounces, absorptive surfaces, crosstalk, time-of-flight quantization.
4. **Formalism** — beam model + EM intrinsics learning; likelihood fields via distance
   transforms; map correlation; landmark models and the correspondence variable $c_t$.
5. **The lie of independence** — overconfidence demonstrated and mitigated.
6. **Implementation & experiment** — the `sensor` crate; the four models benchmarked on identical
   logs in the Apartment (accuracy vs. smoothness vs. speed).

## 2. Prerequisites & Position

- **Builds on:** Ch. 2 (densities, mixtures, ML estimation), Ch. 4 (LiDAR simulation via
  `parry2d` ray casting; sensor taxonomy and noise phenomenology), Ch. 5 (where
  $p(z_t \mid x_t, m)$ sits in the recursion), Ch. 8 (importance weighting — these likelihoods
  become particle weights), Ch. 9 (its twin chapter).
- **Feeds into:** Ch. 11 (landmark model + $c_t$ drive EKF localization and data association),
  Ch. 12 (beam vs. likelihood-field is *the* practical choice inside MCL; the Theater exposes it
  as a toggle), Ch. 13 (inverse vs. forward models), Ch. 16 (correlation ideas return as scan
  matching), Ch. 25 (learned sensor models replace these, calibrated against them).
- **Baseline sources:** Thrun et al. Ch. 6 (§6.2 maps; §6.3 beam models incl. §6.3.2 intrinsic
  parameter fitting and §6.3.3 derivation; §6.4 likelihood fields; §6.5 correlation; §6.6
  feature/landmark models incl. §6.6.4 sampling poses; §6.7 practical considerations; Tables
  6.1–6.5). Niku Ch. 10 (§10.12 proximity sensors, §10.13 range finders — ultrasonic and
  light-based device physics), Niku Ch. 11 (vision context, deferred to Ch. 18). Modernization:
  Felzenszwalb–Huttenlocher distance transforms; log-likelihood tempering for overconfidence.

## 3. Foundation (F) — Mathematical Core

**Notation introduced**: scan $z_t = \{z_t^1, \ldots, z_t^K\}$; per-beam true range $z_t^{k*}$
(ray cast into $m$); intrinsic parameters
$\Theta = (z_{hit}, z_{short}, z_{max}, z_{rand}, \sigma_{hit}, \lambda_{short})$; likelihood
field distance $d(x, y)$; feature $f_t^i = (r_t^i\ \phi_t^i\ s_t^i)^\top$; correspondence
variable $c_t^i \in \{1..N, \varnothing\}$; landmark $m_j = (m_{j,x}, m_{j,y}, m_{j,s})$;
measurement noise covariance $Q_t = \mathrm{diag}(\sigma_r^2, \sigma_\phi^2, \sigma_s^2)$.

**Definitions & key equations.**

- *Beam model* — four-way mixture per beam:
  $$p(z_t^k \mid x_t, m) = z_{hit}\, p_{hit} + z_{short}\, p_{short} + z_{max}\, p_{max} + z_{rand}\, p_{rand},
  \qquad z_{hit} + z_{short} + z_{max} + z_{rand} = 1$$
  with $p_{hit} = \eta\, \mathcal{N}(z; z^{k*}, \sigma_{hit}^2)$ truncated to $[0, z_{max}]$;
  $p_{short} = \eta\, \lambda_{short} e^{-\lambda_{short} z}$ for $z \le z^{k*}$ (unexpected
  obstacles are *closer* than the map's surface); $p_{max} = \mathbb{1}[z = z_{max}]$ (a point
  mass — stored as a narrow box in discretized range space); $p_{rand} = 1/z_{max}$.
  Scan likelihood under conditional independence:
  $p(z_t \mid x_t, m) = \prod_{k=1}^{K} p(z_t^k \mid x_t, m)$.
- *Device grounding* (Niku §10.13): $p_{short}$ ← people/furniture and ultrasonic specular
  pre-returns; $p_{max}$ ← absorptive/black or specular-away surfaces and out-of-range;
  $p_{rand}$ ← crosstalk between transducers and multi-path; $\sigma_{hit}$ ← timing resolution
  and surface roughness of time-of-flight ranging. One margin table maps each mixture component
  to the physical mechanism, for both ultrasonic and light-based (LiDAR) range finders.
- *Likelihood field*: precompute $d(x,y) = $ distance to nearest occupied cell; project each beam
  endpoint through the pose,
  $$\begin{pmatrix} x_{z^k} \\ y_{z^k} \end{pmatrix} = \begin{pmatrix} x \\ y\end{pmatrix}
    + R(\theta)\begin{pmatrix} x_{k,sens} \\ y_{k,sens}\end{pmatrix}
    + z_t^k \begin{pmatrix} \cos(\theta + \theta_{k,sens}) \\ \sin(\theta + \theta_{k,sens}) \end{pmatrix}$$
  then $q \mathrel{{\times}{=}} \big(z_{hit}\,\mathcal{N}(d(x_{z^k}, y_{z^k}); 0, \sigma_{hit}^2) + z_{rand}/z_{max}\big)$,
  skipping max-range beams. Not a generative model of $z$ (state this honestly); its payoff is
  smoothness of $x \mapsto p(z \mid x, m)$ — no ray casting, no cliff when a beam slips past a
  corner.
- *Map correlation*: build a local map $m_{loc}$ from the scan at pose $x_t$; score
  $$\rho_{m,m_{loc},x_t} = \frac{\sum_i (m_i - \bar m)(m_{loc,i} - \bar m)}
    {\sqrt{\sum_i (m_i - \bar m)^2 \sum_i (m_{loc,i} - \bar m)^2}},\qquad
    p(m_{loc} \mid x_t, m) = \max\{\rho, 0\}$$
  — the ancestor of Ch. 16's scan matching.
- *Landmark model with known correspondence* ($c_t^i = j$):
  $$\hat r = \sqrt{(m_{j,x} - x)^2 + (m_{j,y} - y)^2},\qquad
    \hat\phi = \operatorname{atan2}(m_{j,y} - y,\ m_{j,x} - x) - \theta$$
  $$p(f_t^i \mid c_t^i, x_t, m) = \mathcal{N}(r_t^i - \hat r; 0, \sigma_r^2)\,
    \mathcal{N}(\phi_t^i - \hat\phi; 0, \sigma_\phi^2)\, \mathcal{N}(s_t^i - s_j; 0, \sigma_s^2)$$
  The correspondence variable $c_t$ is introduced *here*, one chapter before it becomes the
  villain of data association (Ch. 11).

**Derivations** (name — statement — sketch — collapsible):

1. **EM for beam intrinsics** (Thrun §6.3.2–6.3.3) — *given ranges $Z$ with known poses/map,
   maximize $\prod_i p(z_i \mid x_i, m)$ over $\Theta$; with latent per-beam causes, EM gives
   closed-form updates.* Sketch (5 steps): (i) introduce cause indicator per beam; (ii) E-step —
   responsibilities $e_{i,c} = \eta\, z_c\, p_c(z_i)$ for $c \in \{hit, short, max, rand\}$;
   (iii) M-step — mixing weights $z_c = \tfrac{1}{|Z|}\sum_i e_{i,c}$;
   (iv) $\sigma_{hit}^2 = \sum_i e_{i,hit}(z_i - z_i^*)^2 \big/ \sum_i e_{i,hit}$,
   $\lambda_{short} = \sum_i e_{i,short} \big/ \sum_i e_{i,short} z_i$; (v) iterate to a local
   optimum, monotone in likelihood. Collapsible: full ML derivation with Lagrange multiplier for
   the simplex constraint; exponential-family view; initialization and degenerate-cluster guards.
2. **Distance transform in two passes** — *the likelihood field's $d(x,y)$ over an
   $N$-cell grid is computable exactly in $O(N)$.* Sketch (3 steps): 1D squared-distance
   lower-envelope-of-parabolas pass; separability row-then-column; take square root once.
   Collapsible: Felzenszwalb–Huttenlocher pseudocode and the proof of exactness. (Not in the
   baseline — it's the modern reason likelihood fields are effectively free.)
3. **Pose sampling from a landmark observation** (Thrun §6.6.4, Table 6.5) — *the landmark model
   can be inverted into a sampler over poses.* Sketch (4 steps): observing $(r, \phi)$ of a known
   landmark constrains the pose to an annulus; sample nuisance angle
   $\hat\gamma \sim \mathcal{U}(0, 2\pi)$, sample $\hat r, \hat\phi$ from their noise densities;
   set $x = m_{j,x} - \hat r\cos\hat\gamma$, $y = m_{j,y} - \hat r\sin\hat\gamma$,
   $\theta = \hat\gamma - \pi - \hat\phi$. Collapsible: change-of-variables factor; why this
   sampler powers Ch. 12's mixture proposal.
4. **Overconfidence from false independence** — *if beams share error causes (unmodeled objects,
   map errors), the product rule overcounts evidence by roughly the effective correlation.*
   Sketch (4 steps): (i) two perfectly correlated beams counted as independent square the
   likelihood ratio; (ii) posterior sharpness grows exponentially in $K$ while information does
   not; (iii) mitigations — subsample every $k$-th beam, inflate $\sigma_{hit}$, or temper:
   $p(z_t \mid x_t, m)^{1/\kappa}$ (log-likelihood scaling), each shown as the same fix in
   different clothes; (iv) measured demo: adjacent-beam residual correlation in the Apartment.
   Collapsible: mis-specified-likelihood view and the choice of $\kappa$ by calibration
   (forward-pointer to Ch. 25's reliability diagrams).

**Named algorithms** (signatures, complexity; $K$ beams, $N$ grid cells, $|Z|$ training beams):

| Algorithm | Signature | Complexity |
|---|---|---|
| `beam_range_finder_model` | $(z_t, x_t, m) \to q$ | $O(K \cdot C_{ray})$, $C_{ray}$ = grid traversal (Table 6.1) |
| `learn_intrinsic_parameters` | $(Z, X, m) \to \Theta$ | $O(|Z| \cdot \text{iters})$ after caching $z^*$ (Table 6.2) |
| `likelihood_field_range_finder_model` | $(z_t, x_t, m) \to q$ | $O(K)$ lookup after $O(N)$ precompute (Table 6.3) |
| `map_correlation_model` | $(m_{loc}, x_t, m) \to \rho$ | $O(N_{loc})$ (§6.5, no baseline table) |
| `landmark_model_known_correspondence` | $(f_t^i, c_t^i, x_t, m) \to q$ | $O(1)$ per feature (Table 6.4) |
| `sample_landmark_model_known_correspondence` | $(f_t^i, c_t^i, m) \to x_t$ | $O(1)$ per sample (Table 6.5) |

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor: **four causes, one histogram**. Likelihood curves and fields are always
**green** (measurement) per the book color code; ground-truth ranges gray dashed; each mixture
component gets a fixed green-family shade used identically in equations, plots, and code comments
(`// p_short — light green`).

- **Widget w10.1: Beam Mixture Mixer** *(flagship — wasm-sim)*. Left: live histogram of simulated
  beams for a chosen corridor pose (the reader can drag the expected range $z^*$ marker or move
  Rusty between three preset poses). Right: the four-component density stacked as tinted areas.
  Manipulables: four mixing sliders (auto-renormalized on the simplex, with a small ternary-ish
  indicator), $\sigma_{hit}$ and $\lambda_{short}$ knobs, a "people in the hallway" toggle that
  changes the *data* (spawns walkers, fattening the short shoulder), and the **Fit (EM)** button —
  parameters animate over EM iterations with a per-iteration log-likelihood sparkline and
  responsibility-colored histogram bars. Observes: which physical situations move which slider;
  EM recovering the truth from data. Autoplay: medium mixture, EM runs once, slowly.
  Misconception killed: *"sensor noise is a Gaussian around the true range."*
- **Widget w10.2: Likelihood Field Explorer** *(flagship — wasm-sim)*. The Apartment rendered as
  a green heatmap of $\log p(z \mid x, m)$ for the current scan (recomputed live over a coarse
  pose grid at fixed $\theta$); the reader drags Rusty, and a side panel shows per-beam
  likelihood bars with the offending beams highlighted when a door edge is crossed. Toggle:
  beam model ↔ likelihood field — the beam model's heatmap is jagged with cliffs, the field's is
  smooth; a $\sigma_{hit}$ slider visibly blurs the field. A "cost" chip shows evaluations/ms for
  each model, live. Misconception killed: *"the more physically faithful model is always the one
  you want in a filter."*
- **Widget w10.3: Beam Autopsy** *(animation)*. One beam, one wall, one walking person: scrub
  time and watch the returned range flick between the four causes, each flash tallying into the
  histogram of w10.1 (linked views). Teaches the generative story cause-by-cause.
- **Widget w10.4: Overconfidence Meter** *(wasm-sim)*. A 1D pose-likelihood curve over a corridor
  slice as beam count $K$ ramps from 4 → 360 (slider): the curve needles down to a spike even
  when the map is subtly wrong (map-error toggle). Second panel: measured correlation of adjacent
  beam residuals. A tempering slider $1/\kappa$ and a subsample-every-$k$ slider both visibly
  restore honest width. Misconception killed: *"more beams always means better localization."*
- **Widget w10.5: Landmark Donut** *(wasm-sim)*. One landmark, one $(r, \phi)$ reading: 500
  sampled poses from Table 6.5 form the annulus/donut; $\sigma_r, \sigma_\phi$ sliders reshape
  it; observing a *second* landmark (toggle) intersects two donuts into two blobs. Sets up
  triangulation intuition and Ch. 12's mixture proposal. Misconception killed: *"one landmark
  observation localizes the robot."*
- **Dashboard layout**: w10.1 full-width at the top of the beam section; w10.3 inline beside the
  mixture equations (linked to w10.1); w10.2 full-width opening the likelihood-field section;
  w10.4 and w10.5 half-width in their sections. Shared chrome: seed, pause, static fallback.

## 5. Practical (P) — Rust Implementation

**Crates**: `nalgebra` 0.35 (poses, small covariances); `parry2d` 0.30 (ray casting against the
Apartment's collider set — same engine as the Ch. 4 simulator, so the model and the world can
share or *disagree on* maps deliberately); `rand` 0.9/`rand_distr` 0.6 (simulated beams, pose
sampling); `statrs` 0.19 (normal pdf/cdf for truncation normalizers); `rayon` (native-only
parallel EM over big logs; single-threaded on WASM); `eframe` 0.35/`egui_plot` 0.34 (widgets);
`plotters` (fallback figures).

**Module plan**: library `crates/sensor/` + demo crate `demos/ch10-widgets/`.

```
crates/sensor/src/
  lib.rs               // SensorModel trait, Scan, re-exports
  beam.rs              // BeamModel, BeamIntrinsics, learn_intrinsics (EM)
  raycast.rs           // ExpectedRanges: cached z* per (pose, map) via parry2d
  likelihood_field.rs  // DistanceField (two-pass EDT), LikelihoodField
  correlation.rs       // local-map construction + correlation score
  landmark.rs          // LandmarkModel, sample_pose_from_landmark
  tempering.rs         // Tempered<S>: wraps any model with 1/kappa + beam subsampling
```

```rust
use nalgebra::Point2;
use pr_core::geom::se2::SE2;
use sim::{World, Scan}; // the Ch. 4 world is the known map here (OccGrid arrives in Ch. 13)

/// Anything Chs. 11–12 can localize with: a log-likelihood of a scan at a pose.
pub trait SensorModel {
    fn log_likelihood(&self, scan: &Scan, x: &SE2, map: &World) -> f64;
}

#[derive(Clone, Copy, Debug)]
pub struct BeamIntrinsics {
    pub z_hit: f64, pub z_short: f64, pub z_max: f64, pub z_rand: f64,
    pub sigma_hit: f64, pub lambda_short: f64, pub max_range: f64,
}

pub struct BeamModel { pub intr: BeamIntrinsics, pub subsample: usize }
impl BeamModel {
    /// Table 6.2: EM over a log of (scan, true pose) pairs. Monotone in log-likelihood.
    pub fn learn_intrinsics(
        log: &[(Scan, SE2)], map: &World, iters: usize,
    ) -> (BeamIntrinsics, Vec<f64> /* per-iter log-lik, plotted by w10.1 */) { /* ... */ }
}

pub struct DistanceField { cells: Vec<f32>, w: usize, h: usize, res: f64 }
impl DistanceField {
    /// Exact Euclidean distance transform, O(N), Felzenszwalb–Huttenlocher.
    pub fn from_map(map: &World) -> Self { /* rasterize walls, then two separable passes */ }
    #[inline] pub fn dist(&self, p: Point2<f64>) -> f64 { /* bilinear lookup */ }
}
pub struct LikelihoodField { pub field: DistanceField, pub sigma_hit: f64, pub z_rand: f64 }

pub struct Landmark { pub xy: Point2<f64>, pub signature: f64 }
pub struct LandmarkModel { pub q: nalgebra::Matrix3<f64> /* Q_t */ }
impl LandmarkModel {
    pub fn log_likelihood(&self, f: &Feature, j: &Landmark, x: &SE2) -> f64 { /* Table 6.4 */ }
    pub fn sample_pose(&self, f: &Feature, j: &Landmark, rng: &mut SmallRng) -> SE2 { /* Table 6.5 */ }
}
```

**Worked end-to-end example** (`cargo run --example fit_and_race`): (1) drive Rusty a fixed
seeded loop through the Apartment with 2 simulated walkers; log 10,000 beams with ground-truth
poses. (2) `learn_intrinsics` from a deliberately wrong initialization; print the recovered
$\Theta$ next to the simulator's true mixture (expected: within a few percent; reproduced by a
unit test with fixed seed). (3) Race all four models on a 500-pose localization-likelihood sweep:
report evaluations/sec, and each model's pose-likelihood peak sharpness and offset. Emits
`race.svg` (grouped bars + one heatmap pair) — the numbers quoted in the chapter prose come from
this exact run.

**Runnable artifact**: WASM demo = w10.1–w10.5; natively, `fit_and_race` reproduces every number
and figure in the chapter. The `Tempered<LikelihoodField>` configuration chosen by the race is
exported as `sensor::defaults::apartment()` — the exact object Ch. 12's MCL Theater constructs.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w10.1 | Beam Mixture Mixer | wasm-sim | sensor, sim, eframe, egui_plot | 4 mixing sliders, σ/λ knobs, people toggle, EM button, pose presets | four-cause mixture; EM intrinsics learning |
| w10.2 | Likelihood Field Explorer | wasm-sim | sensor, sim, parry2d, eframe | drag Rusty; beam↔field toggle; σ_hit slider; cost chip | smoothness vs. fidelity; distance-transform fields |
| w10.3 | Beam Autopsy | animation (wasm) | sensor, sim, eframe | time scrubber; cause tally linked to w10.1 | the generative story per beam |
| w10.4 | Overconfidence Meter | wasm-sim | sensor, eframe, egui_plot | K slider, map-error toggle, tempering + subsample sliders | independence lie; mitigation equivalences |
| w10.5 | Landmark Donut | wasm-sim | sensor, eframe | σ_r/σ_φ sliders; second-landmark toggle; re-roll | pose sampling from features; triangulation preview |
| f10.6 | Model race table + histogram montage | static-svg | sensor, plotters | — (build-time) | fallback for w10.1/w10.2; the benchmark figure |

## 7. Exercises & Extensions

1. **(F)** Derive the truncation normalizer $\eta$ for $p_{hit}$ on $[0, z_{max}]$ in terms of
   the normal CDF, and show what silently breaks in EM if you omit it (which component absorbs
   the missing mass?).
2. **(F)** Write one EM update by hand for a 5-beam toy dataset (numbers given in the text);
   check your responsibilities and M-step against the book's unit test
   `sensor::beam::tests::toy_em_step`.
3. **(C, w10.1)** Turn on "people in the hallway", refit with EM, and predict *before pressing
   Fit* which two parameters move and in which direction. Verify; explain via the device-physics
   table.
4. **(C, w10.4)** Find the smallest tempering $\kappa$ that keeps the 360-beam likelihood peak
   covering the true pose when the map-error toggle is on. Compare with subsampling to 36 beams.
   Which throws away less information, and how would you decide on a real robot?
5. **(P)** Implement `map_correlation_model` and add it to the race harness. Where does it beat
   the likelihood field (hint: map errors), and why is it the odd one out as a probability?
6. **(P, harder)** Extend `LikelihoodField` with per-cell max over a *set* of maps (doors open
   and closed), and demo it in w10.2. Relate to the dynamic-environment handling coming in
   Ch. 12 §dynamic.

## 8. Modernization Notes

- **Added vs. baseline:** the exact $O(N)$ Euclidean distance transform (the 1999 draft
  hand-waves field construction); a measured, widget-backed treatment of overconfidence with
  tempering ($1/\kappa$) presented alongside the baseline's subsample/inflate advice — tempering
  is standard practice in modern MCL implementations; a benchmark harness so the "which model
  when" advice is experimental, not anecdotal; sensor grounding shifted from sonar rings to
  simulated 2D LiDAR (sonar retained as the *explanation* for $p_{short}$/specular physics via
  Niku §10.13.1).
- **Kept:** all five algorithm tables including EM intrinsics learning (often skipped by modern
  courses, but it is the book's first learned-from-data model and seeds Ch. 25); the
  correspondence variable $c_t$ introduced exactly as in the baseline because Ch. 11 depends on
  that framing; the honest admission that likelihood fields are not generative.
- **Dropped/condensed:** Thrun §6.6.1 feature *extraction* (line/corner detectors) is compressed
  to one paragraph — extraction is a perception topic and our simulator emits features directly;
  correlation models get one section and one exercise rather than parity with the other three
  (their descendant, scan matching, gets all of Ch. 16); the baseline's extended
  cone-modeling for wide sonar beams is summarized in the device-physics margin table.
