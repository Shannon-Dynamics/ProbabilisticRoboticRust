# Chapter 12 — Localization II: Global Localization with Grids and Particles

> Part IV — Localization · Estimated length: 12 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Chapter 11 ended with a confession: a Gaussian cannot say "I could be in any of these four
rooms." This chapter is the payoff of the whole first half of the book — the Bayes filter
(Ch. 5), nonparametric representations (Ch. 8), motion samplers (Ch. 9), and sensor likelihoods
(Ch. 10) assemble into the two algorithms that actually solve global localization: grid
localization and Monte Carlo Localization. The hook is the book's centerpiece demo: Rusty wakes
up somewhere in the Apartment with 20,000 particles spread over every room; within fifteen
seconds of driving, the cloud collapses through a two-room ambiguity into a single confident
hypothesis — and then we press the **Kidnap** button and watch plain MCL fail, confidently, and
Augmented MCL save itself by noticing its own surprise. The "aha": MCL is not a new filter — it
is `Particle_filter(sample_motion_model, measurement_model)`, three lines of glue — and the
engineering around it (recovery, adaptive proposals, dynamic-world filtering) is where the 2026
production localizer AMCL actually lives. The chapter closes by *reproducing Thrun's comparison
table as an experiment*, racing every localizer of Part IV on identical logs.

Story line:

1. **Problem** — global localization: the uniform prior no Gaussian can represent (autoplay: the
   Ch. 11 EKF initialized uniformly-ish, failing instantly).
2. **Brute force** — grid localization: the Ch. 8 discrete Bayes filter with Ch. 9/10 models;
   resolution vs. cost vs. correctness.
3. **Elegance** — MCL: the Ch. 8 particle filter instantiated; watch it condense (w12.1).
4. **Humility** — kidnapping breaks MCL; Augmented MCL's $w_{fast}/w_{slow}$ self-surprise
   detector; mixture proposals for peaked sensors.
5. **The real world moves** — localization among walking people: novelty-filtering measurements.
6. **Science** — the great comparison table, measured, not asserted.

## 2. Prerequisites & Position

- **Builds on:** Ch. 5 (Bayes filter), Ch. 8 (particle filter, low-variance resampler, ESS,
  KLD-adaptive sizing, binary/discrete grids), Ch. 9 (`sample_motion_model_odometry` — every
  particle's predict step), Ch. 10 (beam vs. likelihood-field models as the weight function;
  `sample_pose` from landmarks for mixture proposals), Ch. 11 (taxonomy; the Gaussian failure
  that motivates everything here; the `Localizer` trait).
- **Feeds into:** Ch. 13 (a localized robot can map), Ch. 17 (FastSLAM = MCL with per-particle
  maps), Ch. 22 (active localization decides *where to drive* to make w12.1's ambiguity
  collapse), Ch. 25 (learned likelihoods dropped into the same MCL), Ch. 26 (the capstone's
  localization subsystem is literally this crate's `AugmentedMcl`).
- **Baseline sources:** Thrun et al. Ch. 8 (§8.2 grid localization incl. resolution and
  pre-caching; §8.3 MCL incl. §8.3.3 random-particle recovery and §8.3.4 proposal modification;
  §8.4 dynamic environments; §8.5 practical considerations; Tables 8.1–8.5; the chapter's
  exercises). Ch. 8 of the 2005 edition contributes KLD-sampling (already built in our Ch. 8).
  Modernization: AMCL as Nav2's default localizer (2026 status); beam-skipping and tempering
  practice; SLAM-Toolbox localization mode as the pose-graph alternative (pointer).

## 3. Foundation (F) — Mathematical Core

**Notation introduced**: grid cells $\{\mathbf{x}_k\}$ with beliefs $\{p_{k,t}\}$; particle set
$\mathcal{X}_t = \{x_t^{[i]}, w_t^{[i]}\}_{i=1}^{M}$; average weight $w_{avg}$; smoothed
likelihoods $w_{fast}, w_{slow}$ with gains $\alpha_{fast} \gg \alpha_{slow}$; mixing rate
$\phi$ (fraction of measurement-proposed particles); novelty test statistic for beam $z_t^k$.

**Definitions & key equations.**

- *Grid localization*: the discrete Bayes filter (Ch. 8) over a regular decomposition of pose
  space (typical: 15 cm × 15 cm × 5°):
  $$\bar p_{k,t} = \sum_j p(\mathbf{x}_k \mid u_t, \mathbf{x}_j)\, p_{j,t-1},\qquad
    p_{k,t} = \eta\, p(z_t \mid \mathbf{x}_k, m)\, \bar p_{k,t}$$
  with the coarsening correction: when cells are large, evaluate models at the cell center with
  inflated noise (state why, per Thrun §8.2.2); likelihood pre-caching over the grid makes the
  correction O(lookup).
- *MCL* — the particle filter instantiated:
  sample $x_t^{[i]} \sim p(x_t \mid u_t, x_{t-1}^{[i]})$ via `sample_motion_model_odometry`;
  weight $w_t^{[i]} = p(z_t \mid x_t^{[i]}, m)$ via a Ch. 10 model; resample with the
  low-variance sampler (Ch. 8). Global initialization: $x_0^{[i]} \sim \mathrm{Uniform}(free(m))$.
- *Augmented MCL* — self-surprise detection:
  $$w_{avg} = \frac{1}{M}\sum_i w_t^{[i]},\qquad
    w_{fast} \leftarrow w_{fast} + \alpha_{fast}(w_{avg} - w_{fast}),\qquad
    w_{slow} \leftarrow w_{slow} + \alpha_{slow}(w_{avg} - w_{slow})$$
  During resampling, with probability $\max\{0,\ 1 - w_{fast}/w_{slow}\}$ draw a *random* pose
  (uniform in free space — or better, from the measurement model), else draw from
  $\mathcal{X}_t$ by weight. Sudden likelihood collapse → $w_{fast}$ dives below $w_{slow}$ →
  injection turns on exactly while the filter is surprised, and turns itself off after recovery.
- *Mixture (dual) proposal*: with probability $\phi$ propose from the measurement,
  $x_t^{[i]} \sim \eta\, p(z_t \mid x_t, m)$ (via Ch. 10's `sample_pose` for landmarks, or
  sampling the likelihood field), and weight by the *motion* side,
  $w_t^{[i]} = \int p(x_t^{[i]} \mid u_t, x_{t-1})\, bel(x_{t-1})\, dx_{t-1}$ (approximated by
  kernel-smoothed evaluation against $\mathcal{X}_{t-1}$); with probability $1-\phi$ do plain
  MCL. Cures peaked-likelihood degeneracy and accelerates recovery; costs variance and an extra
  density estimate — both measured in w12.4.
- *Dynamic environments* — measurement novelty filtering: reject beam $z_t^k$ when the posterior
  probability that it was caused by an unmodeled obstacle exceeds a threshold:
  $$p(\text{short} \mid z_t^k) \approx
    \frac{\sum_i z_{short}\, p_{short}(z_t^k \mid x_t^{[i]}, m)}
         {\sum_i p(z_t^k \mid x_t^{[i]}, m)} > \chi_{rej}$$
  (Table 8.4's test, phrased with the Ch. 10 mixture): people produce *shorter-than-expected*
  readings, so filtering only the short-dominated beams removes people without removing
  surprise-about-the-map (asymmetry stated and demonstrated in w12.5).
- *The comparison axes* (Table 8.5, now an experimental protocol): measurement type (landmarks /
  raw scans), noise model, posterior representation, memory, time per update, ease of
  implementation, resolution/accuracy, robustness to model violations, global localization,
  kidnapped recovery, dynamic environments.

**Derivations** (name — statement — sketch — collapsible):

1. **MCL correctness from importance sampling** — *propagating particles through the motion
   model and weighting by the measurement likelihood targets $bel(x_t)$ exactly as $M \to
   \infty$.* Sketch (4 steps): (i) proposal = motion-propagated prior $\overline{bel}$;
   (ii) target = $\eta\, p(z_t \mid x_t, m)\, \overline{bel}$; (iii) weight = target/proposal =
   likelihood; (iv) resampling returns unweighted posterior samples (Ch. 8 recap, one line
   each). Collapsible: the nonzero-support condition and why a *too-good* sensor breaks it (the
   bridge to mixture proposals); bias for finite $M$.
2. **Why plain MCL cannot recover from kidnapping** — *after convergence, particle diversity near
   the true (new) pose is zero, and no resampling scheme can resurrect unrepresented states.*
   Sketch (3 steps): survival probabilities; expected time to spontaneous coverage
   (astronomical); therefore recovery must come from *injection*, not reweighting. Collapsible:
   the related particle-deprivation-in-ambiguity failure (premature convergence to one of two
   symmetric modes) with the w12.1 symmetric-wing scenario as the exhibit.
3. **The $w_{fast}/w_{slow}$ detector** — *the ratio of short- to long-horizon average
   likelihood is a calibration-free divergence statistic.* Sketch (4 steps): $w_{avg}$ estimates
   $p(z_t \mid z_{1:t-1}, u_{1:t}, m)$; a well-localized filter holds it steady; kidnap drops it
   abruptly; exponential smoothing at two rates turns "abruptly" into a computable ratio, and
   injection proportional to $1 - w_{fast}/w_{slow}$ self-extinguishes. Collapsible: choosing
   $\alpha_{fast}, \alpha_{slow}$ (decades apart), interaction with sensor tempering (Ch. 10),
   and why injected poses should be drawn from the measurement model when available.
4. **Mixture-proposal weights** — *swapping proposal and target roles requires evaluating the
   other factor as the weight.* Sketch (3 steps): importance identity with
   $\pi = p(z\mid x)$-proportional proposal; weight $\propto \overline{bel}(x^{[i]})$; kernel
   approximation against the previous particle set with its cost/variance trade. Collapsible:
   normalization subtleties between the two particle streams, and the KLD interaction.
5. **Grid coarsening correction** — *evaluating point models at cell centers underestimates the
   probability of coarse cells; inflating model noise by the cell diameter compensates.*
   Sketch (3 steps): cell integral vs. point evaluation; the mismatch grows with resolution;
   noise inflation as a cheap integral approximation. Collapsible: the formal smoothing-kernel
   view and its effect on the accuracy column of the great table.

**Named algorithms** ($|G|$ grid cells, $M$ particles, $K$ beams):

| Algorithm | Signature | Complexity |
|---|---|---|
| `Grid_localization` | $(\{p_{k,t-1}\}, u_t, z_t, m) \to \{p_{k,t}\}$ | predict $O(|G|\cdot g_u)$ (bounded motion kernel $g_u$), correct $O(|G| \cdot K)$ or $O(|G|)$ pre-cached (Table 8.1) |
| `MCL` | $(\mathcal{X}_{t-1}, u_t, z_t, m) \to \mathcal{X}_t$ | $O(M \cdot K)$ (Table 8.2) |
| `Augmented_MCL` | $(\mathcal{X}_{t-1}, u_t, z_t, m) \to \mathcal{X}_t$ | $O(M \cdot K)$ + $O(M)$ bookkeeping (Table 8.3) |
| `KLD_MCL` | as MCL, adaptive $M_t$ | per Ch. 8; bins on a coarse grid |
| `test_range_measurement` | $(z_t^k, \bar{\mathcal{X}}_t, m) \to \{\text{keep}, \text{reject}\}$ | $O(M)$ per beam (Table 8.4) |
| `mixture_mcl` | adds $\phi$, dual stream | $O(M \cdot K + \phi M \cdot M_{prev})$ naive; $O(\phi M \log M)$ with a KD-tree (ours) |

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor: **the condensing cloud** — belief as weather. Particles are drawn in the
color of their role: freshly *predicted* particles **orange**, weights visualized as **green**
glow intensity immediately after sensing, the resampled posterior cloud **purple**, ground-truth
Rusty **gray dashed** outline. This is the same color story as Ch. 5's Hallway, now in 2D.

- **Widget w12.1: MCL Theater** *(flagship — the book's centerpiece and public demo; full-page
  wasm-sim)*. Designed as a stage with an orchestra pit:
  - **The stage (main panel, ~70% width)**: the Apartment floorplan. Up to 50,000 particles
    render as oriented specks (position + heading tick), colored by the predict→weight→resample
    cycle (orange → green-glow → purple, a ~300 ms choreography per filter step so the reader
    *sees* the Bayes filter breathe at 3 steps/s). Rusty's true pose is the gray-dashed ghost;
    the estimated pose (weighted mean of the dominant cluster) is a solid purple triangle with a
    thin trail. LiDAR beams flash from the *estimate*, so a delocalized filter visibly
    hallucinates scans through walls.
  - **The pit (bottom strip)**: three synced live charts — (a) ESS as a fraction of $M$;
    (b) $w_{fast}$ and $w_{slow}$ as two lines with the injection probability shaded red between
    them whenever $w_{fast} < w_{slow}$; (c) cluster count / belief entropy. Injection events
    spark as red dots rising from chart (b) onto the stage as newborn particles.
  - **The control rail (right, ~30%)**: 
    - Scenario buttons: **Global wake-up** (uniform init) · **Kidnap!** (teleports Rusty; big
      red button, the book's most famous control) · **Symmetric wing** (init in the mirrored
      two-bedroom wing — the cloud *must* stay bimodal until Rusty sees the asymmetric kitchen
      doorway; a banner celebrates "ambiguity honestly represented" while it lasts).
    - Sliders: $\log_{10} M$ (100 → 50,000; the renderer decimates above 10k but the filter is
      real), sensor model toggle (beam ↔ likelihood field, with the Ch. 10 cost chip),
      tempering $1/\kappa$, odometry $\alpha$ preset (good / sloppy / drunk), mixture rate
      $\phi$ (0 → 0.25), speed (pause / 1× / 4×) and a **time scrubber** over the last 60 s of
      history (every particle set is ring-buffered so the reader can rewind the collapse and
      replay it).
    - Algorithm switch: **MCL ↔ Augmented MCL ↔ KLD-Augmented** — switching mid-run keeps the
      particle set, so "kidnap under MCL, then switch to AMCL and watch it recover" is a
      two-click experiment.
    - Seed chip + re-roll die; a small "what am I looking at?" overlay for first-time visitors
      (this page is the book's public demo and must welcome cold traffic).
  - **Autoplay script** (no interaction, ~35 s loop): global wake-up with $M{=}5{,}000$ → drive
    a preset path → cloud condenses through the two-room ambiguity (camera gently zooms) →
    kidnap fires → $w_{fast}$ dives, red injection rain → re-convergence → caption card with the
    recovery time. The loop then invites: "Now break it yourself."
  - **Misconceptions killed**: *"the filter tracks a pose"* (no — it sculpts a distribution);
    *"more particles is always better"* (the $M$ slider vs. the wall-clock chip); *"resampling
    is where information enters"* (the choreography shows weighting is); *"kidnap recovery is
    automatic"* (plain MCL demonstrably never recovers).
  - **Static fallback**: a 6-keyframe filmstrip (wake-up / ambiguity / converged / kidnapped /
    injection / recovered) with the three pit-charts as a printed strip below.
- **Widget w12.2: Grid vs. Cloud** *(wasm-sim)*. Split screen on the same logged run: left, grid
  localization as a heatmap over the floorplan with a resolution slider (60 cm → 7.5 cm; the
  cell count and ms/update tick up alarmingly); right, MCL with an $M$ slider; a shared cost
  meter (memory + time per update) and shared error chart. Misconception killed: *"grids and
  particles are interchangeable"* — same math, wildly different economics, and the grid's
  approximation error is *structured* (quantized headings) while MCL's is stochastic.
- **Widget w12.3: Recovery Ward** *(wasm-sim)*. The $w_{fast}/w_{slow}$ mechanism in isolation:
  a 1D corridor version (Hallway world) where the likelihood traces are large and legible;
  $\alpha_{fast}, \alpha_{slow}$ sliders; kidnap button; a "false alarm" button that injects a
  one-off sensor glitch to show the slow filter's immunity to transients. Misconception killed:
  *"recovery needs a detector threshold tuned per map"* — the ratio is self-calibrating.
- **Widget w12.4: Proposal Mixer** *(wasm-sim)*. Landmark-world MCL with the $\phi$ slider 0 →
  1: at $\phi = 0$ a peaked sensor starves the filter (ESS crashes on each update); at
  $\phi = 1$ the cloud teleports noisily (motion information wasted, jittery estimate); the
  sweet spot is visible as an ESS/error valley plotted live over $\phi$. Misconception killed:
  *"the motion model is the only legitimate proposal."*
- **Widget w12.5: Crowd Mode** *(wasm-sim)*. The Apartment with 4 walking people (capsule
  colliders crossing Rusty's beams). Toggle `test_range_measurement`: off, the cloud gets
  dragged toward crowds and occasionally delocalizes; on, rejected beams render as dimmed gray
  stubs with a per-beam rejection-probability bar, and the estimate holds. A slider for
  $\chi_{rej}$ shows the trade (reject too eagerly and genuinely-surprising *map* evidence is
  ignored — demonstrated by also toggling a closed door). Misconception killed: *"a good sensor
  model already handles dynamic objects."*
- **Widget w12.6: The Great Table, Measured** *(interactive figure)*. Thrun's Table 8.5
  reproduced as a live results grid: rows = EKF (Ch. 11) / MHT / grid coarse / grid fine / MCL /
  AMCL; columns = the comparison axes. Every numeric cell was produced by the benchmark harness
  on identical logged runs (three logs: tracking, global, kidnap+crowd), and *clicking a cell
  replays the exact run* that produced it in a mini-viewer. Qualitative baseline claims are
  shown alongside our measured values, with disagreements flagged honestly.
- **Dashboard layout**: w12.1 owns a full-bleed page section early (play first); w12.2 sits in
  the grid section; w12.3/w12.4 flank the Augmented-MCL derivations; w12.5 in the dynamic
  section; w12.6 closes the chapter as its empirical summary. Shared chrome throughout: seed,
  pause, static fallback, one-parameter-featured sliders.

## 5. Practical (P) — Rust Implementation

**Crates**: `nalgebra` 0.35 (poses, cluster covariances); `rand` 0.9 + `rand_distr` 0.6 (seeded
`SmallRng` per filter — determinism is what makes the Theater's scrubber and the benchmark
possible); `rayon` (native parallel weighting over particles; the same code runs
single-threaded on WASM via a `cfg` shim, per book policy); `parry2d` 0.30 (walker collisions in
Crowd Mode); `sim`, `motion`, `sensor`, `localize` (Chs. 4/9/10/11 crates); `eframe` 0.35 +
`egui_plot` 0.34; `plotters` (filmstrip fallbacks); `kiddo` or in-tree KD-tree for the
mixture-proposal density estimate (small, WASM-clean — final pick recorded in CLAUDE.md).

**Module plan**: extends library `crates/localize/`; demo crate `demos/ch12-widgets/` whose
`mcl_theater` binary is *deployed as the book's public standalone demo* (own URL, linked from
the landing page).

```
crates/localize/src/
  grid/mod.rs      // GridLocalizer: discrete Bayes filter over PoseGrid
  grid/cache.rs    // per-cell likelihood pre-cache
  mcl/mod.rs       // Mcl<S: SensorModel>: Table 8.2 on Ch. 8's ParticleFilter
  mcl/augmented.rs // AugmentedMcl<S>: w_fast/w_slow + injection (Table 8.3)
  mcl/mixture.rs   // mixture proposal: dual stream + kernel weights
  mcl/dynamic.rs   // test_range_measurement (Table 8.4)
  mcl/cluster.rs   // mode extraction: density clustering for the estimate + entropy
  bench/harness.rs // the great-table experiment runner (feature = "bench")
```

```rust
use rand::rngs::SmallRng;
use pr_core::geom::se2::SE2;
use motion::{OdometryModel, OdomDelta};
use sensor::SensorModel;
use sim::{World, Scan}; // the known map is the Ch. 4 world (OccGrid arrives in Ch. 13)
use ch08_particles::{ParticleSet, low_variance_resample}; // Ch. 8 machinery

pub struct Mcl<S: SensorModel> {
    pub particles: ParticleSet<SE2>,       // {x^[i], w^[i]}, i = 1..M
    pub motion: OdometryModel,
    pub sensor: S,
    pub rng: SmallRng,                      // seeded: runs are replayable
}

impl<S: SensorModel> Mcl<S> {
    pub fn global_init(map: &World, m: usize, seed: u64, /* models */) -> Self;
    /// Table 8.2: sample-motion / weight / low-variance resample.
    pub fn step(&mut self, u: &OdomDelta, z: &Scan, map: &World);
    pub fn estimate(&self) -> (SE2, nalgebra::Matrix3<f64>, f64 /* cluster mass */);
}

pub struct AugmentedMcl<S: SensorModel> {
    pub mcl: Mcl<S>,
    pub w_fast: f64, pub w_slow: f64,
    pub a_fast: f64, pub a_slow: f64,       // defaults 0.1 / 0.001 (decades apart)
    pub injector: Injector,                 // UniformFree | FromSensor (Ch. 10 sample_pose)
    pub kld: Option<KldConfig>,             // adaptive M, from Ch. 8
}
impl<S: SensorModel> AugmentedMcl<S> {
    /// Table 8.3; returns injection count this step (the Theater's red rain).
    pub fn step(&mut self, u: &OdomDelta, z: &Scan, map: &World) -> usize;
}

/// Table 8.4: posterior probability the beam was caused by an unmodeled obstacle.
pub fn test_range_measurement<S: SensorModel>(
    beam: usize, z: &Scan, particles: &ParticleSet<SE2>, sensor: &S, map: &World,
    chi_rej: f64,
) -> bool;

pub struct GridLocalizer {
    pub bel: PoseGrid<f32>,                 // (nx, ny, ntheta) row-major
    pub cache: Option<LikelihoodCache>,
    pub res: GridRes,                       // xy meters, theta radians
}
```

All three localizers implement Ch. 11's `Localizer` trait — that single trait bound is what the
benchmark harness and the Theater's algorithm switch iterate over.

**Worked end-to-end example** (`cargo run --example great_table --features bench`): replays
three fixed-seed logs recorded in the Apartment — (a) 120 s tracking, (b) global wake-up, (c)
kidnap at $t{=}45\,$s with 4 walkers — through EKF, MHT, grid (60 cm and 15 cm), MCL
($M{=}5{,}000$), and AMCL (KLD, $M \in [500, 20{,}000]$). Prints the great table: RMSE after
convergence, convergence time (b), recovery rate and median recovery time over 50 kidnap seeds
(c), update ms (native + measured-on-WASM column), peak memory. Expected qualitative outcome
(asserted loosely by a regression test, exact numbers quoted in prose from a pinned run): EKF
wins tracking accuracy per CPU-ms; grid-fine matches MCL accuracy at ~30× memory; plain MCL
recovery rate ≈ 0%; AMCL ≈ 100% with median recovery under 20 s; novelty filtering roughly
halves crowd-induced error. Emits `great_table.md` + the w12.6 replay bundle.

**Runnable artifact**: the WASM Theater (`trunk build` of `ch12_widgets`) — the book's public
demo; natively, `great_table` regenerates every number in the chapter. The capstone (Ch. 26)
imports `AugmentedMcl` unchanged.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w12.1 | MCL Theater | wasm-sim (full-page; standalone public demo) | localize, motion, sensor, sim, parry2d, eframe, egui_plot | scenario buttons (global/kidnap/symmetric), M + α + κ + φ sliders, sensor & algorithm switches, time scrubber, seed | the Bayes filter breathing at scale; ambiguity, convergence, kidnap recovery; MCL as assembled parts |
| w12.2 | Grid vs. Cloud | wasm-sim | localize, sim, eframe, egui_plot | resolution + M sliders on a shared log; cost meter | representation economics; structured vs. stochastic error |
| w12.3 | Recovery Ward | wasm-sim | localize, sim (Hallway), eframe, egui_plot | α_fast/α_slow sliders; kidnap + false-alarm buttons | the w_fast/w_slow detector; self-extinguishing injection |
| w12.4 | Proposal Mixer | wasm-sim | localize, sensor, eframe, egui_plot | φ slider; ESS/error-vs-φ live curve | proposal/target duality; peaked-sensor starvation |
| w12.5 | Crowd Mode | wasm-sim | localize, sim, parry2d, eframe | filter toggle; χ_rej slider; door toggle | novelty filtering; the short-reading asymmetry |
| w12.6 | The Great Table, Measured | interactive figure (wasm) | localize (bench), eframe | click cells to replay runs | experimental comparison; honest engineering trade-offs |
| f12.7 | Theater filmstrip | static-svg | localize, plotters | — (build-time) | fallback for w12.1; print figure |

## 7. Exercises & Extensions

1. **(F)** For a two-room symmetric world, show that with exact symmetry the MCL posterior must
   remain bimodal for all $t$, and compute the expected time until finite-$M$ resampling noise
   spuriously kills one mode as a function of $M$ (the premature-convergence failure of w12.1's
   symmetric wing).
2. **(F)** Derive the steady-state injection probability of Augmented MCL under a permanent 20%
   likelihood drop (e.g., a remodeled room), as a function of $\alpha_{fast}/\alpha_{slow}$.
   What does a *persistent* nonzero injection rate tell you, and why is that a feature?
3. **(C, w12.1)** Predict-then-verify: with the likelihood field + $\kappa{=}1$ and
   $M{=}1{,}000$, will global wake-up converge before the first corridor junction? Check, then
   find the smallest $M$ (powers of two) that converges reliably across 5 re-rolls. Repeat with
   tempering $\kappa{=}3$; explain the direction of the change via Ch. 10 §overconfidence.
4. **(C, w12.3)** Set $\alpha_{fast} = \alpha_{slow}$. Predict what the detector does on kidnap
   and on the false-alarm glitch; verify; then state in one sentence why the *two* timescales
   are the mechanism.
5. **(P)** Implement `Injector::FromSensor` using the likelihood field (importance-sample free
   cells by field value at scan endpoints) and measure the change in median kidnap-recovery
   time in the `great_table` harness versus uniform injection.
6. **(P, harder)** Add a `SLAM-Toolbox-style` sanity column to the great table: implement a
   scan-to-map gradient refinement (5 iterations of hill climbing on the likelihood field) of
   the AMCL estimate each step, and report its accuracy/cost row. This is the bridge exercise
   to Ch. 16's scan matching.

## 8. Modernization Notes

- **Added vs. baseline:** KLD-adaptive particle counts integrated into `AugmentedMcl` (2005-
  edition material absent from the 1999 draft; built in our Ch. 8); measurement-model injection
  for recovery (the baseline mentions uniform injection; sensor-driven injection is what
  production AMCL implementations do); sensor tempering and beam subsampling as first-class
  Theater controls (Ch. 10 carry-through); determinism discipline (seeded, replayable runs —
  which makes the time scrubber and the reproduced table possible at all); the explicit 2026
  status report: **AMCL, this chapter's algorithm, still ships as Nav2's default localizer**,
  with SLAM-Toolbox's pose-graph localization mode presented as the modern alternative and a
  pointer to Ch. 16; WASM-measured performance numbers alongside native ones.
- **Kept:** Tables 8.1–8.4 essentially verbatim as the algorithmic spine; the dynamic-
  environment novelty filter (people-filtering is *more* relevant in 2026 service-robot
  deployments, not less); the great comparison table — upgraded from asserted to measured, with
  baseline claims and our measurements shown side by side and discrepancies discussed.
- **Dropped/condensed:** the baseline's topological/coarse-grid localization discussion is
  compressed into the w12.2 resolution slider plus two paragraphs (fine metric grids won);
  likelihood pre-caching gets a module and a sentence, not pages (it is an implementation
  detail once the distance-transform field exists); the baseline's extended real-robot museum
  anecdotes are replaced by the reproducible Apartment logs — the narrative loss is real and
  acknowledged, the scientific gain (every number regenerable by `cargo run`) is the book's
  thesis.
