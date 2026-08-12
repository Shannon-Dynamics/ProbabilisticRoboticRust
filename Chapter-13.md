# Chapter 13 — Occupancy Grid Mapping

> Part V — Mapping and SLAM · Estimated length: 8 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Part IV ended with Rusty localizing beautifully — against a map we handed it. This chapter flips the
problem: the poses are known, the map is not. The reader's "aha" is double. First: *a map is not a
drawing, it is a field of beliefs* — one tiny binary Bayes filter (Ch. 8, §binary static-state) per
cell, run in parallel a hundred thousand times, with gray meaning "I honestly don't know." Second,
the darker aha: that parallelism is a *lie* — cells are coupled through every beam that crosses
them, and the chapter ends by showing exactly where the lie bites (doorways, thin obstacles) and
what honest inference (MAP with forward models) looks like. The closing sentence plants Ch. 14's
hook: the *real* fiction in this chapter was "poses are known."

Story line:
1. **Hook** — replay a Ch. 12 log; strip away the map; ask: could Rusty have drawn it itself?
   Naive "stamp what the sensor sees" mapping smears and contradicts itself within seconds.
2. **Play** — w13.1 Map Weaver autoplays: gray fog resolving into walls as evidence accumulates.
3. **Intuition** — each cell is the Ch. 8 binary log-odds filter; a scan is a paintbrush whose
   bristles are the inverse sensor model.
4. **Formalism** — map posterior, the per-cell factorization (named explicitly as Approximation #1),
   the log-odds recursion derived, the hand-crafted inverse model constructed.
5. **Second thoughts** — where does the inverse model come from? Learn it from the forward model.
   What if sensors disagree? Multi-sensor fusion and its failure mode.
6. **The trap** — w13.2: conflicting evidence in doorways; the joint posterior does not factor;
   MAP mapping with forward models repairs it (at a price).
7. **Implementation & lab** — `OccGrid` in Rust, Bresenham ray updates, 60 fps map streaming;
   integration lab maps the Apartment from a fixed-seed log.
8. **Bridge** — the entropy of the map (currency for Ch. 24) and the confession about known poses
   (setup for Ch. 14).

## 2. Prerequisites & Position

- **Builds on:** Ch. 4 (the `sim` crate, ray-cast LiDAR via `parry2d`, the Apartment world); Ch. 8
  (binary Bayes filter with static state, log odds $\ell$ — this chapter is its mass deployment);
  Ch. 10 (the forward beam model `beam_range_finder_model`, reused verbatim for learned inverse
  models and MAP mapping); Ch. 12 (supplies the "known poses" via localization on logs).
- **Feeds into:** Ch. 14 (drops the known-pose assumption → SLAM); Ch. 16 (occupancy submaps in
  RustSLAM-2D); Ch. 17 (per-particle grids in grid-based RBPF); Ch. 19 (octrees/TSDF as successor
  representations); Ch. 24 (map entropy and frontiers drive exploration); Ch. 25 (the learned
  inverse-model idea modernized with `candle`).
- **Baseline sources:** Thrun et al. (1999–2000 draft) Ch. 9 §9.1–9.5 — Tables 9.1
  (`occupancy_grid_mapping`), 9.2 (`inverse_range_sensor_model`), 9.3
  (`MAP_occupancy_grid_mapping`); Ch. 4 §4.1.4 (binary Bayes filter, Table 4.2); Ch. 6 §6.3 (beam
  model as the forward model). Modernization set: Nav2/SLAM-Toolbox costmap practice (log-odds
  clamping, dynamic environments); OctoMap lineage (pointer only, → Ch. 19).

## 3. Foundation (F) — Mathematical Core

### 3.1 Notation introduced (chapter-scoped table)

| Symbol | Meaning |
|---|---|
| $m = \{m_i\}$ | grid map as a set of cells; $m_i \in \{0,1\}$ binary occupancy of cell $i$ |
| $p(m_i)$ | occupancy prior of a cell (default $0.5$) |
| $\ell_{t,i}$ | log odds of cell $i$ at time $t$ (TOC notation table) |
| $\ell_0$ | prior log odds $\log \frac{p(m_i)}{1-p(m_i)}$ |
| $\ell_{occ}, \ell_{free}$ | evidence increments of the inverse model |
| $p(m_i \mid z_t, x_t)$ | inverse sensor model (cell posterior from one measurement) |
| $\alpha, \beta$ | obstacle depth and beam opening angle of the hand-crafted inverse model |
| $H(m)$ | map entropy $\sum_i H_b(p_i)$, the "how much is left to learn" scalar |

Color code in this chapter: beams and evidence overlays are **green** (measurement); the prior
$\ell_0$ line is **blue**; the map itself uses the field-standard grayscale (white = free, black =
occupied, mid-gray = ignorance) — the text explicitly notes this is the one place the book's
palette yields to a stronger universal convention, and that "gray = ignorance" is the point.

### 3.2 Definitions

- **Occupancy grid map**: a tessellation of the plane into cells $m_i$, each a static binary
  random variable. The full map posterior is $p(m \mid z_{1:t}, x_{1:t})$ — a distribution over
  $2^{|m|}$ maps, hopelessly intractable to represent directly.
- **Approximation #1 (per-cell independence)**:
  $p(m \mid z_{1:t}, x_{1:t}) \approx \prod_i p(m_i \mid z_{1:t}, x_{1:t})$. Named, numbered, and
  put on trial in §3.3-D4. Every occupancy mapper in production makes it; almost none admit it.
- **Log odds / recovery**: $\ell_{t,i} = \log \frac{p(m_i \mid z_{1:t}, x_{1:t})}{1 - p(m_i \mid z_{1:t}, x_{1:t})}$,
  with recovery $p(m_i \mid z_{1:t}, x_{1:t}) = 1 - \frac{1}{1 + \exp \ell_{t,i}}$.
- **Inverse vs. forward model**: forward $p(z_t \mid x_t, m)$ (Ch. 10) generates measurements from
  maps; inverse $p(m_i \mid z_t, x_t)$ assigns cell posteriors from one measurement. The inverse
  direction is the unnatural one — which is why §3.3-D3 learns it *from* the forward one.
- **Perceptual field**: the set of cells a measurement $z_t$ says anything about (the scan cone);
  cells outside it keep $\ell$ unchanged.

### 3.3 Key derivations

**D1 — The static-state log-odds filter.**
*Statement:* under Approximation #1 and static $m_i$, the per-cell posterior obeys
$$\ell_{t,i} = \ell_{t-1,i} + \log \frac{p(m_i \mid z_t, x_t)}{1 - p(m_i \mid z_t, x_t)} - \ell_0 .$$
*Sketch (5 steps):* (1) Bayes rule on $p(m_i \mid z_{1:t}, x_{1:t})$ splitting off $z_t$;
(2) apply Bayes again to $p(z_t \mid m_i, z_{1:t-1}, x_{1:t})$ to swap in the inverse model —
static state makes $z_t$ depend only on $m_i$ and $x_t$; (3) write the same expression for the
complement $\neg m_i$ and take the ratio — every normalizer $\eta$ cancels; (4) take logs: the
product telescopes into an additive recursion; (5) identify the $-\ell_0$ term as the correction
preventing the prior from being counted once per measurement. *Collapsible:* the full odds-ratio
algebra for both branches, and the boundary condition $\ell_{0,i} = \ell_0$.

**D2 — Constructing the hand-crafted inverse range model.**
*Statement:* for a beam with reading $z_t^k$, cells at range $r$ and bearing offset within
$\beta/2$ of the beam get: $\ell_0$ (no information) if $r > \min(z_{max}, z_t^k + \alpha/2)$;
$\ell_{occ}$ if $z_t^k < z_{max}$ and $|r - z_t^k| < \alpha/2$; $\ell_{free}$ if $r \le z_t^k$.
*Sketch (3 steps):* geometry of the cone; the three regions (free along the ray, occupied in an
$\alpha$-thick shell at the reading, unknown beyond); why $\alpha$ ≈ obstacle depth + discretization
and $\beta$ ≈ beam divergence. Not a derivation so much as an honest *design*, which sets up D3:
nothing about these constants is canonical. *Collapsible:* discretization error analysis (cell
centers vs. cell coverage) and the LiDAR specialization $\beta \to$ one ray.

**D3 — Learned inverse models are Bayes-optimal regression (Thrun §9.3).**
*Statement:* the minimizer of expected cross-entropy loss, over triplets $(m, x_t, z_t)$ sampled
from the map prior and forward model, among all functions $f(z_t, x_t, i)$, is exactly
$f^*(z_t, x_t, i) = p(m_i \mid z_t, x_t)$.
*Sketch (5 steps):* (1) sample maps from the prior (randomized Apartment variants in our sim);
(2) sample poses, then $z_t \sim p(z_t \mid x_t, m)$ via the Ch. 10 beam model — this is the
"sampling from the forward model" step; (3) label each nearby cell with its ground-truth $m_i$;
(4) write the expected log-loss and condition on $(z_t, x_t, i)$; pointwise minimization gives the
conditional probability; (5) conclude: a function approximator trained this way *converges to the
true inverse model*, automatically consistent with sensor physics and map prior. *Collapsible:*
the pointwise variational argument, and feature engineering (relative cell coordinates in beam
frame, reading $z_t^k$, incidence angle).

**D4 — The independence trap, and MAP mapping with forward models (Thrun §9.4).**
*Statement:* measurements couple cells ("explaining away"), so the true posterior does not factor;
the factored filter double-counts and manufactures conflict. The honest alternative maximizes the
un-factored posterior: $m^* = \arg\max_m \left[ \log p(z_{1:t} \mid x_{1:t}, m) + \log p(m) \right]$.
*Sketch (5 steps):* (1) two-cell, one-beam worked example: a beam that ends at cell B after
crossing cell A supports "A free AND B occupied" — the joint has mass on configurations the
product form cannot represent; (2) doorway scenario: beams from two sides give contradictory
per-cell evidence that the joint explains consistently; (3) write the log-posterior with the
Ch. 10 forward model per beam; (4) hill climbing: flip the cell that most increases the
log-posterior, using precomputed beam–cell incidence so a flip re-evaluates only intersecting
beams; (5) name the costs: point estimate (no uncertainty), batch, local maxima. *Collapsible:*
the full two-cell joint posterior table with numbers, and the flip-gain bookkeeping.

**D5 — Multi-sensor fusion (Thrun §9.2.1), short.**
*Statement:* adding log odds across sensors is correct only if both sensors detect the *same*
notion of "occupied." Sonar (tabletop height) and LiDAR (shin height) don't — LiDAR's confident
"free" erases sonar's "table." Remedy: one grid per modality, combined pessimistically,
$p(m_i) = \max_k p(m_i^{[k]})$. Two-step sketch; no collapsible.

### 3.4 Named algorithms

| Algorithm (Thrun table) | Signature | Complexity |
|---|---|---|
| `occupancy_grid_mapping` (T9.1) | $(\{\ell_{t-1,i}\}, x_t, z_t) \to \{\ell_{t,i}\}$ | $O(B \cdot L/\rho)$ per scan with ray traversal ($B$ beams, range $L$, resolution $\rho$) — not $O(\lvert m \rvert)$ |
| `inverse_range_sensor_model` (T9.2) | $(i, x_t, z_t) \to \ell$ | $O(1)$ per cell given nearest-beam lookup |
| `learn_inverse_sensor_model` (§9.3, no table) | $(p(z\mid x,m), \text{map prior}, \text{sim}) \to \hat f$ | offline; $O(\text{samples} \times \text{epochs})$ |
| `MAP_occupancy_grid_mapping` (T9.3) | $(x_{1:t}, z_{1:t}) \to m^*$ | $O(\text{sweeps} \times \lvert m \rvert \times \bar{B}_i)$, $\bar{B}_i$ = beams incident per cell |

Numeric micro-example (kalmanfilter.net-style, reproduced by a unit test): $\ell_{occ} = \ln(0.7/0.3)
\approx 0.8473$, $\ell_{free} = -0.8473$, $\ell_0 = 0$. One cell observed occupied, occupied, free:
$\ell = 0.8473 \to 1.6946 \to 0.8473$, i.e. $p = 0.700 \to 0.845 \to 0.700$. Three lines of
arithmetic the reader can check by hand and against the Rust test.

## 4. Conceptual (C) — Intuition & Visual Design

One metaphor carried end-to-end: **the scan as a paintbrush, the map as slow-developing film** —
every stroke deposits a little evidence; gray is undeveloped film, not "empty."

- **Widget w13.1: Map Weaver** *(flagship, TOC name)* — type: wasm-sim (interactive).
  Autoplay: Rusty replays a fixed-seed exploration of the Apartment; the map develops from uniform
  gray, walls crisping as they are re-observed. Manipulates: arrow keys to take over driving;
  **one headline parameter** — an "evidence strength" slider scaling $|\ell_{occ}|, |\ell_{free}|$;
  seed re-roll; hover any cell for a **cell inspector** (that cell's private $\ell$ time series —
  literally its Ch. 8 filter, plotted with the blue prior line and green evidence ticks).
  Observes: single scans claim little; repetition converges; unexplored stays gray.
  Misconception killed: "a robot either has a map or it doesn't" / "mapping = stamping the sensor
  footprint" — mapping is per-cell evidence accumulation with an explicit ignorance state.
  Static fallback: three development stills (t = 5 s, 30 s, full) + final map.
- **Widget w13.2: Independence Trap** *(flagship, TOC name)* — type: wasm-sim.
  Preset scene: a doorway/thin wall observed from both sides (Thrun's conflict case). Left pane:
  standard log-odds map — the doorway cells flicker and wash toward gray as passes disagree.
  Right pane: MAP with forward models; button **"step hill climb"** flips cells one at a time,
  each flip annotated with the per-beam log-likelihood bars it improved. Manipulates: scene
  selector (doorway / outside corner / thin pole), step/play optimization. Observes: the factored
  filter *manufactures* contradiction; the joint explanation settles it. Misconception killed:
  "per-cell independence is harmless bookkeeping." Static fallback: before/after pair with the
  conflicting-beam diagram.
- **Widget w13.3: Inverse-Model Workbench** — type: wasm-sim (supporting).
  A single frozen beam; the model's response painted along/around it as the "paintbrush profile"
  (green evidence overlay on grayscale). Sliders $\alpha, \beta$; tab switches hand-crafted ↔
  learned model (the learned profile shows soft edges, range-dependent widening, and near-$z_{max}$
  skepticism the hand-crafted one lacks). Misconception killed: "the inverse model is canonical" —
  it is a design choice, and the learned one disagrees exactly where intuition is weakest.
- **Widget w13.4: Fusion Fumble** — type: wasm-sim (supporting, small).
  Sonar ring + LiDAR map a room containing a table. Toggle "sum log odds" vs "per-sensor maps +
  max". Observes: addition erases the table; max keeps it. Misconception killed: "more sensors +
  more Bayes = automatically better map."
- **Figure s13.5:** static-svg, build-time `plotters`: the log-odds ↔ probability curve with
  saturation/clamping annotations ($\pm\ell_{max}$) and the $\ell_0$ reference line in blue.

Dashboard sketch: the chapter's **integration lab** is Map Weaver full-width with two additions —
an inverse-model dropdown (hand-crafted / learned) and a live map-entropy sparkline $H(m)$ under
the map (falling entropy = learning; teaser caption points to Ch. 24 where this curve becomes the
objective). All widgets: autoplay defaults, seeded RNG with visible re-roll, static PNG fallback
rendered from the same Rust code at build time.

## 5. Practical (P) — Rust Implementation

Crates (per TOC verified stack):
- `nalgebra` 0.35 — poses, small vectors (`Vector2<f32>`), no heap in the hot loop.
- `rand` 0.9 + `rand_distr` 0.6 — seeded `SmallRng` everywhere (WASM-clean, reproducible).
- `parry2d` 0.30 — ray-cast LiDAR inside `crates/sim` (built in Ch. 4, reused).
- `egui`/`eframe` 0.35 + `egui_plot` 0.34 — Map Weaver widget; the map streams as an
  `egui::ColorImage` texture updated per frame (60 fps target; only dirty cells re-blitted).
- `plotters` (SVG backend) — build-time static figures and fallbacks.
- `serde` — replay logs and trained-model weights.
- Grid storage is a plain `Vec<f32>` (row-major, index math shown explicitly — a teaching point);
  `ndarray` 0.17 is named as the alternative and deliberately not used.
- The learned model is a **hand-rolled logistic regressor / 2-layer MLP on nalgebra** trained by
  SGD — small enough to print; `candle` is deliberately deferred to Ch. 25.

Module plan:

```text
crates/ch13_occgrid/
  src/lib.rs         — OccGrid, GridIdx, world↔grid transforms, entropy()
  src/inverse.rs     — InverseSensorModel trait; HandCraftedModel; LearnedModel
  src/bresenham.rs   — integer ray traversal (supercover variant, so no corner-skipping)
  src/learn.rs       — forward-model sampling + SGD training (D3)
  src/map_map.rs     — MAP_occupancy_grid_mapping (D4) with beam–cell incidence cache
  examples/weave_map.rs          examples/independence_trap.rs
  tests/micro_example.rs         tests/determinism.rs
demos/ch13-map-weaver/     — eframe/trunk crate: w13.1, w13.3, w13.4
demos/ch13-independence/   — eframe/trunk crate: w13.2
```

Key types & signatures (compiles-in-spirit):

```rust
use nalgebra::Vector2;
use pr_core::geom::SE2;   // Ch. 3 hand-rolled SE(2)
use sim::Scan;            // Ch. 4's scan type

/// One (range, bearing) ray of a `sim::Scan` — the inverse model's unit of evidence.
pub struct Beam { pub range: f32, pub bearing: f32 }

pub struct OccGrid {
    width: usize, height: usize,
    pub resolution: f32,                 // meters per cell
    pub origin: Vector2<f32>,
    log_odds: Vec<f32>,                  // ℓ = 0.0  ⇔  p = 0.5 (gray)
    pub l_clamp: f32,                    // ±ℓ_max saturation (Nav2 practice)
}

impl OccGrid {
    pub fn probability(&self, c: GridIdx) -> f32;         // 1 − 1/(1+exp ℓ)
    pub fn entropy(&self) -> f32;                         // Σ H_b(p_i), feeds Ch. 24
    /// Table 9.1 restricted to the perceptual field via Bresenham traversal.
    pub fn integrate_scan<M: InverseSensorModel>(
        &mut self, pose: &SE2, scan: &Scan, model: &M);
}

pub trait InverseSensorModel {
    /// log-odds evidence (already minus ℓ0) for one cell under one beam
    fn evidence(&self, cell_center: Vector2<f32>, pose: &SE2, beam: &Beam) -> f32;
}
pub struct HandCraftedModel { pub alpha: f32, pub beta: f32,
                              pub l_occ: f32, pub l_free: f32 }   // Table 9.2
pub struct LearnedModel { w1: nalgebra::DMatrix<f32>, w2: nalgebra::DVector<f32> }

pub fn bresenham(a: GridIdx, b: GridIdx) -> impl Iterator<Item = GridIdx>;

/// Table 9.3 — batch MAP with the Ch. 10 forward beam model.
pub fn map_occupancy_grid_mapping(
    poses: &[SE2], scans: &[Scan],
    fwd: &sensor::BeamModel, init: &OccGrid, sweeps: usize) -> OccGrid;
```

Worked end-to-end example — `cargo run --example weave_map`: fixed seed `0xC0FFEE`, ground-truth
poses from the Ch. 4 replay log `apartment_loop.log` (720 scans). Output: final map SVG/PNG, map
entropy vs. $t$ curve, and a printed table of probe-cell probabilities. Expected: wall probe cells
$> 0.95$, corridor cells $< 0.05$, never-seen closet cells $= 0.5$ exactly; entropy monotonically
non-increasing except during the (optional) moving-person segment. `tests/micro_example.rs` asserts
the §3.4 micro-example ($0.700, 0.845, 0.700$ to $10^{-6}$); `tests/determinism.rs` asserts the
whole map is bit-identical across runs and platforms (the book's seeded-reproducibility contract).

Runnable artifact: the WASM demo *is* Map Weaver — the same `OccGrid::integrate_scan` compiled to
WASM drives the widget; the native example writes the chapter's printed figures.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w13.1 | Map Weaver | wasm-sim | ch13_occgrid + sim + eframe 0.35 | drive/autoplay, evidence-strength slider, cell inspector, seed re-roll | mapping = per-cell evidence accumulation; gray = ignorance |
| w13.2 | Independence Trap | wasm-sim | ch13_occgrid (map_map) + sensor + eframe | scene selector, step/play hill climb | the factorization lie; MAP with forward models as repair |
| w13.3 | Inverse-Model Workbench | wasm-sim | ch13_occgrid + eframe | α, β sliders; hand-crafted↔learned tab | inverse model is a design choice; learned ≠ hand-crafted |
| w13.4 | Fusion Fumble | wasm-sim | ch13_occgrid + sim + eframe | fusion-rule toggle | when log-odds addition across sensors is wrong; max-fusion |
| s13.5 | Log-odds curve | static-svg | plotters | — | ℓ ↔ p mapping, clamping, prior line |

## 7. Exercises & Extensions

1. **(F)** Derive the log-odds recursion (D1) from Bayes rule; then redo it with prior
   $p(m_i) = 0.2$ and show precisely where $\ell_0$ enters and why omitting the $-\ell_0$ term
   counts the prior once per measurement.
2. **(F)** Two cells, one beam: compute the exact joint posterior table for the D4 mini-example and
   exhibit a query (e.g. $p(m_A = 1 \land m_B = 1)$) that the factored representation gets wrong.
3. **(C)** *Predict-then-verify with w13.2:* before pressing "step hill climb," write down which
   doorway cells will flip first and why; verify, and explain the order using the per-beam
   likelihood bars.
4. **(C)** *Predict-then-verify with w13.1:* set $|\ell_{occ}| \gg |\ell_{free}|$; predict the two
   artifacts (thickened walls; ghosts of the moving-person preset that never fade); verify, then
   find the clamp value $\ell_{max}$ that bounds ghost-decay time below 5 s.
5. **(P)** Implement `max`-fusion for the sonar+LiDAR preset of w13.4 and add a regression test
   showing the table survives; benchmark `integrate_scan` with Bresenham vs. a naive
   every-cell-in-cone update and report the speedup on the Apartment map.
6. **(P, stretch)** Swap the `LearnedModel` into Map Weaver; compare against `HandCraftedModel`
   cell-wise with the Brier score over the ground-truth Apartment; report where the learned model
   wins (grazing beams, near-$z_{max}$ readings) and why, per D3.

## 8. Modernization Notes

- **Baseline:** the 1999–2000 draft Ch. 9 is essentially the published 2005 Ch. 9; the algorithmic
  core (Tables 9.1–9.3) is intact here and remains the industry's 2D substrate (Nav2 costmaps,
  SLAM Toolbox submaps) — this chapter is mostly *faithful*, not replaced.
- **Added beyond baseline:** log-odds clamping and semi-dynamic environments (ghost decay) from
  Nav2-era practice; map entropy as a first-class output (the Ch. 24 exploration currency); LiDAR
  as the default sensor with sonar demoted to the historical hard case that motivated wide-$\beta$
  cones; the Bayes-optimal-regression framing of learned inverse models (D3) stated as a theorem
  rather than folklore; determinism/reproducibility discipline.
- **Condensed:** the draft's 1990s neural-network machinery for §9.3 (feature engineering, MLP
  training minutiae) becomes one derivation + a printable logistic model; the honest modern
  version with `candle` lives in Ch. 25 by design. Multi-sensor fusion kept to one section.
- **Kept with honest framing:** MAP mapping with forward models is taught for its *lesson* about
  dependencies, not as production practice — modern systems resolve most "conflicting evidence"
  by fixing the poses (Ch. 14–16) and fusing submaps (Ch. 16/19), because pose error, not
  cell coupling, dominates real-world artifacts. The chapter says this out loud.
- **Deferred:** 3D representations (OctoMap, TSDF/ESDF) to Ch. 19; learned models done properly to
  Ch. 25; occupancy grids under *unknown* poses to Ch. 14/16/17.
