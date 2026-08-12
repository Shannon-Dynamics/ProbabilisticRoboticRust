# Chapter 24 — Exploration and Active SLAM

> Part VI — Planning and Acting under Uncertainty · Estimated length: 9 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Every chapter so far handed Rusty a goal. This one asks the question autonomy actually starts with:
where should Rusty go *to learn*? Exploration closes the book's last conceptual loop — the belief
machinery of Parts II–V becomes not just the input to decisions (Ch. 22) but the *objective* of
them: act to make your own belief sharp. The chapter builds the modern decision stack in three
rings: (1) exploration of the *map* — frontiers, expected information gain, entropy curves;
(2) active localization — choosing motions that disambiguate the *pose*; (3) active SLAM — utility
over the Ch. 15 graph, where coverage and loop-closing compete and the information matrix $\Omega$
becomes a decision variable. Framed honestly: active SLAM *is* a POMDP (Ch. 22), intractable
exactly, and the field's working answer is the identify–select–execute pipeline of Placed et al.
2023 — the survey the chapter uses as its map of the field. The "aha": information gain is not a
heuristic bolted onto navigation; it is expected entropy reduction, computable from the same
log-odds cells and the same graph $\Omega$ the reader has been maintaining since Chs. 13 and 15.
The chapter ends with the book's pre-capstone: Rusty, dropped into a floorplan it has never seen,
maps it autonomously — every subsystem the reader built, deciding for itself.

Story line:
1. **Hook:** the "lawnmower fallacy" — a scripted boustrophedon sweep of an unknown apartment
   (autoplay) wastes half its time re-scanning known space and drifts unrecoverably; overlay: the
   entropy curve barely falls while odometry error climbs.
2. **Play (C):** Frontier Chaser — watch utilities scored, targets chosen, entropy fall.
3. **Formalize (F):** map entropy from log odds; mutual information; expected gain of a sensing
   pose; frontiers and their completeness.
4. **Active localization (F/C):** expected entropy reduction over *pose* beliefs; the
   Disambiguation Detour.
5. **Active SLAM (F):** utility over the pose graph — D-optimality via $\log\det\Omega$; the
   loop-closing vs. coverage tension; stopping criteria.
6. **Practical (P):** frontier detector + information-gain scorer over `OccGrid`; greedy explorer
   wired to the Ch. 16 SLAM stack, Ch. 20 planner, Ch. 23 controller.
7. **Integration lab:** the autonomous mapping run, with ablations (nearest-frontier vs. info-rate
   vs. +loop-closure utility).
8. **Bridge out:** what remains for Ch. 26 is only *orchestration* — every decision, estimate, and
   control now exists.

## 2. Prerequisites & Position

- **Builds on:** Ch. 2 (entropy, mutual information), Ch. 8 (log-odds binary filter — the entropy
  substrate), Ch. 12 (MCL; the bimodal beliefs active localization disambiguates), Ch. 13
  (`OccGrid`; per-cell independence, again), Ch. 15 (graph $\Omega$; sparse Cholesky via `faer`),
  Ch. 16 (RustSLAM-2D, the stack being driven), Chs. 20–23 (goal planning and execution; Ch. 22's
  POMDP frame and its AMDP entropy caveat).
- **Feeds into:** Ch. 26 (the capstone is this chapter's lab plus failure-mode orchestration),
  Ch. 25 (learned exploration policies pointer).
- **Baseline sources:** the 1999–2000 draft PDF has **no exploration chapter**; the published 2005
  edition's Ch. 17 (greedy entropy-based exploration, active localization, exploration with
  occupancy grids / multi-robot coordination) is the conceptual ancestor, rebuilt here from the
  modernization set: Placed et al., "A Survey on Active SLAM," IEEE T-RO 39(3) 2023 (unified
  formulation, identify–select–execute, optimality criteria, open problems — the chapter's
  skeleton); Yamauchi 1997 (frontier exploration); Burgard/Fox/Thrun active-localization lineage;
  Stachniss (RBPF exploration, historical note); Kaess/Dellaert information-theoretic graph
  utilities; Khosoussi et al. (tree-connectivity surrogates, mention).

## 3. Foundation (F) — Mathematical Core

**Notation introduced:**

| Symbol | Meaning |
|---|---|
| $H(b) = -\sum_x b(x) \log b(x)$ | entropy of a belief (pose belief: over MCL particles/grid) |
| $p_i = 1 - \frac{1}{1 + e^{\ell_i}}$ | cell occupancy probability from log odds $\ell_i$ (Ch. 8/13) |
| $H(m) = \sum_i H_b(p_i)$ | map entropy; $H_b$ the binary entropy function |
| $a$, $\mathcal{A}$ | candidate action (sensing pose / trajectory), candidate set |
| $I(a) = H(b) - E_{z_a}\!\left[ H(b \mid z_a) \right]$ | expected information gain = mutual information $I(m; z_a)$ |
| $U(a) = w_I\, I(a) - w_C\, C(a)$ | utility: gain vs. cost (path length/time); weights $w_I, w_C$ |
| $\Omega$ | pose-graph information matrix (Ch. 15's, now a decision object) |
| $\mathrm{Dopt}(\Omega) = \exp\!\big(\tfrac{1}{n}\log\det\Omega\big)$ | D-optimality criterion (Placed et al.'s modern form) |

Weights are $w_\bullet$, never $\alpha/\beta$ — those symbols are spoken for (Ch. 9 noise,
Ch. 22 vectors); a margin note says so.

**Definitions:** frontier (free cell adjacent to unknown), frontier region (connected component of
frontier cells, with centroid and size); candidate action; information gain; utility; A-/D-/E-
optimality of an information matrix (one definition box, eigenvalue view: mean / product / min);
stopping criterion; the identify–select–execute pipeline (definition of each stage, Placed et al.).

**Key derivations:**

1. **Sensing never hurts (in expectation).** *Statement:* $I(a) = I(m; z_a) \ge 0$ — expected
   posterior entropy never exceeds prior entropy; individual measurements can still *raise*
   entropy. *Sketch (4 steps):* write $I(a)$ as mutual information; express as
   $KL(p(m, z_a) \,\|\, p(m)p(z_a))$; KL non-negativity; a two-cell counterexample where one
   surprising $z$ raises $H$ (the widget shows real instances flagged live). *Collapsible:* the
   Jensen/KL algebra and the distinction between expected and realized gain — the reader's
   inoculation against "the entropy went up, the code is broken" bug reports.
2. **Beam-wise expected gain over an occupancy grid.** *Statement:* under the Ch. 13 per-cell
   independence approximation, the expected gain of a sensing pose decomposes over beams and cells:
   $I(a) \approx \sum_{\text{beams}} \sum_{i \in ray} P(\text{beam reaches } i)\, \big[H_b(p_i) -
   E_z[H_b(p_i \mid z)]\big]$, computable by ray-casting through the *current* map with the Ch. 10
   forward model. *Sketch (5 steps):* factor map entropy over cells; condition beam traversal on
   the occupancy of earlier cells along the ray (reach probability as a running product); per-cell
   binary-filter update for hit/pass outcomes; sum; note the systematic bias of assuming unknown
   cells behave like their prior. *Collapsible:* the full forward-simulation recursion, plus the
   cheap surrogate everyone uses (count unknown cells within sensor range, weighted by reach
   probability) and a measured plot of surrogate-vs-exact correlation from the crate.
3. **Completeness of frontier exploration.** *Statement:* with an ideal sensor and a planner
   complete on the current known-free space, repeatedly navigating to *any* reachable frontier
   until none remain maps every cell reachable from the start. *Sketch (4 steps):* unknown
   reachable space is separated from known-free space by frontier cells (connectivity argument);
   visiting a frontier strictly grows known space (sensor sees past it); monotone bounded growth
   terminates; termination ⇒ no reachable frontier ⇒ reachable closure mapped. *Collapsible:* the
   topological argument done carefully, and the real-world failure modes (range limits at glass,
   inflation swallowing narrow doorways — each demonstrated in the lab world).
4. **Active localization as expected entropy minimization.** *Statement:* choose
   $a^* = \arg\max_a\, [H(bel) - E_{z_a}[H(bel')]]$ over candidate motions, with $bel$ the MCL
   belief and $bel'$ its Bayes-filter update — the Burgard/Fox/Thrun greedy rule. *Sketch (4
   steps):* for each candidate and each particle hypothesis, forward-simulate motion + measurement
   (Ch. 4's simulator as generative model, again); average posterior entropies over simulated $z$;
   subtract cost; note this is a depth-1 POMDP backup — the honest connection to Ch. 22, with the
   AMDP caveat (entropy alone can't tell *which* mode wins — but expected entropy *reduction* can
   compare actions). *Collapsible:* particle-weighted implementation details and why symmetric
   corridors defeat any policy that only looks one step ahead along the shortest path.
5. **Graph utility for active SLAM.** *Statement:* score a candidate trajectory by the predicted
   change in the pose-graph information: $\Delta_a = \log\det\Omega_{+a} - \log\det\Omega$, where
   $\Omega_{+a}$ augments $\Omega$ with the expected odometry factors along $a$ and expected
   loop-closure factors where $a$ re-observes mapped regions; combined utility
   $U(a) = w_I I_{map}(a) + w_G \Delta_a - w_C C(a)$. *Sketch (5 steps):* MAP-SLAM covariance
   $\approx \Omega^{-1}$ (Ch. 15); D-optimality as volume of the uncertainty ellipsoid,
   $\log\det$ as its log-volume; factor addition is additive in $\Omega$ (the information-form
   lesson of Chs. 11-lineage/15 paying off); predicted loop factors from map overlap; evaluate
   $\log\det$ per candidate via sparse Cholesky (`faer`), $O(n^{1.5})$-ish for planar graphs.
   *Collapsible:* A-/E-opt alternatives and why the field converged on D-opt (monotonicity,
   invariance — Placed et al. §V); tree-connectivity surrogates for large graphs.
6. **Stopping criteria.** *Statement (survey-honest, no theorem):* stop when expected gain per unit
   cost falls below a threshold tied to the task ($\max_a I(a)/C(a) < \epsilon_{task}$), or when
   map entropy plateaus over a window; absolute-entropy thresholds are map-size dependent and
   fragile; principled stopping remains an open problem (Placed et al. §VIII). Presented with the
   measured entropy/gain curves from the lab, not abstractly.

**Named algorithms:**

| Algorithm | Signature | Complexity |
|---|---|---|
| `detect_frontiers` | `(grid) -> Vec<Frontier>` | $O(\#cells)$ BFS + labeling |
| `expected_info_gain` | `(grid, pose, sensor) -> f64` | $O(\#beams \cdot range/res)$ per candidate |
| `score_candidates` | `(frontiers, slam, weights) -> Vec<(a, U)>` | gain + `a_star_grid` cost + optional $\Delta_a$ |
| `graph_d_opt` | `(pose_graph) -> f64` | one sparse Cholesky ($\log\det$ = 2·Σ log diag(L)) |
| `active_localize` | `(bel, candidates, sim) -> a` | $O(\#cand \cdot \#particles \cdot \#z_{sim})$ |
| `explore_step` (identify–select–execute) | `(slam_state) -> Decision` | pipeline of the above |

**Numeric micro-example:** one 5-cell ray into unknown space ($p_i = 0.5$ each, hit model from
Ch. 10 with $z_{hit} = 0.9$): compute reach probabilities, per-cell expected $H_b$ drop, total
$I \approx 2.1$ bits; then the same ray into known-free space: $I \approx 0.1$ bits. Two hand-
checkable tables that make "point the sensor at the unknown" quantitative — and unit-tested.

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w24.1: Frontier Chaser** *(flagship, interactive sim)* — type: wasm-sim, full-width
  dashboard. Left: the live map (Ch. 13 log-odds render, gray = ignorance) as Rusty explores a
  seeded random apartment; frontier regions outlined and *filled by utility* (posterior-purple
  ramp), the chosen target starred, the planned path (Ch. 20) and executed trajectory (Ch. 23)
  drawn live. Right rail: the map-entropy curve $H(m)$ falling in real time (green measurement
  ticks marking each scan), plus per-candidate utility bars. Autoplays a full run (~45 s,
  time-warped). The *one meaningful parameter*: the cost-weight slider $w_C/w_I$ — at one end
  "info-greedy" (crosses the flat to a big far frontier), at the other "step-greedy"
  (nearest-frontier busywork); the entropy-vs-distance-travelled chart re-plots per setting, and
  the reader watches greed lose on *per-meter* efficiency. *Misconception killed:* "exploration =
  go to the nearest unknown" — nearest is a special case of a utility, and rarely the right one.
- **Widget w24.2: Disambiguation Detour** *(flagship, interactive sim)* — active localization in a
  symmetric world: the Hallway with identical doors (the book's oldest friend, one last time).
  Rusty's MCL belief is bimodal (two blue particle clumps); the goal lies down a corridor that
  looks identical under both hypotheses; a side detour passes a *distinguishing* asymmetric
  alcove. Panel shows each candidate route scored by expected entropy reduction (bars); autoplay
  runs the entropy-greedy policy — Rusty takes the detour, the belief collapses to one purple
  clump at the alcove, then it commits confidently. Toggle **"greedy goal-seeker"**: Rusty heads
  straight down and, across seeded runs, ends at the wrong destination ~half the time (tally
  displayed). One parameter: sensor noise slider — as noise grows, the detour's information value
  shrinks below its cost and the *scored* decision flips, visibly. *Misconception killed:* "when
  lost, head for the goal and the filter will sort itself out" — motion is a sensing action, and
  sometimes the informative path *is* the optimal path.
- **Widget w24.3: Loop or Push On?** *(interactive sim, supporting)* — mid-exploration dilemma,
  staged: two candidate targets glow — a fresh frontier (high $I_{map}$) and a revisit of the start
  region (high $\Delta_a$ via an expected loop closure). Split preview: choosing each, then
  fast-forwarding — coverage-only yields more gray conquered but a visibly sheared map (graph
  drift, Ch. 16's rubber-band problem un-closed); the loop-closure choice pauses coverage but
  snaps $\Omega$ tight ($\log\det$ readout jumps; map straightens). Slider: $w_G$. *Misconception
  killed:* "a good exploration policy maximizes coverage" — without deliberate loop closing, the
  map you cover is a map you can't trust.
- **Widget w24.4: Stop Sign** *(small interactive chart, supporting)* — the recorded entropy and
  gain-per-meter curves of a full lab run; the reader drags a stopping threshold and sees where
  the run would have ended: map completeness %, time spent, and wasted-motion overlay. Nearly
  static (chart + one draggable line) — the designed-for-fallback widget. *Misconception killed:*
  "explore until 100%" — the last 3% of entropy costs a third of the run.

Color code discipline: unknown-space gray is the book's "ignorance" gray (Ch. 13); pre-decision
beliefs and candidate bars in prior-blue, expected-gain annotations in measurement-green, chosen
targets/collapsed beliefs in posterior-purple, ground truth gray dashed. All widgets autoplay
seeded defaults, expose one headline parameter, and ship build-time SVG fallbacks.

## 5. Practical (P) — Rust Implementation

Crates:
- `nalgebra` 0.35 — poses, candidate scoring math.
- `faer` 0.24 — sparse Cholesky for `graph_d_opt` (the Ch. 15 backend, reused).
- `rand` 0.9 (`SmallRng`, seeded) — randomized apartments, measurement simulation in
  `active_localize`.
- `rayon` — parallel candidate scoring natively; sequential on WASM.
- `eframe`/`egui` 0.35 + `egui_plot` 0.34 — widgets; `plotters` for build-time fallbacks.
- Depends on `sim`, `localize` (Ch. 12 MCL), `ch13_occgrid`, `ch15_graph`, `ch16_slam2d`
  (RustSLAM-2D), `ch20_planning`, `ch23_mppi` — this crate is the Part-VI integrator.

Module plan: `crates/ch24_explore/` with `src/frontier.rs`, `info_gain.rs`, `utility.rs`,
`active_loc.rs`, `explorer.rs` (the identify–select–execute loop), `examples/autonomous_explore.rs`,
`examples/detour_trial.rs`, `examples/stop_study.rs`.

```rust
use nalgebra::{Isometry2, Point2};

pub struct Frontier { pub cells: Vec<GridIdx>, pub centroid: Point2<f64>, pub size: usize }

/// BFS labeling of free-cells adjacent to unknown; filters regions below `min_size`.
pub fn detect_frontiers(grid: &ch13_occgrid::OccGrid, min_size: usize) -> Vec<Frontier>;

pub struct InfoGainEstimator<'a> {
    grid: &'a ch13_occgrid::OccGrid,
    sensor: sim::LidarParams,
    pub exact: bool,          // F.2 recursion vs. the unknown-cell-count surrogate
}
impl InfoGainEstimator<'_> {
    /// Expected map-entropy reduction (bits) of scanning once from `pose`. Eq.-linked to F.2.
    pub fn expected_gain(&self, pose: &Isometry2<f64>) -> f64;
}

pub struct UtilityWeights { pub w_i: f64, pub w_g: f64, pub w_c: f64 }

/// log det Ω via faer sparse Cholesky over the current pose graph
/// (Ch. 16's `PoseGraph`, built on the Ch. 15 optimizer).
pub fn graph_d_opt(graph: &ch16_slam2d::PoseGraph) -> f64;

pub enum Decision { Goto { target: Isometry2<f64>, path: Vec<Isometry2<f64>> },
                    CloseLoop { node: ch16_slam2d::NodeIx },
                    Done { reason: StopReason } }

/// The identify–select–execute loop, one tick: detect frontiers + loop candidates,
/// score U(a) = w_i·I + w_g·ΔlogdetΩ − w_c·C, plan with ch20, hand off to ch23.
pub struct Explorer {
    pub weights: UtilityWeights,
    pub stop: StopRule,       // gain-per-meter threshold and/or entropy-plateau window
    gain: InfoGainEstimator<'static>,
}
impl Explorer {
    pub fn decide(&mut self, slam: &ch16_slam2d::Slam2d) -> Decision;
}

/// Depth-1 expected-entropy-reduction action selection over an MCL belief (F.4).
pub fn active_localize(
    bel: &ch08_particles::ParticleSet<SE2>, // the Ch. 12 MCL belief (Ch. 8 machinery)
    candidates: &[Isometry2<f64>],
    world: &sim::World, n_z_samples: usize, rng: &mut rand::rngs::SmallRng,
) -> (usize, Vec<f64>);   // chosen index + per-candidate expected gains (widget bars)
```

Worked end-to-end example (`examples/autonomous_explore.rs`, seed 24): Rusty dropped into a
randomized apartment it has never seen; full stack live (MCL is off — SLAM's own pose estimate is
used, honestly noted). Ablation table printed and unit-test-pinned over 10 seeds: nearest-frontier
vs. info-rate ($w_g{=}0$) vs. full utility — columns: distance travelled to 95% of terminal map
information, final map entropy (bits), final trajectory ATE vs. ground truth (cm), loop closures
made. Expected shape of results (the design's success criterion): full utility travels ~15% farther
than nearest-frontier but halves final ATE — the w24.3 lesson, in numbers. `detour_trial` pins the
w24.2 tally (goal-greedy ≈ 50% wrong-room; entropy-greedy > 95% correct).

Runnable artifact: `cargo run --release --example autonomous_explore` — the book's pre-capstone:
an unmapped floorplan in, a finished map + closed graph + entropy log out, autonomously. The WASM
demo is w24.1 running this exact `Explorer`.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w24.1 | Frontier Chaser | wasm-sim | ch24_explore + ch16_slam2d + ch20_planning + ch23_mppi + eframe 0.35 | cost-weight slider, seed re-roll, pause/scrub, utility bars | utility-driven exploration; entropy as the falling score |
| w24.2 | Disambiguation Detour | wasm-sim | ch24_explore + localize (Ch. 12 MCL) + sim + eframe | policy toggle (entropy-greedy vs. goal-greedy), sensor-noise slider, run tally | active localization: motion as a sensing action |
| w24.3 | Loop or Push On? | wasm-sim | ch24_explore + ch15_graph + ch16_slam2d + eframe | choose target, w_G slider, split fast-forward preview | coverage vs. graph information; why explorers must close loops |
| w24.4 | Stop Sign | wasm-sim (chart) | ch24_explore + egui_plot 0.34 | drag stopping threshold on recorded curves | diminishing returns; principled stopping |
| f24.5 | Frontier anatomy (free/unknown/frontier cells) | static-svg | plotters (build-time) | — | the frontier definition, pixel-precise |

## 7. Exercises & Extensions

1. **(F)** Prove $I(a) \ge 0$ from the KL form, then construct an explicit two-cell, one-beam
   example where a specific measurement *increases* map entropy. Confirm your example numerically
   with `InfoGainEstimator` in a test.
2. **(F)** Work the 5-cell-ray micro-example by hand for $z_{hit} = 0.7$ and compare with the
   chapter's $0.9$ table: explain in one paragraph why a worse sensor lowers expected gain
   *sublinearly*.
3. **(C, predict-then-verify)** In w24.2, predict the sensor-noise level at which the scored
   decision flips from detour to direct (read the bars, don't run the policy) — then verify with
   the slider. Explain the flip in terms of $I(a)$ vs. $C(a)$.
4. **(C)** Use w24.1 to find a seeded apartment where nearest-frontier beats info-rate on
   *distance-to-95%-information*. What structural property of the floorplan makes greed win there?
5. **(P)** Implement frontier clustering by centroid-splitting (large regions → multiple candidate
   poses at the region's ends) and measure its effect on the ablation table's distance column.
6. **(P, stretch)** Replace the depth-1 `active_localize` with a POMCP search (Ch. 22's `pomcp.rs`)
   over a 3-action macro-space (detour / direct / wait-and-scan); quantify when the lookahead earns
   its compute on the w24.2 world.

## 8. Modernization Notes

- **Baseline honesty:** the draft PDF this book works from (1999–2000) contains no exploration
  chapter at all; the published 2005 edition's Ch. 17 covered greedy information-gain exploration,
  active localization, and coordinated multi-robot coverage. This chapter is therefore *rebuilt*,
  not revised: the 2005 skeleton (entropy objectives, greedy one-step selection, active
  localization) survives, but the field's frame is now Placed et al. 2023 — active SLAM as an
  intractable POMDP approached via identify–select–execute, with information-theoretic utilities
  over the *graph*, not just the grid.
- **Added beyond the 2005 baseline:** frontier exploration as the structural backbone (Yamauchi
  1997 — older than the baseline yet absent from it); D-/A-/E-optimality vocabulary and
  $\log\det\Omega$ utilities over the Ch. 15 factor graph (the 2005 book scored utility over
  occupancy grids only); loop-closure-seeking as a first-class exploration objective; stopping
  criteria as an explicit (and explicitly open) problem; the entropy-vs-ATE ablation methodology.
- **Dropped, with pointers:** multi-robot exploration and market-based coordination (2005 Ch. 17's
  centerpiece — one pointer box; the machinery generalizes but the book's lab is single-robot);
  coverage path planning (Choset Ch. 6 boustrophedon — one paragraph, since the hook parodies it);
  RBPF-specific exploration utilities (Stachniss) — historical note now that the book's SLAM spine
  is the graph; learned exploration policies and neural map-completion priors — pointer to Ch. 25
  and the survey's open-problems section.
- **Scope caveat stated in-chapter:** our expected-gain computation inherits Ch. 13's per-cell
  independence lie; the text quantifies the bias on one worked scene rather than hiding it — the
  book's standing policy of naming its approximations.
