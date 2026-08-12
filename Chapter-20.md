# Chapter 20 — Motion Planning: From Geometry to Probability

> Part VI — Planning and Acting under Uncertainty · Estimated length: 10 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Parts II–V made Rusty *know* where it is and what the world looks like. This chapter finally makes it
*go somewhere*. It is the deliberately geometric interlude of Part VI: uncertainty steps back for one
chapter so the reader can master the deterministic skeleton of acting — configuration space, graph
search, potential fields, and sampling-based planners — before Chapters 21–24 wrap it back in
probability. The title's "probability" enters twice and the chapter is explicit about the difference:
(1) randomness as an *algorithmic device* (PRM/RRT sample configurations the way Ch. 8 sampled
states), and (2) planners as the *deterministic core* that the uncertainty-aware layers (MDPs,
POMDPs, MPPI, exploration) will drive. The "aha": a planner is a search over a space the robot never
physically visits — configuration space — and once you see obstacles as C-obstacles, every planner in
the zoo is "build a graph in $\mathcal{Q}_{free}$, then search it."

Story line:
1. **Hook:** give Rusty a goal across the Apartment; the naive "drive toward the goal" controller
   wedges into the first wall (small autoplay widget, reusing the Ch. 4 sim).
2. **Play (C):** the Planner Arena — four planners race on the same query before any formalism.
3. **Formalize (F):** workspace vs. configuration space; C-obstacles; paths as continuous maps;
   completeness taxonomy (complete / resolution-complete / probabilistically complete).
4. **Search done right:** Dijkstra and A* with proofs, on the occupancy lattice from Ch. 13.
5. **Potential fields** and their local-minima disease; wave-front/Brushfire as the grid cure — and
   the reveal that Brushfire *is* the Ch. 19 distance transform.
6. **Sampling-based planning:** PRM, RRT, RRT*, with probabilistic completeness and asymptotic
   optimality stated precisely (this is where the chapter earns its "stated precisely" promise).
7. **Rusty is not a point:** nonholonomic constraints, Dubins/Reeds–Shepp steering, hybrid-A* sketch.
8. **Integration lab:** plan in the Apartment with RRT* + Dubins steering; hand the path to a naive
   tracker and watch it *almost* work — the cliffhanger that motivates Chs. 21 and 23.

## 2. Prerequisites & Position

- **Builds on:** Ch. 3 (SE(2) poses, $\boxplus/\boxminus$), Ch. 4 (Rusty, the Apartment world,
  `parry2d` collision), Ch. 8 (sampling as representation — the mental model reused by PRM/RRT),
  Ch. 13 (occupancy grid as the planning substrate), Ch. 19 (ESDF distance fields for clearance
  costs and potential fields).
- **Feeds into:** Ch. 21 (plans vs. policies under action noise), Ch. 23 (MPPI tracks this chapter's
  paths), Ch. 24 (planning to frontiers), Ch. 26 (capstone stack).
- **Baseline sources:** Choset et al. Ch. 3 (configuration space, §3.1–3.4), Ch. 4 (potential
  functions, §4.1–4.5), Ch. 5 §5.1 (visibility graph, as exercise), Ch. 7 (sampling-based
  algorithms, §7.1–7.4), App. G (completeness), App. H (A*/Dijkstra/D*); Lynch & Park Ch. 10
  (§10.1–10.6, whole-chapter planning survey), Ch. 13 §13.3 (nonholonomic mobile robots, Dubins/
  Reeds–Shepp context); Spong et al. Ch. 7 (§7.1–7.4: C-space, potential fields, PRM/RRT);
  modernization set: Karaman & Frazzoli 2011 (RRT*), Dolgov/Thrun et al. 2008–2010 (hybrid A*),
  LaValle *Planning Algorithms* as free companion reference. Thrun et al. (draft) has **no**
  geometric-planning chapter — this chapter is sourced from the motion-planning baselines.

## 3. Foundation (F) — Mathematical Core

**Notation introduced** (chapter-scoped table at the top of the F section):

| Symbol | Meaning |
|---|---|
| $\mathcal{W} \subset \mathbb{R}^2$, $\mathcal{WO}_i$ | workspace, workspace obstacles |
| $\mathcal{Q}$, $q$ | configuration space, configuration ($q = (x, y, \theta)^\top \in SE(2)$ for Rusty) |
| $R(q) \subset \mathcal{W}$ | footprint of the robot at configuration $q$ |
| $\mathcal{QO}_i = \{q : R(q) \cap \mathcal{WO}_i \neq \emptyset\}$, $\mathcal{Q}_{free}$ | C-obstacle, free space |
| $\tau : [0,1] \to \mathcal{Q}_{free}$ | path (continuous map); $c(\tau)$ its cost/length |
| $h(v)$, $g(v)$ | heuristic and cost-to-come in A* |
| $U(q) = U_{att}(q) + U_{rep}(q)$ | artificial potential |
| $r_n = \gamma_{RRT^*} (\log n / n)^{1/d}$ | RRT* connection radius, $d = \dim \mathcal{Q}$ |
| $\rho$ | minimum turning radius (Dubins/Reeds–Shepp) |

Note the deliberate contrast with book-wide notation: $q$ is a *configuration the planner imagines*,
$x_t$ is the *state the filter believes*; for Rusty they coincide numerically, and the text says so
once, precisely.

**Definitions:** configuration space; C-obstacle; free space; path vs. trajectory (Choset's
distinction: geometric curve vs. time-parameterized); planner properties — completeness,
resolution completeness, probabilistic completeness, optimality, asymptotic optimality (each given
a one-sentence formal definition in a definition box); admissible and consistent heuristics;
nonholonomic constraint (Pfaffian form $\dot{x}\sin\theta - \dot{y}\cos\theta = 0$).

**Key derivations** (each: name, statement, sketch, collapsible full version):

1. **C-obstacle of a disc robot = Minkowski inflation.** *Statement:* for a disc robot of radius
   $r$, $\mathcal{QO} = \mathcal{WO} \oplus B_r$, so planning for the disc equals planning for a
   point in the map inflated by $r$. *Sketch (3 steps):* footprint at $q$ intersects obstacle ⇔
   center within distance $r$ of obstacle ⇔ center inside the Minkowski sum. *Collapsible:* general
   Minkowski-sum C-obstacles for polygons (Choset App. F star algorithm), and why $SE(2)$ footprints
   make the inflation $\theta$-dependent for non-circular robots.
2. **A\* optimality under admissibility.** *Statement:* with $h$ admissible, A* returns a minimal-cost
   path; with $h$ consistent, no node is re-expanded. *Sketch (5 steps):* suppose goal popped with
   suboptimal $g$; some frontier node $v$ on an optimal path has $f(v) = g^*(v) + h(v) \le c^* <
   f(goal)$; contradiction with priority order; consistency ⇒ $f$ non-decreasing along expansions ⇒
   closed-set safety. *Collapsible:* full proof plus the classic non-optimistic counterexample grid
   (Choset App. H.2), and Dijkstra as $h \equiv 0$.
3. **Local minima of additive potentials.** *Statement:* $U_{att} + U_{rep}$ admits spurious minima
   in $\mathcal{Q}_{free}$ for non-convex obstacle layouts; no smooth potential on a space with
   "holes" can have a single minimum unless it is a navigation function, which exists but is only
   constructive for sphere/star worlds (Rimon–Koditschek). *Sketch:* construct the U-trap; show
   $\nabla U = 0$ with positive definite Hessian; state the navigation-function existence result.
   *Collapsible:* the sphere-world navigation function formula and its diffeomorphism argument
   (Choset §4.6), wave-front planner as the discrete navigation function.
4. **Probabilistic completeness of PRM/RRT.** *Statement (precise):* if a query admits a path with
   clearance $\delta > 0$, then the probability that PRM (with uniform sampling, $n$ samples) fails
   to answer it satisfies $P(\text{fail}) \le a e^{-b n}$ for constants $a, b > 0$ depending on
   $\delta$, path length, and $\mu(\mathcal{Q}_{free})$; the same exponential decay holds for RRT.
   *Sketch (5 steps):* cover the $\delta$-clear path with $m$ balls of radius $\delta/2$; a sample
   landing in consecutive balls yields connectable milestones; failure requires some ball to receive
   no sample; union bound over balls gives the exponential. *Collapsible:* the full covering argument
   with constants, plus the $(\epsilon,\alpha,\beta)$-expansiveness view (Choset §7.4) explaining
   *narrow passages*: the constants blow up as visibility sets shrink.
5. **Asymptotic (sub)optimality — Karaman & Frazzoli, stated precisely.** *Statement:* (i) RRT
   converges to a *suboptimal* solution with probability one — $P(\lim_{n\to\infty} c_n^{RRT} =
   c^*) = 0$; (ii) RRT* with connection radius $r_n = \gamma_{RRT^*}(\log n / n)^{1/d}$ and
   $\gamma_{RRT^*} > 2(1 + 1/d)^{1/d} (\mu(\mathcal{Q}_{free})/\zeta_d)^{1/d}$ is *almost surely
   asymptotically optimal* — $P(\lim_{n\to\infty} c_n^{RRT^*} = c^*) = 1$ — while keeping
   $O(\log n)$ expected per-iteration cost. *Sketch (4 steps):* RRT's failure: the root-anchored tree
   commits to homotopy-suboptimal cheap-to-reach edges and never rewires; RRT*'s repair: choose-parent
   + rewire inside $r_n$-balls; the $\log n / n$ radius shrinks slowly enough that consecutive balls
   along an optimal path keep receiving samples (percolation-style argument). *Collapsible:* theorem
   statements verbatim with the measure-theoretic assumptions ($c^*$ robustly optimal), and what
   breaks without them.
6. **Dubins shortest paths.** *Statement:* for a forward-only car with turning radius $\rho$, the
   optimal path between two poses is one of six words $\{LSL, RSR, LSR, RSL, RLR, LRL\}$ (arcs
   $L/R$ at full lock, straight $S$). *Sketch:* Pontryagin's maximum principle forces bang-bang
   steering; enumerate switch structures; case analysis. *Collapsible:* the closed-form arc-length
   formulas per word (the ones the Rust module implements) and the Reeds–Shepp extension to 48 word
   classes when reverse is allowed.

**Named algorithms** (no Thrun tables exist for this material; names follow Choset's algorithm
numbering and standard usage):

| Algorithm | Signature | Complexity |
|---|---|---|
| `dijkstra` | `(G, s) -> dist[]` | $O((V+E)\log V)$ binary heap |
| `a_star` | `(G, s, g, h) -> Option<Path>` | worst-case Dijkstra; expansions shrink with $h$ |
| `gradient_descent_potential` (Choset Alg. 4) | `(q_s, U, step) -> Path ∪ {stuck}` | per-step $O(\text{obstacle terms})$ |
| `brushfire` / `wavefront` | `(grid, seeds) -> dist_grid` | $O(\#cells)$ BFS |
| `build_prm` (Choset Algs. 6–7) | `(n, k, sample, connect) -> Roadmap` | $O(n \log n)$ NN + $nk$ local plans |
| `rrt_extend` / `build_rrt` (Choset Algs. 10–13) | `(T, q_rand) -> Status` | $O(\log n)$ NN per iteration |
| `rrt_star` | `(q_s, q_g, n) -> Tree` | $O(n \log n)$ total expected |
| `dubins_shortest_path` | `(q_0, q_1, ρ) -> (Word, len)` | $O(1)$ — six closed forms |
| `hybrid_a_star` (sketch) | `(grid, q_s, q_g, prims) -> Path` | lattice-A* with continuous cell states |

Hybrid A* is presented as a one-page *sketch* (motion primitives over an $(x, y, \theta)$ lattice,
one continuous state kept per cell, analytic Dubins/RS shots near the goal, dual heuristic
$\max(h_{\text{nonholonomic, no obstacles}}, h_{\text{holonomic, obstacles}})$), explicitly flagged
as recipe-level, not derived. D* Lite gets a paragraph (replanning when the grid changes) with a
pointer, not an implementation.

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w20.1: Planner Arena** *(flagship, interactive sim)* — type: wasm-sim. Same Apartment
  map, same start/goal query; four lanes: A* on the 8-connected lattice, PRM, RRT, RRT*. Reader
  presses **Race** (autoplays once on load with a fixed seed); watches wavefront/roadmap/tree growth
  animated per planner, then the returned paths overlaid. Manipulates: the *one meaningful
  parameter* is the sample/expansion budget slider (shared across lanes); secondary controls behind
  a disclosure: re-roll seed, drag start/goal, toggle "keep refining" for RRT*. Observes: live
  scoreboard — time-to-first-solution, path cost, nodes in memory; RRT's jagged first answer; RRT*'s
  cost curve decaying toward the A* lattice cost as the budget grows. *Misconception killed:* "RRT
  finds good paths" — it finds *feasible* paths fast; optimality is a different contract you pay
  for. Static fallback: three-frame filmstrip (early/mid/final) + final scoreboard table.
- **Widget w20.2: Narrow Passage** *(interactive sim)* — two rooms joined by a corridor whose width
  is the single slider. PRM samples rain down (autoplaying); samples inside the passage highlighted;
  a success-probability meter (fraction of 100 seeded trials connecting the query) updates as width
  shrinks. Toggle: bridge-test sampling, which visibly concentrates samples in the passage.
  Observes: success probability collapsing roughly with passage volume — the visibility intuition
  from $(\epsilon,\alpha,\beta)$-expansiveness made physical. *Misconception killed:* "uniform
  random sampling sees everything at practical sample counts."
- **Widget w20.3: Potential Well** *(interactive sim, supporting)* — drag the goal and a U-shaped
  obstacle; a bead follows $-\nabla U$ (autoplay), trailing its path; potential rendered as a
  heightfield underlay. When trapped, a "stuck" badge appears; toggling **Wave-front mode** swaps
  $U$ for the BFS distance-to-goal field and the bead escapes. *Misconception killed:* "just follow
  the gradient" — and it plants the seed that a *value function* (Ch. 21) is the principled
  gradient field.
- **Widget w20.4: Dubins Dial** *(animation + drag, supporting)* — drag two poses (position +
  heading handle); all six Dubins words drawn faintly, the optimal word bold; length readouts;
  $\rho$ slider; toggle Reeds–Shepp to allow cusps/reverse. *Misconception killed:* "a car's
  shortest path is turn–straight–turn toward the goal" (watch $LRL$ win at close range).

Color code: planners are *not* belief roles, so the arena uses the neutral widget palette, but the
book code appears where it belongs — ground truth map in gray, and in w20.3 the wave-front
distance field reuses the Ch. 19 ESDF colormap so the "Brushfire = distance transform" reveal is
visual. Dashboard layout: w20.1 is full-width; w20.2–w20.4 are half-width cards inline with their
sections. All widgets autoplay a seeded default and degrade to pre-rendered SVG (per pedagogy
findings: autoplay-first, one meaningful parameter, static fallback).

## 5. Practical (P) — Rust Implementation

Crates:
- `petgraph` 0.8 — roadmap and lattice graph storage (`StableGraph`), used by PRM and hybrid A*.
- `pathfinding` 4.15 — reference `astar`/`dijkstra`; our hand-rolled A* is cross-checked against it
  in tests (the Ch. 6 `adskalman` pattern, repeated).
- `parry2d` 0.30 — footprint/segment collision queries against the Apartment (`intersection_test`,
  shape-cast for swept edges).
- `nalgebra` 0.35 — `Isometry2`, `Point2`, fixed-size states.
- `rand` 0.9 (`Pcg64`) — seeded sampling for PRM/RRT; every figure reproducible.
- `eframe`/`egui` 0.35 + `egui_plot` 0.34 — the four widgets.

Module plan: `crates/ch20_planning/` with `src/cspace.rs`, `search.rs`, `potential.rs`, `prm.rs`,
`rrt.rs` (RRT + RRT* sharing a tree type), `steer.rs` (straight-line/Dubins/Reeds–Shepp),
`hybrid_astar.rs`, `examples/planner_race.rs`, `examples/dubins_park.rs`. Depends on `sim`
(Ch. 4 worlds) and `ch13_occgrid` (OccGrid + inflation), `ch19_maps` (ESDF for clearance costs).

```rust
use nalgebra::{Isometry2, Point2};

/// Configuration space of a disc robot over an inflated occupancy grid,
/// with exact parry2d checks for the non-circular footprint variant.
pub struct CSpace2<'w> {
    world: &'w sim::World,
    inflated: ch13_occgrid::OccGrid,   // Minkowski-inflated, from the derivation F.1
    footprint: parry2d::shape::ConvexPolygon,
}

impl CSpace2<'_> {
    pub fn is_free(&self, q: &Isometry2<f64>) -> bool;
    /// Swept collision check along the segment, resolution `step` (used by PRM/RRT edges).
    pub fn edge_free(&self, a: &Isometry2<f64>, b: &Isometry2<f64>, step: f64) -> bool;
}

/// Local planner abstraction: straight-line for holonomic point, Dubins/RS for Rusty.
pub trait Steer {
    /// Path from `a` toward `b` (possibly truncated at `max_len`), with its cost.
    fn steer(&self, a: &Isometry2<f64>, b: &Isometry2<f64>, max_len: f64)
        -> Option<(Vec<Isometry2<f64>>, f64)>;
}
pub struct StraightLine;
pub struct Dubins { pub rho: f64 }
pub struct ReedsShepp { pub rho: f64 }

pub struct RrtStar<S: Steer> {
    steer: S,
    gamma: f64,            // γ_RRT* — checked against the F.5 lower bound in debug builds
    nodes: Vec<Node>,      // parent, cost-to-come
    nn: KdTree2,           // hand-rolled 2-d tree in this crate; no extra dependency
}
impl<S: Steer> RrtStar<S> {
    pub fn plan(&mut self, cs: &CSpace2, q_s: Isometry2<f64>, q_g: Isometry2<f64>,
                n: usize, rng: &mut rand::rngs::SmallRng) -> Option<PathResult>;
}

pub fn a_star_grid(
    grid: &ch13_occgrid::OccGrid, start: GridIdx, goal: GridIdx,
    h: impl Fn(GridIdx) -> f64,
) -> Option<(Vec<GridIdx>, f64)>;
```

Worked end-to-end example (`examples/planner_race.rs`, seed 20): one Apartment query. Expected
output table (reproduced by a unit test): A* lattice cost 23.4 m in 4.1 ms; PRM ($n{=}2000$,
$k{=}10$) 24.9 m; RRT first solution 31.7 m at 6 ms; RRT* 24.1 m after 5000 samples with the cost
curve logged to CSV; then `Dubins`-steered RRT* ($\rho = 0.4$ m) yielding a drivable path 26.0 m
that the straight-line planners cannot execute. `plotters` renders the four-panel comparison figure
used as w20.1's static fallback.

Runnable artifact: `cargo run --example planner_race` prints the scoreboard and writes the figure;
the WASM build of the same crate *is* w20.1 — the arena's planners are these exact functions.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w20.1 | Planner Arena | wasm-sim | ch20_planning + sim + eframe 0.35 + egui_plot 0.34 | race button, budget slider, drag start/goal, seed re-roll | feasibility vs. optimality vs. compute across planner families |
| w20.2 | Narrow Passage | wasm-sim | ch20_planning + eframe | corridor-width slider, bridge-sampling toggle | why sampling struggles; visibility/expansiveness intuition |
| w20.3 | Potential Well | wasm-sim | ch20_planning + ch19_maps + eframe | drag goal/obstacle, wave-front toggle | local minima; distance fields as the cure |
| w20.4 | Dubins Dial | animation + drag | ch20_planning + eframe | drag poses, ρ slider, RS toggle | nonholonomic shortest paths are word-structured |
| f20.5 | C-space morph (arm & disc robot) | static-svg | plotters (build-time) | — | workspace obstacle ⇒ C-obstacle mapping |

## 7. Exercises & Extensions

1. **(F)** Derive the C-obstacle of a disc robot of radius $r$ against a convex polygonal obstacle;
   show the boundary consists of edge-offsets and circular arcs. Then explain in two sentences why a
   rectangular Rusty makes $\mathcal{QO}$ depend on $\theta$.
2. **(F)** Exhibit a 4-node graph and an inadmissible heuristic for which A* returns a suboptimal
   path; verify with the crate's `a_star_grid` in a unit test.
3. **(C, predict-then-verify)** In w20.2, the corridor is at width 0.6 m with success probability
   ≈ 0.9 at $n = 2000$. Predict the success probability at width 0.3 m before moving the slider;
   reconcile your prediction with the exponential-decay statement in F.4.
4. **(C)** Use w20.1 to find a query where RRT beats A* on time-to-first-solution by 10× — then a
   query where the lattice wins. State the structural property of each map that decides the winner.
5. **(P)** Implement bidirectional RRT-Connect in `rrt.rs` and add it as a fifth lane to the arena;
   benchmark on the narrow-passage map.
6. **(P, stretch)** Implement shortcut smoothing (random pairwise shortcutting with `edge_free`)
   and quantify: cost reduction vs. iterations on 50 seeded RRT paths.

## 8. Modernization Notes

- The Thrun draft contains no geometric motion planning at all; this chapter is built from Choset,
  Lynch & Park, and Spong, then modernized past all three: **RRT\*** (Karaman & Frazzoli 2011) and
  its precise asymptotic-optimality statement post-date Choset 2005 and are absent from Spong 2020's
  planning chapter; **hybrid A\*** (DARPA-era, Dolgov/Thrun) appears in none of the baselines.
  Informed RRT*/BIT* are name-checked with pointers only — the RRT* proof idea is the pedagogical
  payload, not the planner zoo.
- Dropped from Choset: Canny's roadmap and silhouette methods, GVD/GVG construction machinery, and
  cell decompositions (§6) — mathematically rich but off the book's spine; boustrophedon coverage
  gets one pointer paragraph in Ch. 24. The visibility graph survives only as exercise material.
  Bug algorithms dropped entirely (sensor-based reactive planning is subsumed by Ch. 23's DWA/MPPI).
- D* family compressed to a mention: modern stacks replan by re-running A*/hybrid-A* at costmap
  rate, and Ch. 23's receding-horizon control absorbs the reactive role D* served in 2005.
- Kept deliberately classical: potential fields, despite their flaws — because their failure sets up
  value functions (Ch. 21) and ESDF costs (Chs. 19/23), and because the wave-front planner is
  secretly the reader's first value iteration.
