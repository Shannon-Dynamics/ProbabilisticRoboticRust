# Chapter 26 — Capstone: A Complete Autonomous Robot

> Part VII — Frontiers and Integration · Estimated length: 12 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Twenty-five chapters built parts; this chapter builds *the robot*. Rusty is dropped into an
apartment it has never seen (randomized floorplan, fixed seed available) and must map it
autonomously: exploration chooses where to learn (Ch. 24), SLAM builds the map and the pose
belief (Ch. 16), a planner routes through partial knowledge (Ch. 20), MPPI drives the wheels
(Ch. 23) — all as concurrent tasks exchanging *stamped, frame-tagged beliefs*, not bare numbers.
The "aha" is architectural: an autonomy stack is not one algorithm but a tower of stated
approximations of a single intractable POMDP, and each layer's assumption is precisely a failure
mode waiting for the probabilistic machinery to catch. So we break it on purpose — kidnap Rusty,
walk a person through the LiDAR, cut the sensor — and watch detection, mode-switching, and
recovery work. The chapter closes with an engineering retrospective (what Rust's type system
caught during the writing of this book, with real compile errors; what it cost) and pointers
onward (ROS 2 via ros2-rust, hardware, 3D). The closing argument: the entire teaching stack runs
at real time, in the reader's browser, and every internal is inspectable.

Story line:

1. Hook: the Grand Demo runs on page load — 90 seconds from empty map to full coverage. Then the
   narrator presses "kidnap" and it *recovers*. The chapter promises to explain every panel.
2. Problem: why not one big filter/planner? The mission POMDP and why it is hopeless directly.
3. Intuition: the stack as a tower of approximations; boxes-and-arrows with rates and staleness.
4. Formalism: interface contracts (belief + stamp + frame), certainty-equivalence and its safety
   patch (covariance-inflated ESDF margins), latency budget, failure-detection tests.
5. Algorithm/implementation: the `Task` trait, the message bus, the supervisor state machine;
   wiring Chs. 9/10/12/13/15/16/19/20/23/24 artifacts together; native threads vs. WASM
   round-robin with identical semantics.
6. Experiment: the failure-modes tour (kidnap / dynamic obstacle / sensor dropout) with recovery,
   run live; the retrospective; where to go next.

## 2. Prerequisites & Position

This is the integration chapter: it **builds on every chapter of the book**. Exhaustive
dependency table (artifact = what is literally linked into `crates/capstone`; background = load-
bearing concept, no direct code import):

| Chapter | Contribution to the capstone | Kind |
|---|---|---|
| Ch. 1 | the hallway thesis the demo finally cashes in | background |
| Ch. 2 | probability, entropy (map-entropy stopping criterion) | background |
| Ch. 3 | `SE2`, frame conventions, frame-safety newtypes | artifact |
| Ch. 4 | `sim` crate: Apartment generator, Rusty, LiDAR, encoders, widget chrome | artifact |
| Ch. 5 | `BayesFilter` trait; predict/correct vocabulary | artifact |
| Ch. 6 | KF machinery; NIS/innovation test reused for divergence detection | artifact |
| Ch. 7 | error-state EKF on `SE2` (odometry prediction inside SLAM front-end) | artifact |
| Ch. 8 | particle machinery + low-variance resampler (relocalization) | artifact |
| Ch. 9 | `motion` module: odometry model + samplers | artifact |
| Ch. 10 | `sensor` module: likelihood field (scan-match scoring, relocalization weights) | artifact |
| Ch. 11 | Mahalanobis gating (loop-closure verification), localization taxonomy | background |
| Ch. 12 | `AugmentedMcl` ($w_{fast}/w_{slow}$) — global relocalization + kidnap detector | artifact |
| Ch. 13 | `OccGrid` log-odds mapping (submaps + global costmap substrate) | artifact |
| Ch. 14 | why not EKF SLAM here: quadratic cost + inconsistency (motivates Ch. 16 stack) | background |
| Ch. 15 | sparse GN/LM pose-graph optimizer (SLAM back-end); robust kernels | artifact |
| Ch. 16 | **RustSLAM-2D**: ICP odometry + loop closure + pose graph + submaps | artifact |
| Ch. 17 | Rao-Blackwellization perspective; why we chose pose-graph over RBPF | background |
| Ch. 18 | marginalization/fill-in caveats (why the graph is kept, not marginalized) | background |
| Ch. 19 | ESDF (distance transform of `OccGrid`) → MPPI obstacle costs | artifact |
| Ch. 20 | `planning` module: A* on lattice + Dubins steering (global planner) | artifact |
| Ch. 21 | certainty-equivalence framing; value-of-a-policy vocabulary | background |
| Ch. 22 | belief-space reasoning: the supervisor's "relocalize before committing" rule | background |
| Ch. 23 | `Mppi` controller (rollouts over ESDF costs) | artifact |
| Ch. 24 | frontier detector + information-gain scorer + stopping criterion | artifact |
| Ch. 25 | optional calibrated learned sensor model (demo toggle); retrospective data | artifact |

- **Feeds into:** nothing — terminal chapter. Appendix D documents the widget/simulator framework
  the Grand Demo stresses hardest.
- **Baseline sources:** the whole book (see table). The 1999–2000 Thrun draft has **no systems
  chapter**; its nearest ancestors are the open-ended "Projects" sections (draft Ch. 10.6, 14.9,
  16.8) and the museum-robot narratives in Ch. 1. Systems framing follows the Choset et al.
  taxonomy style; the architecture mirrors the modern ROS 2 stack (SLAM Toolbox + Nav2 costmaps +
  controller server) as described in the modernization set, so readers can map each of our tasks
  onto a production counterpart by name.

## 3. Foundation (F) — Mathematical Core

The capstone's F-section is systems mathematics: no new estimators, but precise statements about
*composition* — the assumptions each layer adds, and the tests that detect their violation.

**Notation introduced this chapter:**

| Symbol | Meaning |
|---|---|
| $b_t$ | mission-level belief (pose belief + map posterior), per TOC POMDP row |
| $\tau_{i}$ | period of task $i$; $\varsigma_i$ its worst-case staleness |
| $\rho_t = w_{fast}/w_{slow}$ | dual-EMA fitness ratio (Ch. 12 recovery statistic, reused) |
| $\epsilon_t = \nu_t^\top S_t^{-1} \nu_t$ | NIS divergence statistic (innovation $\nu_t$) |
| $H(m_t)$ | occupancy-map entropy; $\dot H$ its rate (stopping criterion) |
| $\delta$ | collision-chance bound; $k_\sigma = \Phi^{-1}(1-\delta)$ inflation gain |

**Definitions:**

- **D26.1 Autonomy stack.** A set of tasks $\{T_i\}$ with periods $\tau_i$, exchanging typed
  messages; each message is a `Stamped` value: payload + timestamp + frame id. An estimator task
  publishes *beliefs* (mean + covariance, or particles), never bare states.
- **D26.2 Mission.** Maximize expected map information subject to safety:
  $\pi^* = \arg\max_\pi \mathbb{E}\big[\sum_t \gamma^t\, r(b_t, u_t)\big]$ with
  $r = \Delta H(m)$ per unit cost, subject to $P(\text{collision}) \le \delta$ — a POMDP over
  $b_t$ (Ch. 22's frame, at building scale).
- **D26.3 Stopping criterion.** Terminate when no frontier of area $\ge A_{min}$ remains **and**
  $|\dot H(m_t)| < h_{min}$ over a trailing window (both conditions; either alone is gameable).
- **D26.4 Mode.** The supervisor's discrete state:
  `Explore | Navigate | Relocalize | Recover(kind) | Done` — an exhaustive enum, by design (§5).

**Key derivations** (name · statement · sketch · collapsible):

1. **The layer tower (why stacks are layered).** *Statement:* each subsystem is the mission POMDP
   under one named approximation: frontier exploration = one-step information-gain greedy
   (Ch. 24); planning on the current map = certainty equivalence in the map (Ch. 21's frame);
   MPPI = receding-horizon stochastic control with known state (Ch. 23); SLAM = MAP point estimate
   of $b_t$'s map marginal (Ch. 16). *Sketch (5 steps):* write D26.2; substitute each
   approximation in turn; tabulate assumption ↔ induced failure mode (greedy → oscillating
   targets; certainty equivalence → planning through unseen walls; known-state control → drift
   during relocalization; MAP map → confident wrong loop closures). *Collapsible:* the formal
   chain of substitutions. This table *is* the chapter's conceptual spine — the failure tour in §4
   demonstrates one row each.
2. **Safety under pose uncertainty (chance-constraint margin).** *Statement:* planning and MPPI on
   an ESDF are safe at level $\delta$ if trajectories keep
   $d_{esdf}(x) \ge r_{robot} + k_\sigma \sigma_{pose}$ with $k_\sigma = \Phi^{-1}(1-\delta)$,
   where $\sigma_{pose}^2$ is the largest eigenvalue of the position block of $\Sigma_t$.
   *Sketch (4 steps):* collision iff true clearance < 0; true position = estimate + Gaussian
   error; one-dimensional worst-case projection onto the nearest-obstacle direction; Gaussian tail
   bound. *Collapsible:* the projection argument and why it is conservative (union over the path
   needs a Bonferroni note).
3. **Latency budget (staleness bound).** *Statement:* a dynamic obstacle moving at $v_{obs}$ is
   handled iff $v_{obs} (\varsigma_{map} + \tau_{plan} + \tau_{ctrl}) + \frac{v_{rusty}^2}{2 a_{max}}
   < d_{detect}$; with the book's default rates (LiDAR 10 Hz, SLAM 10 Hz, costmap 5 Hz, MPPI
   20 Hz) the demo's walker at 0.8 m/s clears the bound with 2.1 m detection range. *Sketch (3
   steps):* worst-case pipeline delay chain; add braking distance; compare with sensing horizon.
   *Collapsible:* the arithmetic with the demo's actual constants (reader can verify in the
   Timing panel).
4. **Failure-detection tests (three named statistics).** *Statement:* (a) kidnap/mislocalization —
   dual-EMA scan-match fitness ratio $\rho_t < \rho_{min}$ (Ch. 12's $w_{fast}/w_{slow}$
   transplanted from particle weights to ICP fitness, with the same exponential-forgetting
   justification); (b) filter divergence — NIS $\epsilon_t$ exceeding the $\chi^2_m$ 95% gate for
   $k$ consecutive scans (Ch. 6/11 machinery); (c) sensor dropout — watchdog on message age
   $\varsigma_{scan} > 3\tau_{scan}$. *Sketch (4 steps each, short):* statistic definition; its
   distribution under nominal operation; threshold choice; which `Recover` branch it triggers.
   *Collapsible:* forgetting-factor algebra for (a); chi-square table lookup discipline for (b).

**Named algorithms** (signature · complexity; per-tick costs for the default Apartment, ~120×90
cells, 360-beam LiDAR):

| Algorithm | Signature | Complexity / rate |
|---|---|---|
| `stack_tick` | `(bus, tasks, now) -> ()` — run every due task once, in topological order on WASM | sum of below; 60 fps budget kept |
| `slam_tick` | `(scan, odom) -> Stamped<PoseBelief> + MapPatch` (Ch. 16 pipeline) | ICP $O(k \cdot B)$; graph solve amortized, sparse |
| `frontier_tick` | `(OccGrid) -> Vec<ScoredFrontier>` (Ch. 24) | $O(\text{cells})$ |
| `plan_tick` | `(map, esdf, goal) -> Path` — A* + Dubins smoothing (Ch. 20) | $O(E \log V)$ |
| `control_tick` | `(path, esdf, bel) -> Cmd` — MPPI (Ch. 23) | $O(K \cdot H)$, $K{=}256$, $H{=}30$ |
| `supervisor_step` | `(Mode, Events) -> Mode` — D26.4 state machine | $O(1)$ |
| `kidnap_detector` | `(fitness) -> bool` — F4(a) | $O(1)$ |
| `relocalize_global` | `(scan stream, frozen map) -> PoseBelief` — Ch. 12 `AugmentedMcl` over the SLAM map until ESS concentrates | $O(M \cdot B)$ per scan |
| `replan_on_surprise` | `(path, esdf, novelty cells) -> Path` — invalidate + replan; novelty via Ch. 12's `test_range_measurement` idea | $O(E \log V)$ worst case |
| `run_mission` | `(seed, MissionCfg) -> MissionReport` | full mission, deterministic under seed |

## 4. Conceptual (C) — Intuition & Visual Design

Color code: pose belief **purple**, predictions/rollouts **orange**, measurements/scans **green**,
the map in Ch. 13's grayscale (white = confident-free, dark = occupied, gray = ignorance),
ground truth **gray dashed** (available as an overlay toggle, default off — the reader should
feel the epistemic situation Rusty is in).

- **Widget w26.1: The Grand Demo** *(flagship — TOC name; full-page)* — interactive sim, the
  book's finale. Center canvas: the randomized Apartment, Rusty, the growing log-odds map (the
  Ch. 13 grayscale: white where confident-free, dark where occupied, gray where unknown), green
  live scan, purple pose covariance ellipse, orange MPPI rollout fan. Left rail: mode state machine (D26.4) with the
  active mode lit, plus the three **chaos buttons — Kidnap, Walker, LiDAR dropout** — and
  seed/randomize + speed controls. Right rail, tabbed inspectors: **Belief** (pose ellipse
  history, particle cloud during `Relocalize`, NIS strip chart), **Graph** (pose-graph nodes/edges
  live, loop-closure edges flashing on acceptance, robust-kernel downweights shown), **Frontiers**
  (frontier cells + utility bars, chosen target starred), **MPPI** (rollout storm with
  cost-coloring, chosen control highlighted), **Timing** (per-task period/staleness bars against
  the F3 budget). Bottom: mission timeline — map entropy $H(m_t)$ falling, event pins (loop
  closures, mode switches, chaos injections). Reader manipulates: chaos buttons (headline
  interaction), seed, speed, ground-truth toggle, Ch. 25 calibrated-model toggle. Observes: the
  full sense→SLAM→frontier→plan→control loop and, on sabotage, detection statistics crossing
  thresholds and recovery playing out. **Misconception killed:** "autonomy demos are magic
  monoliths that hide their failures" — every internal is inspectable and the failures are on the
  panel, not in the gag reel. Autoplay: mission starts on load, default seed; static fallback: a
  six-frame SVG storyboard of one mission with annotated callouts.
- **Widget w26.2: Stack Anatomy** — interactive diagram (animation-grade). The block diagram of
  §5 with live message-rate counters on each arrow and staleness heat on each block (replaying a
  recorded mission trace, so it works without running the full sim). Click a block: its chapter
  badge, message types, and period pop up; toggle a block *off* to see degradation ripple (loop
  closure off → map shears; frontier off → Rusty idles when reachable unknowns remain).
  **Misconception killed:** architecture diagrams as decoration — here the arrows carry measured
  traffic and turning boxes off has visible consequences.
- **Widget w26.3: Failure Theater** — animation (three curated recorded missions, scrubbable).
  One tab per failure: *Kidnap* (teleport → fitness ratio $\rho_t$ dives → `Relocalize` → particle
  cloud condenses in the frozen map → resume), *Walker* (person crosses → novelty cells flagged
  green→red → local replan bulge → map unpoisoned because novelty measurements were withheld from
  mapping), *Dropout* (scans stop → watchdog fires → MPPI coasts on prediction with inflating
  margin (F2 visibly widening) → creep-to-stop → recovery on signal return). Timeline pins mark
  detection, mode switch, recovery-complete; the relevant statistic is plotted under the canvas.
  **Misconception killed:** "the filter always converges" / recovery as magic — each recovery is a
  detector crossing a threshold plus a mode with a plan.
- **Widget w26.4: Retrospective Scorecard** — static figure (SVG, two panels). Panel A: five
  real compile-error vignettes from the book's own history (frame mixup caught by the Ch. 3
  newtypes; an $H_t$ dimension mismatch caught by const generics; a non-exhaustive `Mode` match
  caught at compile time when `Recover` was added; a map double-mutation caught by the borrow
  checker; a non-`Send` RNG caught at the rayon boundary) each rendered as the actual rustc
  message with a one-line moral. Panel B: the cost column — borrow-checker friction in graph code
  (index-based `petgraph` workaround), compile times, WASM threadlessness, ecosystem gaps. Honest
  by construction: both panels, same size.

Dashboard layout sketch (w26.1): `[left rail 220px: mode/chaos/seed] [center: world canvas]
[right rail 300px: tab strip + inspector] / [bottom 140px: entropy timeline + event pins]`.
On narrow screens the rails collapse into a tab bar above the canvas.

## 5. Practical (P) — Rust Implementation

**Crates** (all already in the TOC stack; the capstone adds one):

- `nalgebra` 0.35, `parry2d` 0.30, `petgraph` 0.8, `pathfinding` 4.15, `faer` 0.24 (via the Ch. 15
  optimizer), `rand`/`rand_distr` 0.9/0.6 (seeded `SmallRng`) — inherited through the workspace
  crates being integrated.
- `crossbeam-channel` 0.5 — bounded MPMC channels for the native task bus (new dependency; the
  one concurrency crate the book adds, justified by `std::sync::mpsc` lacking multi-consumer).
- `egui`/`eframe` 0.35 + `egui_plot` 0.34 — the Grand Demo.
- `rerun` 0.26 (optional, native-only feature flag) — mission replay logging; the `.rrd` powers
  w26.2/w26.3 recordings and the optional viewer embed w26.5 (Mission Replay, see §6 manifest).

**Concurrency design (the chapter's key engineering content):** tasks implement one trait; on
native, each task runs on its own thread at its own rate over `crossbeam-channel`; on WASM (no
threads), a deterministic round-robin scheduler ticks due tasks inside the eframe update loop —
*the same task code, the same message types, the same seeds, bit-identical missions*. This
"portable by construction" property is what makes the failure tour reproducible in the reader's
browser and is called out as the book's closing structural argument.

**Module plan:** `crates/capstone/`

```
src/
  bus.rs         // Stamped<T>, FrameId, Msg, Bus (native: channels; wasm: queues)
  task.rs        // Task trait, native ThreadRunner, wasm RoundRobin scheduler
  tasks/
    slam.rs      // wraps ch16 RustSLAM-2D
    explore.rs   // wraps ch24 frontier detector + scorer
    plan.rs      // wraps ch20 A* + Dubins
    control.rs   // wraps ch23 Mppi + ch19 Esdf2
    supervisor.rs// Mode machine, detectors (F4), recovery behaviors
  mission.rs     // run_mission, MissionCfg, MissionReport
demos/ch26-grand-demo/   // w26.1 (full-page eframe app, repo-root demos/)
```

**Key types & signatures:**

```rust
use localize::GaussianBelief;  // Ch. 11's shared belief type: { mean: SE2, cov: Matrix3<f64> }

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct FrameId(pub &'static str); // "map", "odom", "base" — Ch. 3 newtype discipline

pub struct Stamped<T> { pub t: SimTime, pub frame: FrameId, pub v: T }

pub enum Msg {
    Scan(Stamped<sim::Scan>),          // Ch. 4 raw ranges; the slam task projects to ch16_slam2d::PointCloud
    Odom(Stamped<Twist2>),
    PoseBelief(Stamped<GaussianBelief>), // Ch. 11 localize type (mean: SE2, cov in tangent space)
    MapPatch(Stamped<OccGridPatch>),
    Frontiers(Stamped<Vec<ScoredFrontier>>),
    Path(Stamped<DubinsPath>),
    Cmd(Stamped<Cmd>),
    Event(StackEvent), // LoopClosure, KidnapSuspected, ScanTimeout, ...
}

/// One capstone subsystem. Identical impls run threaded (native) or cooperatively (wasm).
pub trait Task: Send {
    fn name(&self) -> &'static str;
    fn rate_hz(&self) -> f64;
    fn tick(&mut self, bus: &mut Bus, now: SimTime);
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Mode { Explore, Navigate { goal: FrontierId }, Relocalize, Recover(RecoverKind), Done }

pub struct Supervisor {
    mode: Mode,
    kidnap: DualEmaDetector,   // rho = w_fast / w_slow over ICP fitness (F4a)
    nis_gate: ChiSquareGate,   // F4b
    scan_watchdog: Watchdog,   // F4c
}

pub struct MissionCfg { pub seed: u64, pub delta: f64, pub rates: RateTable, pub chaos: Vec<ChaosEvent> }
pub struct MissionReport {
    pub coverage: f64,           // fraction of reachable cells known
    pub traj_rmse: f64,          // vs ground truth (simulator-only luxury)
    pub loop_closures: usize,
    pub events: Vec<(SimTime, StackEvent)>,
    pub entropy_curve: Vec<(SimTime, f64)>,
}
pub fn run_mission(cfg: &MissionCfg) -> MissionReport;
```

**Worked end-to-end example** (`cargo run --example mission -- --seed 42 --kidnap-at 35`):
prints the event timeline (`t=35.0 KidnapInjected`, `t=36.4 KidnapSuspected rho=0.31`,
`t=41.8 RelocalizeConverged ess=0.87`, ...), then the report: coverage 97.4%, trajectory RMSE
0.11 m, 4 loop closures, mission time 142 s (numbers are design targets; the implementer records
actuals and a unit test then pins them under the fixed seed). With `--features rerun` the same run
writes `mission-42.rrd`. The no-chaos seed-42 mission is the book's canonical regression test:
CI fails if coverage or RMSE regresses.

**Runnable artifact:** `cargo run --example mission` (native, with optional rerun replay);
the WASM Grand Demo is the same `run_mission` machinery stepped by the round-robin scheduler with
every internal streamed to the inspector panels. The book's last page states the claim plainly:
*everything you studied is running here, unmodified, at real time.*

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w26.1 | The Grand Demo | wasm-sim (full-page) | capstone + sim + ch12/13/15/16/19/20/23/24 crates, eframe, egui_plot | chaos buttons, seed, speed, inspector tabs, model toggle | the whole stack, inspectable; detection + recovery are mechanisms, not luck |
| w26.2 | Stack Anatomy | wasm-sim (trace replay) | capstone bus trace + eframe | click blocks, toggle blocks off | architecture as measured dataflow; degradation is causal |
| w26.3 | Failure Theater | animation (recorded, scrubbable) | recorded missions + eframe | tab per failure, time scrub | F1's assumption↔failure table, one row per tab |
| w26.4 | Retrospective Scorecard | static-svg | plotters + hand-set rustc snippets | none | what the type system caught vs. what it cost |
| w26.5 | Mission Replay | rerun-embed (optional) | rerun 0.26 iframe, pinned `.rrd` | rerun viewer controls | the same mission in a production replay tool |

## 7. Exercises & Extensions

1. **(F)** Complete derivation F2 for a path (not a point): show why per-point chance constraints
   at level $\delta$ do not give a path-level guarantee at $\delta$, and derive the Bonferroni
   correction for $n$ waypoints.
2. **(F)** Write the mission of D26.2 as a formal POMDP tuple, then identify — one sentence each —
   the approximation every capstone task makes (F1's table, from memory). Which single assumption
   would you spend research effort removing first, and why?
3. **(C, predict-then-verify)** Using the F3 budget and the Timing panel's numbers, predict the
   walker speed at which the Grand Demo starts clipping the walker. Verify with the Walker chaos
   button at increasing speeds. Which term in F3 dominated?
4. **(C)** Find a randomization seed where `Relocalize` converges to the *wrong* room and explain
   the symmetry that caused it (Ch. 12's kidnapped-robot ambiguity at apartment scale). What
   additional sensor or behavior would break the symmetry?
5. **(P)** Add a `ReturnHome` mode: after `Done`, plan and drive back to the start pose. Requires
   touching only `supervisor.rs` and `mission.rs` — verify the compiler forces you to handle the
   new mode in every `match` (w26.4's third vignette, experienced firsthand).
6. **(P, stretch)** Port `ControlTask` to a `ros2-rust` node exchanging `geometry_msgs` with the
   rest of the stack running in simulation — the chapter's "where to go next" section gives the
   scaffold and names the crates (`rclrs`; `r2r` as the alternate binding).

## 8. Modernization Notes

- **What the baseline lacked entirely:** the 1999–2000 draft (and the 2005 edition) has no
  integration chapter — the reader finishes knowing filters, maps, and planners but never sees
  them composed with rates, interfaces, and failure handling. The closest baseline material is
  the draft's "Projects" prompts (Ch. 10.6, 14.9, 16.8) and the implicit systems lore in its
  museum-robot anecdotes. This chapter makes that lore explicit and executable: the architecture
  is the modern SLAM Toolbox + Nav2 shape (per the modernization research), taught by building it.
- **Deliberately reused rather than re-derived:** every algorithm here is a citation into
  Chs. 9–24; the only new formal content is composition math (F1–F4). This is by design — the
  capstone must prove the book's artifacts compose, not introduce a 27th algorithm.
- **Dropped from the baseline and noted as pointers:** multi-robot SLAM (draft Ch. 12.10, 14.5) —
  out of scope for a single-browser demo, pointer to the literature; 3D mapping (draft Ch. 14.6)
  — pointer to Ch. 19's TSDF showpiece and the sophus/3D ecosystem; manipulation — never in
  either book's scope.
- **Honesty items the prose must keep:** (1) the simulator grades its own homework — trajectory
  RMSE against ground truth exists only because we own the world; the retrospective must say what
  evaluation looks like on hardware (held-out maps, loop-closure precision/recall, no ground
  truth). (2) WASM single-threading means the browser demo is cooperative, not preemptive — the
  identical-semantics claim holds because the scheduler is deterministic, and the text must not
  imply the browser proves real-time *scheduling*, only real-time *throughput*. (3) The
  engineering retrospective reports costs with the same care as wins (w26.4 panels equal-sized;
  compile-time and ecosystem-gap numbers measured, not vibed). (4) `ros2-rust`/hardware/3D
  pointers are dated August 2026 and flagged as such — ecosystem sections rot fastest.
