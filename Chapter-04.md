# Chapter 4 — Rusty, Sensors, and the Simulator

> Part I — Foundations: The Robot and Its Uncertainty · Estimated length: 8 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

This chapter builds the laboratory the rest of the book lives in: Rusty the differential-drive
rover, the Hallway and Apartment worlds, wheel encoders, a ray-cast 2D LiDAR, and — just as
important — the `eframe` widget framework that every later chapter's demos reuse. Two ideas carry
the chapter. First, the pedagogical one: **the simulator is the world; models are beliefs about
the world** — this chapter implements *generative* physics (what actually happens), while Chs. 9
and 10 will fit *inference* models to it, and the two are deliberately not the same equations, so
model mismatch is real from day one. Second, the engineering one: **determinism is a feature** —
(seed, controls) → bit-identical runs, native and WASM, which is what makes every figure
reproducible and every benchmark honest. The "aha": after ten minutes of driving Rusty with arrow
keys, the reader has *felt* odometry drift and LiDAR noise, and owns the instrument that will
measure every algorithm in the book. The Practical section is the book's architectural contract:
workspace layout, crate boundaries, and widget chrome are fixed here and never renegotiated.

Story line:

1. **Hook** — w4.1 autoplay: Rusty drives a scripted loop; the encoder-integrated ghost (orange)
   slides off the ground truth (gray dashed). Then the reader takes the wheel (arrow keys) and
   makes it worse.
2. **Kinematics** — wheels to twist to pose: diff-drive model, nonholonomic constraint, exact arc
   integration = `SE2::exp` (the Ch. 3 payoff, cashed).
3. **Odometry** — encoder ticks to pose increments; systematic vs. stochastic error; drift
   measured, not asserted.
4. **Sensors** — a working taxonomy grounded in real devices (encoders, IMU, sonar, LiDAR,
   cameras); what each actually returns; LiDAR physics down to the beam.
5. **The lab** — the Hallway and Apartment worlds; parry2d ray casting; the sim's noise models
   and RNG-stream discipline.
6. **Noise phenomenology** — measure the simulator like an experimentalist: odometry error growth
   curves, LiDAR error histograms (w4.4, w4.2).
7. **Anatomy of a book widget** — how an `eframe` app becomes an iframe on the page; the chrome
   contract (autoplay, seed, colors, fallback).
8. **Experiment** — the figure-eight run: deterministic replay, drift statistics, and the
   chapter's regression-locked worked example.

## 2. Prerequisites & Position

- **Builds on:** Ch. 1 (workspace cloned, Rusty introduced), Ch. 2 (`prob::rng` seeding
  discipline, Gaussians for noise), Ch. 3 (`SE2`, `exp/log`, ⊞ — exact-arc integration and
  odometry composition are direct applications).
- **Feeds into:** every later chapter. Specifically: Ch. 5 (Hallway world hosts the Bayes filter),
  Ch. 9 (motion models are *fit to* this simulator's slip noise; the α-parameters meet a world
  that doesn't share their parametrization), Ch. 10 (beam-model intrinsics learned from `Lidar`
  data), Ch. 12 (the `Log` format powers the EKF-vs-grid-vs-MCL benchmark and the book's public
  MCL demo), Ch. 13 (maps built from `Scan`s), Ch. 16/24/26 (the Apartment is the arena),
  Appendix D (this chapter's architecture, as reference).
- **Baseline sources:** Lynch & Park Ch. 13 §13.3 (nonholonomic wheeled robots: unicycle and
  diff-drive models), §13.4 (odometry via exponential coordinates — our Derivation 3 is their
  method on our `SE2`); Spong et al. Ch. 14 §14.1 (Pfaffian nonholonomic constraints), §14.3
  (kinematic models); Niku Ch. 10 §10.4 (encoders: optical, quadrature), §10.13 (range finders:
  ultrasonic and light-based time-of-flight), Ch. 9 §9.6–9.7 (DC motors/PWM — one context box);
  Thrun et al. Ch. 5 §5.2.1 (kinematic configuration — preview only; the probabilistic models
  wait for Ch. 9); Craig Ch. 8 §8.8 (position sensing, cross-reference).

## 3. Foundation (F) — Mathematical Core

Chapter notation table:

| Symbol | Meaning |
|---|---|
| $r$, $\ell$ | wheel radius, track width (wheel separation) |
| $\omega_L, \omega_R$ | left/right wheel angular velocities |
| $u_t = (v, \omega)^\top$ | commanded body twist (forward, yaw rate) |
| $\Delta s_L, \Delta s_R$ | wheel arc lengths over one tick interval |
| $N_{\mathrm{tpr}}$ | encoder ticks per wheel revolution |
| $z_t^k$, $\varphi_k$ | LiDAR range of beam $k$, beam bearing (body frame) |
| $z_{\max}$, $\sigma_r$ | max range, range noise std |
| $m$ | the world/map (here: ground-truth line segments) |

**Definitions:** differential-drive robot; proprioceptive vs. exteroceptive, active vs. passive
sensors (taxonomy applied to encoders/IMU/sonar/LiDAR/camera, each grounded in its Niku device
section); the nonholonomic no-slip constraint $\dot{x}\sin\theta - \dot{y}\cos\theta = 0$
(Pfaffian, Spong §14.1) — Rusty cannot move sideways, which is exactly why parking is a planning
problem in Ch. 20; simulator tick (fixed $\Delta t = 0.02$ s); reproducible run: the map
$(\text{seed}, \text{control script}) \mapsto \text{trajectory}$ is a pure function; RNG streams
(one independent `SmallRng` per noise source, so adding a sensor never perturbs existing replays).

**Derivations:**

1. **Wheels to twist.** *Statement:* $v = \tfrac{r}{2}(\omega_R + \omega_L)$,
   $\omega = \tfrac{r}{\ell}(\omega_R - \omega_L)$. *Sketch (4 steps):* each wheel's contact
   speed; rigid-body constraint along the axle; solve the two linear equations; sanity limits
   (equal speeds → straight; opposite → spin in place). *Collapsible:* the inverse map (twist →
   wheel speeds) used by the keyboard teleop, and the ICR (instantaneous center of rotation)
   radius $R = v/\omega$.
2. **Exact arc integration is the exponential map.** *Statement:* under constant $(v, \omega)$
   over $\Delta t$, the pose update is exactly
   $x_{t+1} = x_t \boxplus (v\Delta t,\ 0,\ \omega\Delta t)^\top$ — no small-angle approximation.
   *Sketch (3 steps):* constant twist ⇒ the Ch. 3 screw ODE; Derivation 4 of Ch. 3 gives the
   chord $V(\theta)\rho$; identify with the constant-curvature arc of radius $v/\omega$.
   *Collapsible:* comparison against Euler and RK4 integration — the exact form is *simpler* and
   error-free, the chapter's best advertisement for Lie groups. (This is Lynch–Park §13.4's
   odometry formula, recognized as $\exp$.)
3. **Odometry from encoder ticks.** *Statement:* with $\Delta s_{L,R} = 2\pi r\,\Delta\text{ticks}_{L,R}/N_{\mathrm{tpr}}$,
   the dead-reckoning increment is $\boldsymbol{\tau}_{\mathrm{odo}} = (\Delta s, 0, \Delta\theta)^\top$,
   $\Delta s = \tfrac{1}{2}(\Delta s_L + \Delta s_R)$, $\Delta\theta = (\Delta s_R - \Delta s_L)/\ell$,
   applied as $\hat{x}_{t+1} = \hat{x}_t \boxplus \boldsymbol{\tau}_{\mathrm{odo}}$.
   *Sketch (4 steps):* ticks → arcs; arcs → chord/heading change (Derivation 1 integrated);
   quantization error bound $\pm \pi r / N_{\mathrm{tpr}}$ per wheel; separate *systematic*
   errors (miscalibrated $r$, $\ell$ — bias that no filter removes) from *stochastic* slip.
   *Collapsible:* error-propagation preview: variance of $\hat{x}_t$ grows without bound —
   computed empirically in w4.4, derived properly in Ch. 9.
4. **The simulator's LiDAR forward model.** *Statement:* beam $k$ returns
   $z_t^k = \min\!\big(z^{k*} + \varepsilon,\ z_{\max}\big)$, $\varepsilon \sim \mathcal{N}(0, \sigma_r^2)$,
   with dropout to $z_{\max}$ w.p. $p_{\mathrm{drop}}$, where
   $z^{k*} = \mathrm{raycast}(m, x_t \cdot T_{\mathrm{lidar}}, \varphi_k)$ is the true distance
   along the beam. *Sketch (3 steps):* beam pose from body pose ∘ extrinsic (a `Pose<Body,
   LidarF>` — Ch. 3's newtypes earning rent); first-hit distance via parry2d; additive noise +
   dropout. *Honesty note stated in text:* this is a **two-component** generative model (hit +
   max); Ch. 10's four-way mixture (hit/short/max/rand) is an *inference* model that will be fit
   to data from this simulator plus its dynamic-obstacle option — the gap between them is the
   lesson.

**Named algorithms** (book API names; Thrun's `sample_motion_model_velocity` etc. are explicitly
*deferred to Ch. 9* — this chapter is the world, not the belief):

- `diff_drive_step(pose, u, dt, slip_rng) -> pose'` — per-wheel multiplicative slip
  $\tilde\omega_i = \omega_i(1 + \epsilon_i)$, $\epsilon_i \sim \mathcal{N}(0, \sigma_{\mathrm{slip}}^2)$,
  then exact-arc integration; $O(1)$.
- `encoders_observe(robot) -> EncoderTicks` — integer quantization of wheel angles; $O(1)$.
- `odometry_delta(ticks, params) -> Tangent2` — Derivation 3; $O(1)$.
- `raycast_scan(world, pose, lidar) -> Scan` — $n_{\mathrm{beams}}$ ray casts against the world's
  BVH-accelerated polyline set: $O(n_{\mathrm{beams}} \log n_{\mathrm{segments}})$.
- `sim_step(sim, cmd) -> Frame` — one tick of the whole lab: actuate, integrate, sense; $O(\text{scan})$.

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor: *the flight simulator you get to open up.* Autoplay defaults, one primary
parameter each, book colors (ground truth gray dashed, dead-reckoning/prediction orange,
measurements green), static fallbacks in CI.

- **Widget w4.1: Rusty's Dashboard** (flagship) — type: interactive sim (the chapter's, and
  lab's, front door). Main pane: the Apartment, Rusty driven by arrow keys/WASD (autoplay: a
  scripted tour when idle/unfocused). Overlays: ground truth (gray dashed trail) and the
  encoder-integrated pose (orange trail + orange ghost robot). Side panel (egui): live
  `egui_plot` of $\|\hat{x}_t \boxminus x_t\|$ vs. distance traveled; tick counters; the seed
  field. Reader manipulates: driving, plus **one** slider — slip noise $\sigma_{\mathrm{slip}}$
  (encoder resolution behind an "advanced" disclosure). Observes: drift grows with distance,
  resets never; sharp turns hurt more than straights; zero noise → traces coincide (and the
  caption asks why real robots can't do that: systematic error, Derivation 3). Misconception
  killed: *odometry is a position sensor.* It is a velocity sensor integrated hopefully.
- **Widget w4.2: LiDAR Anatomy** (flagship) — type: interactive sim. A frozen scene (Rusty in a
  doorway); beams drawn one at a time on a timeline scrubber, or all 360 at once (toggle). Per
  beam: the cast ray, the true hit point (gray), the noisy return (green dot), and the growing
  range-vs-bearing plot beneath — the raw $z_t$ vector the algorithms will actually receive.
  Reader manipulates: scrub beam index; one slider $\sigma_r$; dropout toggle. Observes: the
  "wall" in range space is a fuzzy band; dropouts spike to $z_{\max}$; glass/black surfaces
  (marked segments with high $p_{\mathrm{drop}}$) punch holes in the scan. Misconception killed:
  *the LiDAR sees geometry* — it returns 360 noisy numbers, and everything after this is
  inference.
- **Widget w4.3: World Tour** — type: interactive sim (light). The Hallway and the Apartment
  side by side; click to teleport Rusty and see its local scan; hover shows world coordinates and
  the door/room labels the book's prose will refer to for 22 chapters. One control: world
  selector. Purpose: make the two arenas familiar landmarks, not diagrams. (Static fallback: the
  labeled floorplan SVG that also serves as the book's map figure everywhere.)
- **Widget w4.4: Seed Lab** — type: interactive sim. The same 5 s command script run in $N = 50$
  seeded parallel universes; trajectory fan (thin orange traces) over ground-truth-with-zero-
  noise (gray dashed); an `egui_plot` pane shows cross-seed position spread vs. time (the
  empirical √-ish growth). Reader manipulates: seed-set reroll; one slider $\sigma_{\mathrm{slip}}$.
  Observes: the fan is a *distribution over futures* — the banana of w3.4 now grown from real
  actuation noise. Misconception killed: *one simulation run tells you what will happen.* This
  widget is the reason every benchmark in the book reports over many seeds.
- **Widget w4.5: Anatomy of a Book Widget** — type: static-svg (annotated screenshot). The w4.1
  frame with callouts naming the chrome contract: title bar, autoplay/pause, speed, the visible
  seed + reroll, color legend, fullscreen/source links, and the `<noscript>` fallback path.
  Doubles as Appendix D's cover figure. Teaches the reader (and future chapter authors) the
  contract every widget obeys.

Dashboard layout sketch (w4.1, the pattern all later chapter dashboards copy): world canvas left
(≥ 70% width, aspect-locked), egui side panel right with — top to bottom — playback chrome, the
one primary slider, disclosure for advanced params, live plots, seed row. On narrow/mobile
layouts the panel drops below the canvas.

## 5. Practical (P) — Rust Implementation

**This section is the book's architectural contract.** Later chapters add crates and modules but
never restructure what is fixed here.

- **Crates:** `parry2d` 0.30 (f64 build — the `parry2d-f64` variant — for BVH-accelerated ray
  casting and point/shape distance queries; wasm-proven); `nalgebra` 0.35 (shared math);
  `rand` 0.9 + `rand_distr` 0.6 (seeded `SmallRng` streams; no `getrandom` needed on WASM);
  `pr-core` (Ch. 2 `prob`, Ch. 3 `geom`); `eframe`/`egui` 0.35 + `egui_plot` 0.34 (widget
  chrome + live telemetry; WebGPU with WebGL2 fallback via `WebRunner`); `serde` 1 + `postcard`
  (compact `Log` serialization); `plotters` (static fallbacks); `trunk` 0.21 builds each demo bin
  to a Pages-deployable iframe. WASM rules (from the research, enforced in CI): no `rayon`
  features anywhere in wasm builds; seeded RNGs only.

**Workspace layout (the contract):**

```text
crates/
  pr-core/        # accumulating core library: prob (Ch.2), geom (Ch.3)
  <later crates>  # added as the book proceeds: bayes_core (Ch.5), motion (Ch.9), sensor (Ch.10), localize (Chs.11–12), chNN_* algorithm crates
  sim/            # THIS CHAPTER: worlds, robot, sensors, logs — depends on pr-core + parry2d
  widget-kit/     # THIS CHAPTER: eframe chrome every demo reuses — depends on sim + pr-core + egui
  figures/        # plotters generators for static SVGs (run in CI)
demos/            # repo root — one thin crate per chapter (crate name underscored: ch04_lab)
  chNN-<slug>/    # one [[bin]] per widget (w4-1-dashboard, …)
book/             # mdBook 0.5 source; demos embedded as lazy iframes at /demos/chNN/<bin>/
```

- **Module plan:** `crates/sim/src/{world.rs, robot.rs, encoders.rs, lidar.rs, run.rs, script.rs}`;
  `crates/widget-kit/src/{app.rs, chrome.rs, view.rs, style.rs, capture.rs}`;
  `demos/ch04-lab/` (bins `w4-1-dashboard`, `w4-2-lidar-anatomy`, `w4-3-world-tour`,
  `w4-4-seed-lab`).

Key types & signatures (the API later chapters program against):

```rust
// crates/sim/src/world.rs
pub struct World {
    segments: Vec<Wall>,            // Wall { seg: parry2d::shape::Segment, material: Material }
    bvh: parry2d::partitioning::Qbvh<u32>,
    pub bounds: parry2d::bounding_volume::Aabb,
}
pub enum Material { Wall, Glass /* high dropout */, Door(u8) }
impl World {
    pub fn hallway() -> Self;              // 1D-ish corridor, 3 labeled doors (Ch. 1/5 world)
    pub fn apartment() -> Self;            // the book's 2D floorplan (labeled rooms)
    pub fn raycast(&self, origin: Point2<f64>, angle: f64, max: f64) -> Hit; // BVH first-hit
    pub fn collides(&self, pose: &SE2, radius: f64) -> bool;
    pub fn distance_to_nearest(&self, p: Point2<f64>) -> f64; // Ch. 10 likelihood fields will want this
}

// crates/sim/src/robot.rs
pub struct RobotParams {
    pub wheel_radius: f64,   // r  = 0.033 m
    pub track: f64,          // ℓ  = 0.16 m
    pub body_radius: f64,    //     0.11 m
    pub ticks_per_rev: u32,  // N  = 4096
    pub slip_std: f64,       // σ_slip = 0.02 (multiplicative, per wheel) — NOT Thrun's α's, on purpose
}
pub struct Robot { pub pose: SE2, wheel_angles: [f64; 2], params: RobotParams, rng: SmallRng }
impl Robot {
    /// Actuate with slip noise, integrate exactly (SE2::exp), stop on collision.
    pub fn step(&mut self, cmd: Twist, dt: f64, world: &World) -> StepOutcome;
}
pub struct Twist { pub v: f64, pub omega: f64 }

// crates/sim/src/encoders.rs
pub struct EncoderTicks { pub left: i64, pub right: i64 }  // cumulative
pub fn odometry_delta(prev: EncoderTicks, cur: EncoderTicks, p: &RobotParams) -> Tangent2;

// crates/sim/src/lidar.rs
pub struct LidarParams { pub n_beams: usize /*360*/, pub fov: f64 /*2π*/, pub max_range: f64 /*8.0*/,
                         pub sigma_r: f64 /*0.02*/, pub p_dropout: f64 /*0.01*/,
                         pub extrinsic: Pose<Body, LidarF> }
pub struct Lidar { params: LidarParams, rng: SmallRng }
impl Lidar { pub fn scan(&mut self, world: &World, body: &SE2) -> Scan; }
pub struct Scan { pub ranges: Vec<f64>, pub t: f64 }       // angles implicit: k → φ_k
impl Scan { pub fn bearing(&self, k: usize) -> f64; pub fn points_body(&self) -> Vec<Point2<f64>>; }

// crates/sim/src/run.rs — determinism and logging live here
pub struct SimConfig { pub world: WorldId, pub robot: RobotParams, pub lidar: LidarParams,
                       pub seed: u64, pub dt: f64 /*0.02*/, pub scan_every: u32 /*5 ticks*/ }
pub struct Sim { /* World + Robot + Lidar + Encoders; RNG streams split from seed */ }
impl Sim {
    pub fn new(cfg: SimConfig) -> Self;    // seed → independent SmallRng streams per subsystem
    pub fn step(&mut self, cmd: Twist) -> Frame;
}
/// What every estimator in the book consumes. Serializable (serde + postcard).
pub struct Frame { pub t: f64, pub truth: SE2, pub ticks: EncoderTicks, pub scan: Option<Scan> }
pub struct Log { pub cfg: SimConfig, pub cmds: Vec<Twist>, pub frames: Vec<Frame> }
impl Log { pub fn record(cfg: SimConfig, script: &dyn Script) -> Log;
           pub fn save(&self, path: &Path) -> io::Result<()>; pub fn load(...) -> ...; }
// Chs. 11/12 benchmark on *identical* saved Logs — that promise is made (and tested) here.

// crates/widget-kit/src/app.rs — the skeleton EVERY chapter demo implements
pub trait BookWidget {
    fn title(&self) -> &'static str;
    fn reset(&mut self, seed: u64);               // full rebuild from seed — powers reroll & replay
    fn tick(&mut self, dt: f64);                  // fixed-step sim advance (chrome calls it; autoplay)
    fn draw(&mut self, view: &mut WorldView);     // world-space rendering
    fn panel(&mut self, ui: &mut egui::Ui);       // the chapter's sliders/readouts
}
/// Wraps a BookWidget in the standard chrome: autoplay-on-load, play/pause/speed, visible seed +
/// reroll, color-legend toggle, prefers-reduced-motion respected, fullscreen & view-source links.
pub fn run<W: BookWidget + 'static>(widget: W) -> eframe::Result<()>;
// native: eframe::run_native · wasm: eframe::WebRunner::start on canvas #book-widget (one per iframe)

// crates/widget-kit/src/view.rs
pub struct WorldView { /* camera, egui::Painter */ }
impl WorldView {
    pub fn draw_world(&mut self, w: &World);
    pub fn draw_robot(&mut self, pose: &SE2, role: Role);
    pub fn draw_trail(&mut self, poses: &[SE2], role: Role);
    pub fn draw_scan(&mut self, body: &SE2, scan: &Scan);
    pub fn draw_cov_ellipse(&mut self, mean: Point2<f64>, cov: &SMatrix<f64,2,2>, role: Role);
}
// crates/widget-kit/src/style.rs — the color code as a type: no widget can invent colors
pub enum Role { Prior, Prediction, Measurement, Posterior, Truth }
impl Role { pub fn color(self) -> egui::Color32; } // blue / orange / green / purple / gray-dashed

// crates/widget-kit/src/capture.rs — static-fallback discipline
// `--capture` flag: render the autoplay default state to SVG, exit. CI runs it for every bin and
// places the output at book/src/fallback/<widget-id>.svg, wired into the iframe's <noscript>.
```

**Worked end-to-end example** (`cargo run -p sim --example figure_eight`): Rusty runs a scripted
60 s figure-eight in the Apartment at seed `BOOK_SEED`. Prints: final ground truth pose, final
dead-reckoned pose, their gap $\|\hat{x} \boxminus x\|$ (≈ 0.9 m position / 0.3 rad with default
`slip_std = 0.02` — exact golden values recorded at implementation time), and cross-seed drift
stats over 100 seeds (mean/95%). Emits `figures/ch04_figure_eight.svg` (both trails, book
colors — also w4.1's static fallback). Tests: `worked_example_ch04` replays the log and asserts
the final truth pose **exactly** (bit-for-bit float equality — the determinism regression lock);
`log_roundtrip` asserts `Log::save → load → replay` reproduces every `Frame`;
`odometry_zero_noise` asserts that with `slip_std = 0` and infinite encoder resolution,
dead reckoning equals ground truth to 1e-12 (Derivations 2+3 agree with the integrator).

**Runnable artifact:** the example above; WASM demos w4.1–w4.4. `cargo run -p ch04_lab --bin
w4-1-dashboard` runs the dashboard natively; `trunk build` in the same crate produces the iframe
build — the chapter walks through this once, and no later chapter repeats the plumbing.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w4.1 | Rusty's Dashboard | wasm-sim | eframe 0.35 + egui_plot 0.34 + sim + widget-kit | arrow-key driving; σ_slip slider; seed reroll | odometry drifts without bound; commands ≠ pose |
| w4.2 | LiDAR Anatomy | wasm-sim | eframe + egui_plot + sim + widget-kit | beam scrubber; σ_r slider; dropout toggle | a scan is 360 noisy ranges, not geometry |
| w4.3 | World Tour | wasm-sim | eframe + sim + widget-kit | world selector; click-to-teleport | the Hallway & Apartment as named places |
| w4.4 | Seed Lab | wasm-sim | eframe + egui_plot + sim + widget-kit | seed-set reroll; σ_slip slider | runs are samples; distributions over futures |
| w4.5 | Anatomy of a Book Widget | static-svg | plotters + widget-kit capture (CI) | none | the widget chrome contract (Appendix D cover) |

## 7. Exercises & Extensions

1. **(F)** Derive the inverse kinematics $(v, \omega) \mapsto (\omega_L, \omega_R)$ and the ICR
   radius; show which twists a diff-drive cannot produce instantaneously and connect that to the
   Pfaffian constraint.
2. **(F)** Prove Derivation 2's claim by computing the chord of a constant-curvature arc of
   length $v\Delta t$ and curvature $\omega/v$ and matching it to $V(\omega\Delta t)\,(v\Delta t, 0)^\top$.
   Then bound the error of Euler integration over one tick and over 60 s of figure-eight.
3. **(C — predict, then verify with w4.4)** If slip noise afflicted only the *right* wheel,
   predict the shape and skew of the trajectory fan. Verify (the widget has this as a preset
   under advanced). Explain the systematic-vs-stochastic distinction using what you see.
4. **(C — w4.2)** Predict the range-vs-bearing plot for Rusty facing a doorway 2 m ahead: where
   are the discontinuities, and which way do they jump when you add dropout? Verify beam-by-beam
   with the scrubber.
5. **(P)** Add a sonar to `sim`: cone aperture 15°, return = nearest hit *within the cone* plus
   Gaussian noise; expose it in a copy of w4.2. One paragraph: why sonar walls "smear" where
   LiDAR walls don't. (Starter branch provided; the type goes next to `Lidar` under the same
   `scan`-style API.)
6. **(P)** Implement `Log` replay determinism across platforms: record on native, verify the
   asserted hash of all `Frame`s in the WASM build (the repo's CI does this — make your fork's
   pass). Then break it on purpose by sharing one RNG stream between slip and LiDAR, and write
   two sentences on why the streams are separate.

## 8. Modernization Notes

- **What the baselines lack:** all sensor/actuator baselines (Niku Chs. 9–10, Craig §8.8) are
  device catalogs with no noise *models* and no code; Lynch–Park §13.3–13.4 and Spong Ch. 14 give
  exact deterministic kinematics but stop at dead reckoning; Thrun (1999–2000 draft) has no
  simulator, no standard world, and could not have had reproducible in-browser experiments. This
  chapter fuses them: device grounding (Niku) + exact geometry (Lynch–Park, via Ch. 3's `SE2`) +
  a seeded, instrumented lab none of them could ship.
- **Deliberate design decisions, recorded for future authors:** (i) the simulator's actuation
  noise is per-wheel multiplicative slip — intentionally *not* Thrun's $\alpha_1..\alpha_6$
  parametrization — so Ch. 9's models must be *fit* to a world that doesn't share their form
  (model mismatch as a feature, not an accident); (ii) the sim LiDAR is a two-component
  generative model, leaving short-returns/random noise to Ch. 10's dynamic-obstacle option, for
  the same reason; (iii) determinism is enforced by bit-exact regression tests, which constrains
  us to fixed-order summation and no parallelism inside `Sim` (rayon stays outside the sim core);
  (iv) f64 throughout (`parry2d-f64`) — consistency with the estimation stack beats the f32
  default's speed at this scale.
- **Dropped from the baselines and why:** actuator electronics (PWM, H-bridges, gear trains —
  Niku Ch. 9) beyond one context box; resolvers/LVDTs and the long sensor tail (Niku Ch. 10);
  the machine-vision pipeline (Niku Ch. 11 — cameras enter as projection factors in Ch. 18);
  omnidirectional/mecanum and car-like models (Lynch–Park §13.2, Reeds–Shepp — Dubins returns in
  Ch. 20 for planning); dynamics/contact forces (`rapier2d` is reserved for the rare chapter
  where contact matters; the lab is kinematic + noise by design).
- **Added beyond all baselines:** the IMU is *named* in the taxonomy with its error model
  sketched (bias + random walk) but implementation is deferred to Ch. 18's preintegration — the
  honest scope cut is stated in-chapter rather than silently omitted.
