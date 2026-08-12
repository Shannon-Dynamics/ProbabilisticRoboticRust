# Probabilistic Robotics via Rust

**An interactive web book — Foundation, Conceptual, Practical (FCP)**

> Every algorithm derived rigorously, every concept made visible and manipulable in the browser,
> every method implemented in production-quality Rust — and the *same* Rust code that appears in
> the text compiles to WebAssembly and powers the interactive simulations on the page.

---

## 1. Vision & Positioning

Thrun, Burgard, and Fox's *Probabilistic Robotics* defined how a generation of roboticists thinks
about uncertainty. Twenty years later the field's back-end has moved to factor graphs, its filters
live on Lie groups, its maps are TSDFs as often as occupancy grids — and no book teaches any of it
in Rust, the language the robotics industry is steadily adopting for safety-critical autonomy.
Research confirms the niche is open: there is no interactive book, in any language, that teaches
probabilistic robotics with in-page simulations, and no serious "robotics algorithms in Rust" text
at all.

This book fills that gap with three promises:

1. **Foundation (F)** — every concept gets its full mathematical formalism: definitions, stated
   assumptions, and derivations done honestly (collapsible for skimmers, complete for graduate
   readers), in the notation lineage of Thrun et al. extended to modern on-manifold estimation.
2. **Conceptual (C)** — every hard idea gets a visual, interactive treatment: dashboards, animations,
   and web simulations where the reader turns noise parameters, scrubs time, drags robots, and
   watches beliefs evolve. Interaction is invitation, not requirement — every widget autoplays a
   sensible default and degrades to a static figure without JS.
3. **Practical (P)** — every algorithm is implemented in idiomatic Rust with the best current crates,
   inside one cargo workspace where the book's printed listings, its test suite, its figures, and
   its in-page WASM demos are literally the same code.

**Unique structural advantage:** because Rust compiles to WASM, the filter the reader just studied
*is* the filter running in the page. No Python competitor can make that claim.

### The running lab

Two worlds thread through the whole book, hosting almost every demo:

- **The Hallway** — a 1D corridor with indistinguishable doors (Thrun's classic example): the
  pedagogical world where beliefs are plottable as curves.
- **The Apartment** — a 2D floorplan with simulated wheel encoders and a 2D LiDAR (ray-cast via
  `parry2d`): the realistic world where localization, mapping, SLAM, planning, and the capstone all
  take place.

The book's robot is **Rusty**, a differential-drive rover. Rusty is built in Chapter 4 and never
leaves; by Chapter 26 it autonomously explores and maps an apartment it has never seen.

### Reader path per chapter (fixed rhythm)

Hook → **C**onceptual (play first) → **F**oundation (now the rigor) → **P**ractical (Rust) →
Integration lab (run it in the running worlds) → Exercises. Concrete before abstract, always;
theorem-proof ordering is used only *inside* Foundation sections.

---

## 2. Book-wide Conventions

### Notation (Thrun-compatible core + modern manifold extensions)

| Symbol | Meaning |
|---|---|
| $x_t$, $u_t$, $z_t$, $m$ | state, control, measurement, map |
| $x_{0:t}$, $u_{1:t}$, $z_{1:t}$ | trajectories / histories |
| $bel(x_t) = p(x_t \mid z_{1:t}, u_{1:t})$, $\overline{bel}(x_t)$ | belief, predicted belief |
| $\eta$ | generic normalizer |
| $p(x_t \mid x_{t-1}, u_t)$, $p(z_t \mid x_t)$ | motion model, measurement model |
| $\mu_t, \Sigma_t$ / $\xi_t, \Omega_t$ | Gaussian moments form / canonical (information) form |
| $R_t$, $Q_t$ | motion / measurement noise covariance |
| $G_t$, $H_t$, $K_t$, $S_t$ | motion Jacobian, measurement Jacobian, Kalman gain, innovation covariance |
| $\mathcal{X}_t = \{x_t^{[i]}, w_t^{[i]}\}_{i=1}^{M}$ | weighted particle set |
| $\ell_{t,i}$ | log odds of cell $i$ |
| $c_t$, $f_t^i = (r, \phi, s)^\top$ | correspondence variable; feature (range, bearing, signature) |
| $\alpha_1 \ldots \alpha_6$ | motion-model noise parameters |
| $T \in SE(2)/SE(3)$, $\exp / \log$, $\boxplus / \boxminus$ | pose as Lie-group element; retraction operators |
| $\pi$, $r(x,u)$, $\gamma$, $V$, $b$ | policy, reward, discount, value function, belief state |

A per-chapter notation table lists exactly the symbols that chapter introduces. KaTeX with one
global macro file (`katex-macros.txt`) keeps rendering consistent.

### The book color code (figures, equations, and code comments all agree)

| Role | Color |
|---|---|
| Prior / previous belief | **blue** |
| Prediction (after motion) | **orange** |
| Measurement / likelihood | **green** |
| Posterior (after update) | **purple** |
| Ground truth | **gray dashed** |

Equation terms are color-coded to match figure elements and code comments book-wide (the single
most-praised device in Kalman-filter pedagogy).

### Verified Rust crate stack (as of Aug 2026)

| Layer | Primary | Notes / alternates |
|---|---|---|
| Linear algebra (small, fixed-size) | `nalgebra` 0.35 | `SMatrix`/`SVector` for states, covariances, Jacobians; WASM-clean |
| Sparse linear algebra | `faer` 0.24 | sparse Cholesky/QR for graph SLAM; backend of factrs |
| Probability & sampling | `rand` 0.9, `rand_distr` 0.6, `statrs` 0.19 | seeded `SmallRng`/`Pcg64` everywhere for reproducible demos |
| Lie groups (3D) | `sophus` (pinned minor version) | SO(3)/SE(3) exp/log; we hand-roll SE(2) pedagogically first |
| Filters (KF/EKF/UKF/PF) | **hand-rolled on nalgebra** | that's the point of the book; `adskalman` as cross-check in tests |
| Nonlinear least squares | hand-rolled GN/LM (dense → sparse via faer) | `factrs` 0.3 for the production factor-graph chapter; `tiny-solver` alternate |
| Geometry / ray-cast LiDAR / collision | `parry2d` 0.30 | `rapier2d` only where contact dynamics matter |
| Graphs & search | `petgraph` 0.8, `pathfinding` 4.15 | pose graphs, A*/Dijkstra |
| Kinematics asides | `k`, `urdf-rs` (openrr sub-crates) | not the dormant openrr umbrella |
| ML (Ch. 25) | `candle` (or `burn`) | small nets for learned sensor models, differentiable filters |

The Rust stack above is what the book **teaches and prints**: every listing, every exercise, and
the reference implementation of every algorithm.

### Publishing architecture (web)

The book ships as a **React / Next.js** static site, implemented in `web/`.

| Layer | Choice | Notes |
|---|---|---|
| Framework | Next.js 16 (App Router) + React 19 | static export (`output: 'export'`) — every chapter prerendered |
| Book chrome | Fumadocs 16 (`fumadocs-ui`, `fumadocs-core`, `fumadocs-mdx`) | sidebar, TOC, search, dark mode; composed as a library so the design stays ours |
| Content | MDX per chapter in `web/content/chapters/` | React widgets embedded directly in the prose |
| Math | `remark-math` + `rehype-katex` 7 with `katex` 0.18.4 (pinned via `overrides`) | pre-rendered at build time; global macro table in `lib/katex-macros.ts` |
| Code | Shiki 4 (via Fumadocs `rehypeCode`) | Rust highlighting with titles, line highlights, copy button |
| Charts / dashboards | Nivo 0.99 (`line`, `bar`, `heatmap`, `scatterplot`, `network`) | themed from the book's CSS custom properties; SSR-rendered SVG |
| Simulations | Canvas 2D + a shared framework (`useSimulation`, `SimCanvas`, `WidgetFrame`) | fixed-timestep, seeded, DPR-correct, theme-aware |
| Styling | Tailwind v4 | design tokens in `app/global.css` |
| Fonts | Fraunces (display), Source Serif 4 (prose), IBM Plex Sans/Mono | self-hosted via `next/font` |

**The algorithms are implemented twice, on purpose.** The Rust in the text is the canonical,
teachable implementation. `web/lib/` holds a faithful TypeScript port that powers the in-page
simulations, so a reader can compare them line by line — and `web/lib/__checks__.ts` pins the port
with numerical invariants (`npm run check`; 23 checks currently pass). Where the book states a
worked numeric example, both implementations must reproduce it.

Toolchain: Node ≥ 20.9 (Next 16 requirement). `npm run build` type-checks and statically exports;
`npm run check` runs the numerical self-checks. Details and conventions in `CLAUDE.md` and
`web/AUTHORING.md`.

---

## 3. Table of Contents

Legend per chapter: **[F]** foundation depth · **[C]** flagship widgets · **[P]** what gets built ·
**Sources** baseline mapping. Full per-chapter designs live in `Chapter-01.md` … `Chapter-26.md`.

---

### Part I — Foundations: The Robot and Its Uncertainty

**1. The Robot That Doubts — Why Probabilistic Robotics?**
The five sources of uncertainty; a single best guess vs. a belief; the hallway thought experiment
run live; the probabilistic paradigm's implications; tour of the book, the FCP method, and Rusty.
*[C]* **w1.1 Hallway Belief Machine (preview)** — the book's thesis in one autoplay animation.
*[P]* Reader environment setup; the workspace skeleton; first `cargo run`.
*Sources: Thrun Ch. 1; pedagogy of Labbe/Ciechanowski.*

**2. Probability: The Language of Uncertainty**
Random variables, joint/conditional probability, Bayes rule (with background knowledge),
independence, expectation, covariance, entropy and information; the Gaussian in 1D and $n$D,
moments vs. canonical form; sampling as representation.
*[C]* **w2.1 Gaussian Playground** (drag $\mu$, stretch $\Sigma$, watch iso-ellipses ↔ eigenvectors);
**w2.2 Blob Multiplier** (Bayes rule as pointwise product, conjugacy live).
*[P]* `rand_distr`/`statrs`/`nalgebra`: sampling, densities, a `MultivariateNormal` you understand;
seeded reproducibility discipline.
*Sources: Thrun Ch. 2.2; Choset App. on statistics.*

**3. The Geometry of Motion: Poses, Frames, and Lie Groups**
Frames and rotation matrices; homogeneous transforms; SO(2)/SE(2) fully worked; SO(3)/SE(3),
quaternions, axis-angle; exponential/log maps, twists and screws; why "state ⊞ increment" needs
$\boxplus/\boxminus$; uncertainty on manifolds (preview of banana distributions).
*[C]* **w3.1 Frame Composer** (drag frames, compose transforms, see leading-superscript notation
come alive); **w3.2 Exp/Log Lens** (twist slider → screw motion animation).
*[P]* `nalgebra` `Isometry2/3`, `UnitQuaternion`; hand-rolled `SE2` type with `exp/log/adjoint`;
`sophus` for 3D; compile-time frame-safety via newtypes.
*Sources: Lynch & Park Ch. 3; Craig Ch. 2; Spong Ch. 2; Sola micro-Lie (modernization).*

**4. Rusty, Sensors, and the Simulator**
Differential-drive kinematics and wheel odometry from encoder ticks; sensor taxonomy grounded in
real devices (encoders, IMU, sonar, LiDAR physics, cameras); noise phenomenology measured from the
simulator; building the book's lab: the Hallway and Apartment worlds, ray-cast LiDAR via `parry2d`,
fixed-seed determinism; anatomy of a book widget (`eframe` + WASM).
*[C]* **w4.1 Rusty's Dashboard** (drive Rusty with arrow keys; watch encoders drift from ground
truth); **w4.2 LiDAR Anatomy** (beam-by-beam ray casting with noise injection).
*[P]* `sim` crate: `World`, `Robot`, `Lidar`, `Encoders`; the widget framework skeleton every
later chapter reuses.
*Sources: Niku Chs. on sensors/actuators; Lynch & Park Ch. 13.3; Spong mobile-robot models.*

---

### Part II — The Bayes Filter Family

**5. The Bayes Filter: Recursive State Estimation**
States, completeness, and the Markov assumption; controls vs. measurements as two data streams;
beliefs; the Bayes filter recursion derived by induction; "sensing sharpens, moving smears";
representation/computation trade-offs — the family tree of every filter to come.
*[C]* **w5.1 Hallway Belief Machine** (full version: scrub time, toggle sense/move, break the
Markov assumption on purpose).
*[P]* `trait BayesFilter { fn predict(&mut self, u); fn correct(&mut self, z); }` — the trait the
whole book implements; discrete hallway filter as first impl; unit test reproduces the chapter's
3-step numeric worked example.
*Sources: Thrun Ch. 2.*

**6. Kalman Filters: The Linear-Gaussian World**
Linear-Gaussian systems; the KF derived by completing the square; the Kalman gain as precision-
weighted trust; the information filter and canonical form duality; matrix inversion lemma;
smoothing (RTS) as the first taste of "whole-trajectory" estimation.
*[C]* **w6.1 Kalman Tuning Bench** (1D cart: sliders for $R, Q$; color-coded prior/prediction/
measurement/posterior; divergence on purpose); **w6.2 Moments vs. Information** (same belief, two
parameterizations, cost of each operation).
*[P]* `Kf<const N, const U, const M>` on nalgebra `SMatrix`; RTS smoother; cross-validation vs `adskalman`;
1D + 2D tracking labs.
*Sources: Thrun Ch. 3.1–3.4; Barfoot (modern presentation).*

**7. Beyond Linearity: EKF, UKF, and Estimation on Manifolds**
Taylor linearization and its lies; the EKF; the unscented transform and UKF; where 2005 stops and
2026 begins: error-state formulation, $\boxplus/\boxminus$ retractions, EKF/UKF on SE(2)/SE(3),
invariant-EKF intuition; when each filter breaks, demonstrated.
*[C]* **w7.1 Linearization Lens** (nonlinear function + Gaussian in → true pushforward vs. EKF vs.
UKF sigma points, error live as you slide the operating point); **w7.2 Manifold vs. Vector**
(heading wrap-around catastrophe, fixed by on-manifold update).
*[P]* generic `Ekf`/`Ukf` over a `Manifold` trait; error-state EKF on our `SE2`; deliberate
compile-error showing the type system rejecting a dimension mismatch.
*Sources: Thrun Ch. 3.3; UKF & on-manifold from modernization set (Sola, Barfoot).*

**8. Nonparametric Filters: Histograms and Particles**
Discrete Bayes filter; grid decompositions; binary static-state filter in log odds (the occupancy
seed); importance sampling from first principles; the particle filter; resampling variance, low-
variance/systematic resampling, particle deprivation; KLD-adaptive sample sizes.
*[C]* **w8.1 Particle Survival Arena** (watch weights, then resampling kill/clone particles);
**w8.2 Resampling Wheel** (roulette vs. low-variance comb, side by side, variance measured live).
*[P]* `ParticleFilter<S>` with pluggable proposal/likelihood; low-variance resampler in 15 lines;
rayon-parallel weights (native) with the same code single-threaded on WASM.
*Sources: Thrun Ch. 4; KLD-sampling (2005 ed. material) from modernization set.*

---

### Part III — Probabilistic Models

**9. Probabilistic Motion Models**
Probabilistic kinematics; the velocity model (closed form + sampler, $\alpha_1..\alpha_6$) derived
from arc geometry; the odometry model ($\delta_{rot1}, \delta_{trans}, \delta_{rot2}$); the banana
distribution as truth vs. its Gaussian caricature; map-conditioned motion; on-manifold noise
(exponential-coordinate perturbations) as the modern formulation.
*[C]* **w9.1 Banana Machine** (drive a command, sample 1000 futures; sliders for each $\alpha$;
Gaussian overlay shows exactly what EKFs get wrong).
*[P]* `motion` module: both models, both variants, property-tested (samples ↔ closed form via KS
test); the samplers that power every localization demo to come.
*Sources: Thrun Ch. 5; Lynch & Park Ch. 13 (deterministic substrate).*

**10. Probabilistic Sensor Models**
The beam model as a four-way mixture (hit/short/max/rand) with EM fitting of intrinsics;
likelihood fields (and their smoothness advantage); map correlation; feature/landmark models with
range-bearing-signature and the correspondence variable $c_t$; over-confidence from independence
assumptions and its mitigations.
*[C]* **w10.1 Beam Mixture Mixer** (four sliders reshape the density against real simulated beams);
**w10.2 Likelihood Field Explorer** (drag Rusty; per-beam likelihood and the field as heatmap).
*[P]* `sensor` module: `BeamModel` (with `learn_intrinsics`), `LikelihoodField` (precomputed
distance transform), `LandmarkModel`; benchmarked against each other in the Apartment.
*Sources: Thrun Ch. 6; Niku vision/sensing chapters ground the device physics.*

---

### Part IV — Localization

**11. Localization I: Tracking with Gaussians**
The localization taxonomy (tracking / global / kidnapped; static / dynamic; passive / active);
Markov localization as the Bayes filter with a map; EKF localization with known then unknown
correspondences; maximum-likelihood data association, Mahalanobis gating, validation regions;
multi-hypothesis tracking as a Gaussian-mixture stopgap.
*[C]* **w11.1 Association Gate** (drag a measurement around a landmark field; gates as ellipses;
watch a wrong association poison the filter); **w11.2 EKF Localization Lab** (landmark world,
uncertainty ellipses breathing as landmarks come in view).
*[P]* `EkfLocalizer` over the Ch. 7 error-state EKF + Ch. 9/10 models; MHT with pruning as an
optional feature flag.
*Sources: Thrun Ch. 7.*

**12. Localization II: Global Localization with Grids and Particles**
Grid localization (resolution/precision trade, likelihood pre-caching); Monte Carlo Localization
as the particle filter instantiated; Augmented MCL ($w_{fast}/w_{slow}$ random-particle recovery)
solving the kidnapped-robot problem; dual/mixture proposals; localization among moving people
(measurement novelty filtering); the great comparison table, reproduced experimentally.
*[C]* **w12.1 MCL Theater** (the book's centerpiece: global localization in the Apartment —
particle cloud condenses from everywhere to one hypothesis; kidnap button; symmetry ambiguity).
*[P]* `Mcl` + `AugmentedMcl` (AMCL — still Nav2's default localizer — built by hand); side-by-side
benchmark: EKF vs grid vs MCL on identical logs; this chapter's demo is the book's public demo.
*Sources: Thrun Ch. 8.*

---

### Part V — Mapping and SLAM

**13. Occupancy Grid Mapping**
Mapping with known poses; per-cell binary Bayes filters in log odds; inverse sensor models
(hand-crafted, then learned from the forward model); multi-sensor fusion; where per-cell
independence lies to you, and MAP mapping with forward models as the honest alternative.
*[C]* **w13.1 Map Weaver** (drive Rusty, watch the map grow in log odds; gray = ignorance);
**w13.2 Independence Trap** (the conflict-in-doorways artifact, then MAP repair).
*[P]* `OccGrid` with the Ch. 8 binary filter per cell; Bresenham ray updates; live map streaming
into egui at 60 fps.
*Sources: Thrun Ch. 9.*

**14. The SLAM Problem and EKF SLAM**
Online vs. full SLAM; the joint pose-map state; why correlations *are* the map's value; EKF SLAM
with known then unknown correspondence; provisional landmarks and map management; the two fatal
flaws — quadratic cost and linearization-locked inconsistency — that motivate everything after.
*[C]* **w14.1 Correlation Web** (landmarks linked by correlation strength; loop closure snaps the
whole web tight — with linked covariance-matrix heatmap); **w14.2 Consistency Autopsy** (watch
EKF SLAM go overconfident and wrong on a long loop).
*[P]* `EkfSlam` with dynamic state growth; instrumented to export the covariance movie the widgets
replay.
*Sources: Thrun Ch. 10.*

**15. SLAM as Least Squares: Factor Graphs**
The modern backbone. Full-SLAM posterior in log form = sum of quadratic constraints = a factor
graph; MAP inference as sparse nonlinear least squares; Gauss-Newton and Levenberg-Marquardt;
sparsity, variable elimination, and the Schur complement (the EIF "reduce" step, reborn); robust
kernels and outlier-immune SLAM; the information-form lineage (EIF/SEIF) told honestly as history
whose sparsity insight won via smoothing.
*[C]* **w15.1 Spring-Graph Optimizer** (poses as nodes, factors as springs; energy relaxation runs
live; add a bad loop factor, then rescue it with a robust kernel); **w15.2 Sparsity Scope**
(information matrix fill-in as elimination order changes).
*[P]* hand-rolled dense GN/LM on nalgebra → sparse via `faer`; then the same graph in `factrs` 0.3
(GTSAM-style typed factors) to show the production path.
*Sources: Thrun Ch. 11–12 (restructured); Dellaert & Kaess; modernization set.*

**16. Scan Matching and Pose-Graph SLAM**
Front-end/back-end architecture; ICP (point-to-point, point-to-plane) derived and implemented;
NDT; scan-to-map matching; KISS-ICP-style odometry; loop-closure detection and verification;
pose-graph SLAM end-to-end — the Cartographer/SLAM-Toolbox recipe, built by hand.
*[C]* **w16.1 ICP Stepper** (correspondences → transform → iterate, scrub each iteration);
**w16.2 Loop Snap** (drift accumulates around the Apartment; loop closure fires; the rubber-band
correction propagates back through the trajectory).
*[P]* **RustSLAM-2D**: LiDAR odometry (ICP over `parry`-simulated scans) + pose graph on our
Ch. 15 optimizer + occupancy submaps = the book's first complete SLAM system.
*Sources: Thrun Ch. 14 (ancestor); KISS-ICP / LOAM lineage from modernization set.*

**17. FastSLAM and Rao-Blackwellization**
Factoring the SLAM posterior: sample paths, close maps in closed form; Rao-Blackwellization as a
theorem, not a trick; FastSLAM 1.0/2.0 with per-particle landmark EKFs and tree-shared maps;
grid-based RBPF (the gmapping recipe) with improved proposals; when particle SLAM still wins.
*[C]* **w17.1 Parallel Universes** (each particle drags its own private map; resampling picks
which universe survives; loop closure = universe selection).
*[P]* `FastSlam` (landmark version) + grid-RBPF on the Ch. 8 particle machinery; rayon-parallel
per-particle maps.
*Sources: FastSLAM (2005 ed. gap) sourced from modernization set; Thrun Ch. 13 EM as historical note.*

**18. Visual and Visual-Inertial SLAM**
The camera as a probabilistic sensor: projection factors and reprojection error; two-view geometry
in 90 seconds; IMU preintegration as a factor (the Lupton/Forster idea); MSCKF vs. sliding-window
smoothing; marginalization and its fill-in (SEIF's lesson resurfacing); ORB-SLAM3 and VINS as
*systems* — architecture described, not dissected.
*[C]* **w18.1 Reprojection Playground** (drag a 3D point / camera pose; residuals live);
**w18.2 Preintegration Timeline** (IMU samples compress into one factor between keyframes).
*[P]* bundle-adjust a tiny synthetic scene with our Ch. 15 optimizer + `sophus` SE(3); factor
graph with preintegrated IMU factors in `factrs`; deliberately smaller P section.
*Sources: entirely modernization set (Forster, MSCKF, ORB-SLAM3, VINS).*

**19. Modern Map Representations**
Beyond flat grids: octrees/OctoMap; TSDF and ESDF fusion *as recursive estimation* (per-voxel
weighted least squares — the Bayes filter in disguise); meshes; distance fields for planning; one
principled section on differentiable rendering as a measurement model (NeRF/3DGS mapping) with
survey pointers, not system worship.
*[C]* **w19.1 TSDF Sculptor** (watch depth scans carve a signed-distance field; weight slider);
**w19.2 Representation Gallery** (same Apartment as grid / octree / TSDF / mesh, memory & query
cost measured).
*[P]* 2D TSDF fusion + ESDF via distance transform; `three-d` for the one 3D showpiece.
*Sources: modernization set (OctoMap, Voxblox lineage, Tosi survey).*

---

### Part VI — Planning and Acting under Uncertainty

**20. Motion Planning: From Geometry to Probability**
Configuration space (via Lynch/Choset); graph search done right (Dijkstra, A*, D* Lite mention);
potential fields and their local minima; sampling-based planning: PRM, RRT, RRT* with
probabilistic completeness and asymptotic optimality stated precisely; planning on Rusty's real
constraints (nonholonomic RRT variants, Dubins/Reeds-Shepp primitives, hybrid-A* sketch).
*[C]* **w20.1 Planner Arena** (same map, same query: A* on a lattice vs PRM vs RRT vs RRT*, race
mode + tree growth animation); **w20.2 Narrow Passage** (why sampling struggles, visibility
intuition).
*[P]* `planning` module on `petgraph`/`pathfinding`; RRT* from scratch with `parry2d` collision;
Dubins steering for Rusty.
*Sources: Choset Chs. 5–7; Lynch & Park Ch. 10; Spong Ch. 7.*

**21. Decision Making I: MDPs and Value Iteration**
From plans to policies; rewards, discounting, the Bellman equation; value iteration and policy
extraction; stochastic shortest paths; the gridworld that refuses to walk straight; policy
iteration and asynchronous variants; the bridge: what if state itself is uncertain?
*[C]* **w21.1 Policy Painter** (edit rewards/noise; value function as heatmap and policy as arrows
re-converge live).
*[P]* `mdp` module: value/policy iteration over sparse transition models; Rusty navigating with
slippery wheels.
*Sources: Thrun Ch. 15.*

**22. Decision Making II: POMDPs and Belief-Space Planning**
The belief MDP; the tiger problem played honestly; $\alpha$-vectors and piecewise-linear-convex
value functions; exact value iteration and why it explodes; point-based methods (PBVI); modern
online solvers — POMCP and DESPOT — as the practical path; AMDP/coastal navigation (uncertainty-
aware behavior emerging from planning in belief space).
*[C]* **w22.1 Tiger Door Console** (play the POMDP yourself, then watch the optimal policy; belief
segment with $\alpha$-vector envelope underneath); **w22.2 Coastal Navigator** (why the optimal
path hugs the wall).
*[P]* exact finite-world POMDP solver; POMCP over the book's particle machinery; Rusty choosing
to relocalize before committing to a corridor.
*Sources: Thrun Ch. 16; POMCP/DESPOT from modernization set.*

**23. Stochastic Model Predictive Control: MPPI and Friends**
Receding-horizon control; DWA as the classical baseline; the path-integral idea: rollouts, costs,
exponential weighting — MPPI as importance sampling over controls (the particle filter's twin);
constraint handling; when MPPI beats gradient MPC and vice versa.
*[C]* **w23.1 Rollout Storm** (hundreds of candidate trajectories, cost-colored, collapsing into
the chosen control each frame — mesmerizing and precise).
*[P]* `Mppi` with rayon rollouts (single-thread on WASM, still real-time for 2D Rusty); obstacle
costs from the Ch. 19 ESDF; Rusty tracking a path through clutter.
*Sources: entirely modernization set (Williams et al.); Spong/Craig control chapters as classical context.*

**24. Exploration and Active SLAM**
Where should Rusty go *to learn*? Information gain and expected entropy reduction; frontier
exploration; active localization (choosing motions that disambiguate); active SLAM as decision-
making over the Ch. 15 graph (utility = coverage + expected information); stopping criteria;
the Placed et al. survey as the map of the field.
*[C]* **w24.1 Frontier Chaser** (frontiers highlighted, utilities scored, target chosen — watch the
map's entropy curve fall); **w24.2 Disambiguation Detour** (active localization choosing the
informative corridor over the short one).
*[P]* frontier detector + information-gain scorer over `OccGrid`; greedy explorer; hooks the
Ch. 16 SLAM stack — Rusty maps a floorplan it has never seen, autonomously.
*Sources: exploration gap (2005 ed. Ch. 17) rebuilt from modernization set (Placed et al. 2023).*

---

### Part VII — Frontiers and Integration

**25. Learning in the Loop**
Machine learning *inside* the Bayesian frame, not instead of it: learned and calibrated
observation models (the Ch. 13 learned-inverse-model idea, modernized); calibration curves;
differentiable filters (backprop through the KF/PF); diffusion policies as probabilistic action
models; what stays principled and what is still alchemy.
*[C]* **w25.1 Calibration Clinic** (overconfident vs. calibrated sensor model, reliability diagram
live); **w25.2 Differentiable Filter Trainer** (loss falls as the learned noise model adapts).
*[P]* `candle`: train a small learned beam model from simulator data; differentiate through the
Ch. 6 KF; honest benchmarks vs. the hand-tuned models.
*Sources: Thrun Chs. 9.3/13 seeds; differentiable-filter & diffusion-policy literature (modernization set).*

**26. Capstone: A Complete Autonomous Robot**
Everything, integrated and honest: architecture of an autonomy stack (estimation, mapping,
planning, control as concurrent tasks); Rusty dropped into an unknown apartment — explores
(Ch. 24), SLAMs (Ch. 16), plans (Ch. 20), controls (Ch. 23), replans on surprise; failure modes
tour (kidnaps, dynamic obstacles, sensor dropout) with the probabilistic machinery recovering;
engineering retrospective: what Rust's type system caught, what it cost, where the ecosystem
stands; where to go next (ROS 2 via `ros2-rust`, real hardware, 3D).
*[C]* **w26.1 The Grand Demo** (full-page: autonomous exploration of a randomized apartment, every
subsystem's internals inspectable — belief, map, graph, frontier scores, MPPI rollouts).
*[P]* `capstone` crate wiring every module of the workspace; the book's closing argument that the
whole stack — teaching code included — runs at real-time in the browser.
*Sources: whole book; systems framing from Choset taxonomy + modern stack architecture.*

---

### Appendices

- **A. Rust for Roboticists** — crash course for readers arriving from C++/Python: ownership for
  filter state, traits vs. inheritance, const generics for dimensions, error handling, the
  workspace, WASM builds; reading `nalgebra` type signatures without fear.
- **B. Matrix Identities & Gaussian Calculus** — inversion lemma, Schur complement, completing the
  square, Cholesky, marginal/conditional of a joint Gaussian: every identity the derivations use,
  proved once, referenced everywhere.
- **C. Lie Group Reference Card** — SO(2)/SE(2)/SO(3)/SE(3): exp/log, adjoints, left/right
  Jacobians, $\boxplus/\boxminus$ tables, numerical edge cases — the page every reader bookmarks.
- **D. The Simulator & Widget Framework** — architecture reference for the `sim` crate and the
  shared widget chrome (autoplay, seed control, color code, static fallback rendering).

---

## 4. Baseline Source Map (summary)

| Baseline | Role in this book |
|---|---|
| Thrun, Burgard, Fox — *Probabilistic Robotics* (draft in Resource/) | Spine of Parts II–VI; notation baseline; algorithm tables. **Note: the PDF is the 1999–2000 16-chapter draft** — FastSLAM, GraphSLAM (published form), UKF, KLD-sampling, and exploration are rebuilt from the modernization literature. |
| Lynch & Park — *Modern Robotics* | Ch. 3 geometry (SO/SE, exp/log, screws); Ch. 4 mobile-robot substrate; Ch. 20 planning baseline. |
| Craig — *Introduction to Robotics* | Frames/transform pedagogy in Ch. 3; kinematics notation cross-reference. |
| Niku — *Introduction to Robotics* | Ch. 4 sensor/actuator device grounding; vision-system context for Ch. 18. |
| Choset et al. — *Principles of Robot Motion* | Part VI planning (C-space, PRM/RRT); taxonomy scaffolding; appendix style. |
| Spong et al. — *Robot Modeling and Control* | Kinematics/Jacobians cross-reference; control-theory context for Ch. 23; mobile-robot constraints. |
| Modernization set | Dellaert & Kaess (factor graphs), Sola (micro Lie), Barfoot (state estimation), KISS-ICP/LOAM, Forster preintegration, MSCKF/ORB-SLAM3/VINS, OctoMap/Voxblox, MPPI (Williams), POMCP/DESPOT, Placed et al. (active SLAM), differentiable filters & diffusion policies. |

## 5. Deliverable Map

- `TOC.md` — this file: the book's contract.
- `Chapter-01.md` … `Chapter-26.md` — per-chapter design docs (storyline, F/C/P plan, widget
  manifest, Rust module plan, exercises, modernization notes) following the shared template.
- `CLAUDE.md` — the development guide: workspace layout, build/test/deploy pipeline, widget
  framework, authoring conventions, and the definition of done for a chapter.
