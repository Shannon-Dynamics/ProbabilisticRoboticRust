# Chapter 18 — Visual and Visual-Inertial SLAM

> Part V — Mapping and SLAM · Estimated length: 13 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Every sensor so far measured *distance*. A camera measures something stranger: the direction to a
point, projected onto a plane, with depth annihilated. This chapter treats the camera exactly the
way Ch. 10 treated the LiDAR — as a probabilistic sensor with a generative model — and shows that
once you write the reprojection factor, the whole Ch. 15 machinery (sparse least squares, Schur
complement, robust kernels, marginalization) applies unchanged: bundle adjustment is "just" a
factor graph with bearing-only factors. The second protagonist is the IMU, whose preintegration
(Lupton; Forster et al.) is the most elegant on-manifold idea in modern robotics: hundreds of
high-rate measurements compressed into *one* relative factor whose bias dependence survives as a
first-order Jacobian. The "aha": visual-inertial SLAM is not a new theory — it is Chapters 3, 7,
and 15 meeting two new measurement models, and even its skeletons are familiar (marginalization
fill-in is SEIF's sparsification lesson resurfacing). Deliberately, this chapter *describes*
ORB-SLAM3 and VINS as systems rather than dissecting them, and its Practical section is the
smallest in Part V: one tiny synthetic bundle adjustment and one preintegration factor graph.

Story line:
1. Hook: one photo of the Apartment; where is the camera? (Answer: on a ray-constraint manifold — a pixel is a bearing, not a position.)
2. The pinhole model as $p(z_t \mid x_t, m)$; reprojection error; the factor-graph view of bundle adjustment.
3. Two-view geometry in 90 seconds: epipolar constraint, triangulation, and monocular scale-blindness.
4. The IMU as a motion sensor; why naive re-integration inside an optimizer is ruinous; preintegration on the manifold.
5. Two architectures for the same posterior: MSCKF (filtering, structureless) vs sliding-window smoothing (VINS-style), and marginalization's fill-in tax.
6. Systems tour: ORB-SLAM3 and VINS-Mono/Fusion as block diagrams; what the front-end owes to learning (pointers only).
7. Integration lab: bundle-adjust a synthetic scene with our own optimizer + `sophus`; a two-keyframe VI graph in `factrs`.

## 2. Prerequisites & Position

- Builds on: Ch. 3 (SO(3)/SE(3), quaternions, $\exp/\log$, $\boxplus/\boxminus$ — first serious 3D use),
  Ch. 7 (error-state EKF: MSCKF is its payoff), Ch. 10 (sensor-model discipline: generative model
  first), Ch. 14 §flaws + draft SEIF lineage (marginalization fill-in), Ch. 15 (optimizer, Schur
  complement, robust kernels, marginalization as variable elimination).
- Feeds into: Ch. 19 (dense maps consume posed keyframes; differentiable rendering closes the
  loop back to measurement models), Ch. 25 (learned front-ends and calibrated likelihoods),
  Ch. 26 (retrospective: where a camera would slot into Rusty's stack).
- Baseline sources: **entirely modernization set** — Forster, Carlone, Dellaert, Scaramuzza
  (on-manifold preintegration, RSS 2015 / T-RO 2017); Lupton & Sukkarieh 2012; Mourikis &
  Roumeliotis 2007 (MSCKF); Campos et al. 2021 (ORB-SLAM3, T-RO); Qin et al. (VINS-Mono/Fusion);
  Dellaert & Kaess 2017 (BA-as-factor-graph, Schur complement). The 2005 baseline and the
  1999–2000 draft contain no vision material; the draft's Ch. 12 (SEIF §12.5 sparsification) is
  cited when marginalization-induced fill-in and inconsistency resurface. Device grounding for
  cameras: Niku's vision-systems chapters (per the book's source map). Depth boundary: Hartley &
  Zisserman and Szeliski are named as the *actual* geometry textbooks; we teach only the
  probabilistic structure.

## 3. Foundation (F) — Mathematical Core

### Definitions & notation introduced

| Symbol | Meaning |
|---|---|
| $\mathbf{K}$, $(f_x, f_y, c_x, c_y)$ | camera intrinsics (pinhole; distortion mentioned, not modeled) |
| $\pi : \mathbb{R}^3 \to \mathbb{R}^2$ | projection, $\pi(\mathbf{P}) = (f_x X/Z + c_x,\; f_y Y/Z + c_y)^\top$ |
| $T_{cw} \in SE(3)$ | world-to-camera pose; ${}^{c}\mathbf{P} = T_{cw}\,{}^{w}\mathbf{P}$ |
| $\mathbf{e}_{kj}$ | reprojection residual of point $j$ in keyframe $k$ |
| $\mathbf{E} = [\mathbf{t}]_\times \mathbf{R}$ | essential matrix; epipolar constraint $\mathbf{q}_2^\top \mathbf{E}\, \mathbf{q}_1 = 0$ |
| $\mathbf{b} = (\mathbf{b}_g, \mathbf{b}_a)$ | gyro/accelerometer biases (slowly varying random walk) |
| $\Delta \mathbf{R}_{ij}, \Delta \mathbf{v}_{ij}, \Delta \mathbf{p}_{ij}$ | preintegrated rotation/velocity/position deltas between keyframes $i, j$ |
| $\mathbf{g}$ | gravity vector in the world frame |

**The camera as a probabilistic sensor.** $z_{kj} = \pi(T_{cw,k}\, {}^{w}\mathbf{P}_j) + \boldsymbol{\delta}_{kj}$,
$\boldsymbol{\delta} \sim \mathcal{N}(\mathbf{0}, \mathbf{Q}_{px})$ with $\mathbf{Q}_{px} \approx \sigma_{px}^2 \mathbf{I}$ (1–2 px). Everything else in the chapter is
consequences of this one line plus the IMU's strapdown kinematics.

### Derivations

1. **Reprojection factor and its Jacobians.**
   *Statement:* $\mathbf{e}_{kj} = \mathbf{z}_{kj} - \pi(T_{cw,k}\, {}^{w}\mathbf{P}_j)$; MAP over poses+points = sparse NLLS; the
   Jacobian splits by chain rule into $\partial \pi / \partial {}^{c}\mathbf{P}$ (a $2{\times}3$ with the famous $1/Z$,
   $1/Z^2$ entries) times $\partial {}^{c}\mathbf{P}/\partial \boldsymbol{\xi}_k \in \mathbb{R}^{3 \times 6}$ (pose tangent) and
   $\partial {}^{c}\mathbf{P}/\partial {}^{w}\mathbf{P}_j = \mathbf{R}_{cw}$.
   *Sketch:* write the residual; differentiate the projection; differentiate the group action via
   $\boxplus$ perturbation $T \exp(\boldsymbol{\xi}^\wedge)$; assemble the two Jacobian blocks; note the
   arrow-shaped sparsity of the resulting normal equations.
   *Collapsible:* full $2{\times}6$ pose Jacobian entries; left- vs right-perturbation conventions
   (Appendix C); how depth $Z \to 0$ blows up the Jacobian and why systems gate near-frustum points.
2. **Bundle adjustment eliminates points first (the Schur trick, again).**
   *Statement:* the BA normal equations have block structure
   $\begin{pmatrix}\mathbf{H}_{pp} & \mathbf{H}_{pl} \\ \mathbf{H}_{pl}^\top & \mathbf{H}_{ll}\end{pmatrix}$ with $\mathbf{H}_{ll}$ block-diagonal
   ($3{\times}3$ per point); eliminating points via the Schur complement
   $\mathbf{H}_{pp} - \mathbf{H}_{pl} \mathbf{H}_{ll}^{-1} \mathbf{H}_{pl}^\top$ costs $O(L)$ and leaves a small pose system.
   *Sketch:* points are the "map features" of Ch. 15's `reduce` step — literally the draft's
   `EIF_reduce` (Table 11.3) reborn with $2{\times}$ smaller blocks; invert per-point blocks; back-substitute
   for points after solving poses.
   *Collapsible:* fill-in analysis (which pose pairs couple after elimination: co-observing
   keyframes), and the complexity table dense vs Schur vs sparse-Cholesky-on-the-reduced-system.
3. **Two-view geometry in 90 seconds.**
   *Statement:* for calibrated rays $\mathbf{q}_1, \mathbf{q}_2$ of the same point under relative motion
   $(\mathbf{R}, \mathbf{t})$: $\mathbf{q}_2^\top [\mathbf{t}]_\times \mathbf{R}\, \mathbf{q}_1 = 0$; $\mathbf{E}$ has 5 DOF; $\mathbf{t}$ is recoverable only up to
   scale; triangulation is a 2-view least squares whose depth uncertainty grows as baseline shrinks.
   *Sketch:* coplanarity of the two rays and the baseline → triple product = 0; count DOF; show the
   scale gauge freedom ($\mathbf{t} \to s\mathbf{t}$, $\mathbf{P} \to s\mathbf{P}$ leaves all residuals invariant) — monocular
   SLAM's unobservable direction, fixed by the IMU (gravity + accelerometer scale) or a second camera.
   *Collapsible:* DLT triangulation derivation; the five-point algorithm named with references,
   not derived; degenerate motions (pure rotation ⇒ no parallax ⇒ no depth).
4. **IMU preintegration on the manifold (Forster). The chapter's centerpiece.**
   *Statement:* with measurements $\tilde{\boldsymbol{\omega}}_k = \boldsymbol{\omega}_k + \mathbf{b}_g + \boldsymbol{\eta}_g$,
   $\tilde{\mathbf{a}}_k = \mathbf{R}_k^\top(\mathbf{a}_k - \mathbf{g}) + \mathbf{b}_a + \boldsymbol{\eta}_a$, define
   $$\Delta \mathbf{R}_{ij} = \prod_{k=i}^{j-1} \exp\big((\tilde{\boldsymbol{\omega}}_k - \mathbf{b}_g)^\wedge \Delta t\big), \quad
     \Delta \mathbf{v}_{ij} = \sum_{k=i}^{j-1} \Delta \mathbf{R}_{ik} (\tilde{\mathbf{a}}_k - \mathbf{b}_a) \Delta t, \quad
     \Delta \mathbf{p}_{ij} = \sum_{k=i}^{j-1} \big[ \Delta \mathbf{v}_{ik} \Delta t + \tfrac{1}{2} \Delta \mathbf{R}_{ik} (\tilde{\mathbf{a}}_k - \mathbf{b}_a) \Delta t^2 \big].$$
   These depend on measurements and biases but *not* on the absolute states — so they can be
   computed once per keyframe interval and reused across optimizer iterations. The residual, e.g.
   $\mathbf{r}_{\Delta v} = \mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\, \Delta t_{ij}) - \Delta \mathbf{v}_{ij}$, ties states $i, j$ through the deltas;
   bias change is absorbed to first order: $\Delta \mathbf{R}_{ij}(\mathbf{b}_g + \delta \mathbf{b}_g) \approx \Delta \mathbf{R}_{ij} \exp\big((\mathbf{J}_{\Delta R}^{b_g} \delta \mathbf{b}_g)^\wedge\big)$.
   *Sketch:* (i) write strapdown integration of state $i \to j$; (ii) factor out $\mathbf{R}_i, \mathbf{v}_i, \mathbf{g}$
   — what remains is measurement-only; (iii) propagate the $9{\times}9$ covariance of
   $(\delta\boldsymbol{\phi}, \delta\mathbf{v}, \delta\mathbf{p})$ iteratively alongside; (iv) accumulate bias Jacobians the same
   way; (v) one Gaussian factor per keyframe pair, $\sim$200 raw samples compressed.
   *Collapsible:* the full noise-propagation recursion (state-transition blocks), right-Jacobian
   $\mathbf{J}_r$ appearances, the bias random-walk factor, and why the rotation lives in $SO(3)$ not
   $\mathbb{R}^3$ (Ch. 7's wrap-around lesson at 200 Hz).
5. **MSCKF vs sliding-window smoothing; marginalization's fill-in.**
   *Statement:* MSCKF keeps a window of camera poses in an error-state EKF and applies
   *structureless* updates: stack a feature's residuals, $\mathbf{r} \approx \mathbf{H}_x \delta\mathbf{x} + \mathbf{H}_f \delta\mathbf{f} + \mathbf{n}$,
   left-multiply by a basis $\mathbf{N}$ of the left null space of $\mathbf{H}_f$ ($\mathbf{N}^\top \mathbf{H}_f = \mathbf{0}$) — landmarks
   never enter the state. Sliding-window smoothers keep recent states + landmarks in an NLLS
   problem and *marginalize* old ones: $\mathbf{\Omega}' = \mathbf{\Omega}_{\beta\beta} - \mathbf{\Omega}_{\beta\alpha}\mathbf{\Omega}_{\alpha\alpha}^{-1}\mathbf{\Omega}_{\alpha\beta}$ —
   which densifies the remaining prior and freezes linearization points.
   *Sketch:* both are the same posterior handled by different approximation budgets; the Schur
   complement is Ch. 15's elimination and the draft's `EIF_reduce`; fill-in among survivors is
   *exactly* the effect SEIF fought by sparsification (draft Ch. 12 §12.5), and modern systems
   accept it inside a small dense prior instead of breaking links; consistency requires
   first-estimates-Jacobian discipline (named, referenced, not derived).
   *Collapsible:* null-space projection dimensions and QR implementation; observability accounting
   — VIO's four unobservable DOF (global position + yaw), gravity making roll/pitch observable;
   FEJ statement with references.

### Named algorithms

| Algorithm | Signature | Complexity |
|---|---|---|
| `project` / `project_jacobians` | $(\mathbf{K}, T_{cw}, {}^{w}\mathbf{P}) \to \mathbf{z} \in \mathbb{R}^2$ (+ $2{\times}6$, $2{\times}3$ blocks) | $O(1)$ |
| `triangulate_dlt` | $(T_{cw,1}, T_{cw,2}, \mathbf{q}_1, \mathbf{q}_2) \to {}^{w}\mathbf{P}$ | $O(1)$ (small SVD) |
| `ba_solve` | (poses, points, factors) $\to$ MAP estimate | per GN iter $O(F)$ residuals + Schur $O(L)$ + pose solve $O(K^3)$ dense (tiny $K$ here) |
| `preintegrate` | (IMU samples $[t_i, t_j]$, $\mathbf{b}$) $\to (\Delta\mathbf{R}, \Delta\mathbf{v}, \Delta\mathbf{p}, \mathbf{\Sigma}_{9\times9}, \mathbf{J}_b)$ | $O(n)$ samples, once per interval |
| `imu_residual` | (states $i, j$, gravity, preintegrated delta) $\to \mathbf{r} \in \mathbb{R}^9$ | $O(1)$ per optimizer iteration — the whole point |
| `msckf_update` | (window state, feature track) $\to$ updated state | $O(\text{track len})$ QR + EKF update; no landmark states |
| `marginalize` | (graph, old states) $\to$ dense prior factor | Schur complement on the old block |

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w18.1: Reprojection Playground** *(flagship; interactive sim)* — Split view. Right: a
  3D scene (`three-d`: camera frustum, a dozen world points, orbitable). Left: that camera's image
  plane with observed pixels in **green** and current-estimate projections in **purple**, connected
  by residual whiskers; total cost as a live number. The reader drags the camera pose (or one 3D
  point) in the right pane and watches residuals stretch; a "one GN step" button snaps the estimate
  partway back (iterates in **orange** along the way). One meaningful parameter: pixel noise σ.
  Autoplay: a slow orbit + GN re-convergence loop. *Misconception killed:* "a pixel tells you where
  a point is" — dragging the camera *along the viewing ray* leaves residuals flat (the whiskers
  don't care), making depth-blindness and the need for parallax physically tangible. Static
  fallback: two-pane figure with residual whiskers before/after GN.
- **Widget w18.2: Preintegration Timeline** *(flagship; animation + sliders)* — A horizontal
  timeline: two keyframes as posts, 200 IMU sample ticks between them (**green**). On autoplay the
  ticks sweep up and compress into a single factor blob on the graph edge, annotated with
  $(\Delta\mathbf{R}, \Delta\mathbf{v}, \Delta\mathbf{p})$ and a shrinking $9{\times}9$ covariance heatmap. Slider 1: gyro bias
  $\delta b_g$ — the factor *reshapes instantly* via the stored Jacobian (a ghost overlay shows the
  expensive re-integration result for comparison: visually identical for small $\delta b$, diverging
  for large). Slider 2: keyframe spacing — covariance grows superlinearly. *Misconceptions killed:*
  "the optimizer must re-integrate the IMU every iteration" and "an IMU measures position" (it
  measures specific force and angular rate; position is double integration with gravity in the loop).
  Static fallback: three-frame compression storyboard.
- **Widget w18.3: Marginalization Fill-In** *(supporting; interactive sim, reuses w15.2's
  machinery)* — A sliding window of 6 poses + 10 landmarks as a graph beside its information-matrix
  heatmap (**blue** = existing entries). Click "slide window": the oldest pose marginalizes out,
  new dense entries flash **orange** among everything it touched, then settle **purple**. A counter
  tracks nonzeros over repeated slides; a toggle "SEIF-style: drop weak links" shows sparsity
  recovered at the price of a consistency warning banner. *Misconception killed:* "marginalization
  just deletes old states" — information is conserved and *relocated*, and deleting it instead is
  a modeling decision with consequences (the draft Ch. 12 lesson, replayed interactively).
- **Figure f18.1:** ORB-SLAM3 and VINS-Fusion architecture block diagrams, drawn side by side with
  the book's color code (front-end **green** measurement flow, back-end **purple** estimation flow),
  annotated with which chapter taught each block. Static SVG — systems are described, not simulated.

## 5. Practical (P) — Rust Implementation

Deliberately the smallest P-section of Part V (per the book contract): two focused artifacts, no
feature front-end, no full VIO system.

Crates: `nalgebra` 0.35 (small fixed blocks: `SMatrix<f64, 2, 6>`, `SMatrix<f64, 9, 9>`),
`sophus` (pinned minor version — SO(3)/SE(3) exp/log for poses and preintegration; first heavy 3D
use, cross-checked against Appendix C), `factrs` 0.3 (production factor graph for the
VI example; its `fac![]` macro, robust kernels, and rerun hook), `rand`/`rand_distr` 0.9/0.6
(synthetic tracks + IMU noise, seeded), `three-d` 0.19 (w18.1's 3D pane), `egui`/`eframe` 0.35
(controls), `plotters` (convergence figures), `rerun` 0.26 (optional native `.rrd` of the BA scene).

Module plan: `crates/ch18_vio/` — `pinhole.rs`, `synth.rs` (scene + IMU trajectory generator),
`tiny_ba.rs` (Ch. 15 GN + Schur-on-points, poses via `sophus`), `preint.rs` (hand-rolled
preintegration — the teaching artifact), `factrs_vi.rs` (the same math via `factrs` factors);
demos `demos/ch18-reprojection/`, `ch18-preintegration/`, `ch18-marginalization/`.

```rust
use nalgebra::{SMatrix, Vector2, Vector3};
use sophus::lie::{Isometry3F64, Rotation3F64};

pub struct Pinhole { pub fx: f64, pub fy: f64, pub cx: f64, pub cy: f64 }
impl Pinhole {
    pub fn project(&self, p_cam: &Vector3<f64>) -> Option<Vector2<f64>>;      // None if Z <= z_min
    pub fn jacobians(&self, t_cw: &Isometry3F64, p_w: &Vector3<f64>)
        -> (SMatrix<f64, 2, 6>, SMatrix<f64, 2, 3>);                          // Derivation 1
}

pub struct ReprojFactor { pub kf: usize, pub pt: usize, pub z: Vector2<f64>, pub sqrt_info: SMatrix<f64, 2, 2> }

/// Tiny bundle adjustment: Ch. 15 Gauss-Newton with Schur elimination of points.
pub struct TinyBa { pub poses: Vec<Isometry3F64>, pub points: Vec<Vector3<f64>>, pub factors: Vec<ReprojFactor> }
impl TinyBa {
    pub fn solve_gn(&mut self, iters: usize) -> BaReport;                     // Derivation 2
    pub fn gauge_fix(&mut self);                                              // pin pose 0 + one scale
}
pub struct BaReport { pub rmse_px_per_iter: Vec<f64> }

pub struct ImuSample { pub gyro: Vector3<f64>, pub acc: Vector3<f64>, pub dt: f64 }
pub struct ImuBias  { pub gyro: Vector3<f64>, pub acc: Vector3<f64> }
pub struct Preintegrated {
    pub dt_ij: f64,
    pub d_rot: Rotation3F64,
    pub d_vel: Vector3<f64>,
    pub d_pos: Vector3<f64>,
    pub cov: SMatrix<f64, 9, 9>,       // (δφ, δv, δp)
    pub j_bias: SMatrix<f64, 9, 6>,    // first-order bias correction (Derivation 4)
}
pub fn preintegrate(samples: &[ImuSample], bias: &ImuBias) -> Preintegrated;
pub fn corrected(pre: &Preintegrated, delta_bias: &ImuBias) -> Preintegrated;  // O(1), no re-integration
```

Worked end-to-end example (`cargo run --example tiny_ba`): seed `0x5EE3D`, 8 cameras on a ring
viewing 40 points on a cube, σ = 1 px, initial poses perturbed by 5°/0.2 m and points by 0.3 m.
Expected (unit-tested) output: reprojection RMSE 18.4 px → 0.94 px in 5 GN iterations, gauge fixed
by pinning camera 0 and the 0–1 baseline; a `plotters` SVG of the cost curve and the recovered
ring. Second example (`cargo run --example factrs_vi`): two keyframes, 200 synthetic IMU samples,
6 shared landmarks; a `factrs` graph with reprojection + preintegrated-IMU + bias factors recovers
velocity to 2 cm/s and gyro bias to 5e-4 rad/s; demonstrates that our hand-rolled
`preintegrate` matches `factrs`' factor residuals to 1e-9 (the cross-check test). The WASM
artifact is w18.1; `factrs` stays native-side unless its WASM build is CI-verified (flagged risk).

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w18.1 | Reprojection Playground | wasm-sim | three-d, eframe, ch18_vio | drag camera/point; GN-step button; noise slider | pixels are bearings; depth needs parallax; BA convergence |
| w18.2 | Preintegration Timeline | animation + sliders | eframe, egui_plot, ch18_vio | bias slider; keyframe-spacing slider | measurement compression; O(1) bias correction; covariance growth |
| w18.3 | Marginalization Fill-In | wasm-sim | eframe, ch18_vio, ch15 machinery | slide-window button; drop-links toggle | Schur fill-in; SEIF's lesson; sparsity vs consistency |
| f18.1 | ORB-SLAM3 / VINS architectures | static-svg | plotters (hand-layout) | — | real systems as compositions of this book's blocks |

## 7. Exercises & Extensions

1. **[F]** Derive the $2{\times}6$ pose Jacobian of Derivation 1 under a *left* perturbation and show which columns vanish when the point lies on the optical axis. Check numerically against `Pinhole::jacobians` (the test scaffold is provided).
2. **[F]** Prove monocular scale unobservability: exhibit the direction in state space along which all reprojection residuals are exactly invariant, and show why adding the accelerometer term $\mathbf{g}\,\Delta t_{ij}$ in $\mathbf{r}_{\Delta v}$ destroys that invariance.
3. **[F]** Derive the recursion for the preintegrated covariance's $(\delta\boldsymbol{\phi}, \delta\mathbf{v})$ blocks (Derivation 4 collapsible, first two rows) and verify the $\Delta t^{3/2}$ growth of velocity uncertainty against w18.2.
4. **[C]** In w18.1, predict which of the two motions — 10 cm sideways vs 10 cm forward — changes residuals more for a centered near point; verify, then relate to Derivation 3's degenerate motions.
5. **[C]** Using w18.3, predict the nonzero count after 5 window slides with 3 co-observed landmarks per slide; verify, and explain which entries SEIF-style link-dropping would remove first.
6. **[P]** Add a stereo rig to `TinyBa` (second `ReprojFactor` set with a fixed known baseline) and show the scale gauge no longer needs fixing; then corrupt 10% of matches and rescue the solve with a Huber kernel via the Ch. 15 machinery.

## 8. Modernization Notes

- **Nothing in this chapter exists in the 2005 book or the 1999–2000 draft** — no cameras, no
  IMUs, no preintegration, no VIO. It is built entirely from the modernization set (Forster
  preintegration as centerpiece; MSCKF; ORB-SLAM3; VINS), exactly as the book contract's source
  map prescribes.
- The one deliberate thread of continuity: the draft's SEIF chapter (Ch. 12 §12.5 sparsification)
  is *resurrected as a lesson* rather than an algorithm — sliding-window marginalization recreates
  the same fill-in/consistency dilemma, and modern practice (small dense prior + FEJ) is presented
  as the settled answer to the question SEIF asked in 2000. This preserves the historical insight
  while honoring the modernization guidance to cut SEIF as a method.
- Scope discipline (and what was dropped): no feature detection/description/matching pipeline
  (Hartley & Zisserman / Szeliski pointers), no five-point derivation, no rolling shutter, no
  event cameras, no dense/learned visual SLAM (DROID-SLAM et al. — one pointer box feeding
  Ch. 25). ORB-SLAM3/VINS get architecture diagrams and honest accuracy-lineage claims only.
  The P-section is intentionally the smallest in Part V: the book's robot is a 2D LiDAR platform,
  and this chapter's job is to transfer the *probabilistic structure* of visual-inertial
  estimation, not to build a fourth SLAM system.
- Sequencing note for implementers: this is the first chapter whose Foundation runs primarily on
  SE(3); every derivation cross-links to Appendix C's Jacobian tables rather than re-deriving them.
