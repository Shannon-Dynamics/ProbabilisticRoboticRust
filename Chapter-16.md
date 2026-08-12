# Chapter 16 — Scan Matching and Pose-Graph SLAM

> Part V — Mapping and SLAM · Estimated length: 16 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Chapter 15 gave the reader a back-end: a factor graph and a sparse Gauss-Newton/LM optimizer that can
swallow constraints and spit out a maximum-a-posteriori trajectory. But it never said where the
constraints come from. This chapter builds the **front-end** — the machinery that turns raw LiDAR
scans into odometry factors and loop-closure factors — and then wires front-end to back-end into
**RustSLAM-2D**, the book's first complete SLAM system. The "aha": SLAM in practice is not one
algorithm but an *architecture* — a fast, greedy, local matcher feeding a slow, global, honest
optimizer — and every piece of it (ICP as MLE, NDT as a Gaussian mixture, loop verification as a
χ² gate) is the same probability theory the reader has used since Chapter 5. The historical hook:
Thrun's 1999–2000 draft Ch. 14 already contained this chapter in embryo — incremental ML pose
hill-climbing plus "correcting poses backwards in time" on cycle detection — and we tell it as the
ancestor whose two halves grew up into scan matching and pose-graph optimization.

Story line:
1. Problem: Rusty drives one loop of the Apartment on raw odometry; the map shears into nonsense (hook figure).
2. Intuition: two overlapping scans "want" to snap together — alignment is information about relative pose.
3. Formalism: registration as MLE; point-to-point ICP solved in closed form by SVD; point-to-plane; NDT.
4. Engineering: scan-to-map, submaps, KISS-ICP-style adaptive odometry — a front-end that drifts slowly.
5. Loop closure: detection (where might I be back?), verification (is this match real?), and the cost of being wrong.
6. Architecture: front-end/back-end; the Cartographer / SLAM Toolbox recipe stated honestly.
7. Integration lab: RustSLAM-2D in the Apartment — drift accumulates, the loop snaps shut, submaps fuse into one map.

## 2. Prerequisites & Position

- Builds on: Ch. 3 (SE(2), $\exp/\log$, $\boxplus/\boxminus$), Ch. 4 (simulated LiDAR via `parry2d`),
  Ch. 9 (odometry motion model as prior), Ch. 10 (likelihood fields — reused as a matching cost),
  Ch. 13 (occupancy grids for submaps), Ch. 14 (why filtering SLAM breaks — the motivation),
  Ch. 15 (the optimizer, robust kernels, sparsity).
- Feeds into: Ch. 17 (grid RBPF compared against this system), Ch. 19 (map representations consume
  RustSLAM-2D output), Ch. 24 (exploration drives this SLAM stack), Ch. 26 (capstone).
- Baseline sources: Thrun et al. (1999–2000 draft) Ch. 14 §14.2–14.4 (`incremental_ML_mapping`,
  Table 14.1; gradient ascent in pose space; cycle detection and backwards correction) — cited as
  the *historical ancestor*, not the method taught. Modernization set: ICP/GICP lineage and the
  LiDAR Odometry Survey (arXiv:2312.17487); NDT (Biber & Straßer 2003); KISS-ICP (Vizzo et al.,
  RA-L 2023); LOAM lineage as context; Cartographer / SLAM Toolbox as the production 2D recipe;
  Dellaert & Kaess (2017) for the graph view.

## 3. Foundation (F) — Mathematical Core

### Definitions & notation introduced

| Symbol | Meaning |
|---|---|
| $\mathcal{P} = \{\mathbf{p}_k\}_{k=1}^{N}$, $\mathcal{Q} = \{\mathbf{q}_k\}$ | source scan / target (reference) point set, $\mathbf{p}_k \in \mathbb{R}^2$ |
| $T = (\mathbf{R}, \mathbf{t}) \in SE(2)$ | rigid registration transform; acts on points as $T\mathbf{p} = \mathbf{R}\mathbf{p} + \mathbf{t}$ |
| $c(k)$ | correspondence index: source point $k$ matches target point $q_{c(k)}$ (the $c_t$ of Ch. 10, reborn geometrically) |
| $\mathbf{n}_k$ | unit surface normal at target point $\mathbf{q}_{c(k)}$ |
| $(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ | NDT cell Gaussian for cell $i$ |
| $Z_{ij}, \mathbf{\Omega}_{ij}$ | relative-pose measurement between poses $i,j$ and its information matrix |
| $\mathbf{e}_{ij}$ | pose-graph residual, $\mathbf{e}_{ij} = \log\!\big(Z_{ij}^{-1}\, T_i^{-1} T_j\big)^{\vee} \in \mathbb{R}^3$ |
| $\tau_t$ | adaptive correspondence-rejection threshold (KISS-ICP style) |

**Registration as MLE.** Model each matched target point as a noisy observation of the transformed
source point, $\mathbf{q}_{c(k)} = T\mathbf{p}_k + \boldsymbol{\epsilon}_k$, $\boldsymbol{\epsilon}_k \sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$. Then

$$T^\star = \arg\max_T \prod_k p(\mathbf{q}_{c(k)} \mid T, \mathbf{p}_k) = \arg\min_{T \in SE(2)} \sum_k \big\| T\mathbf{p}_k - \mathbf{q}_{c(k)} \big\|^2 ,$$

i.e. least squares *is* maximum likelihood here — one line that connects this chapter to Ch. 15's
factors and licenses robust kernels $\rho(\cdot)$ when the Gaussian assumption fails (wrong correspondences).

### Derivations

1. **Rigid alignment in closed form (Arun/Umeyama SVD).**
   *Statement:* given fixed correspondences, the minimizer of $\sum_k \|\mathbf{R}\mathbf{p}_k + \mathbf{t} - \mathbf{q}_k\|^2$ over $SE(d)$ is
   $\mathbf{R}^\star = \mathbf{U}\,\mathrm{diag}(1, \det(\mathbf{U}\mathbf{V}^\top))\,\mathbf{V}^\top$, $\mathbf{t}^\star = \bar{\mathbf{q}} - \mathbf{R}^\star \bar{\mathbf{p}}$,
   where $\mathbf{W} = \sum_k (\mathbf{q}_k - \bar{\mathbf{q}})(\mathbf{p}_k - \bar{\mathbf{p}})^\top = \mathbf{U}\mathbf{S}\mathbf{V}^\top$.
   *Sketch:* (i) optimize $\mathbf{t}$ first → both clouds centered at centroids; (ii) expand the
   centered cost → minimizing it equals maximizing $\mathrm{tr}(\mathbf{R}^\top \mathbf{W})$;
   (iii) orthogonal Procrustes: trace is maximized by $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$;
   (iv) the $\det$ correction excludes reflections; (v) back-substitute for $\mathbf{t}^\star$.
   *Collapsible:* full trace-inequality argument ($\mathrm{tr}(\mathbf{R}^\top\mathbf{U}\mathbf{S}\mathbf{V}^\top) \le \sum_i s_i$), the reflection
   case with a worked degenerate example (collinear scan), and the weighted variant.
2. **ICP as alternating minimization (the EM echo).**
   *Statement:* alternating (a) $c(k) \leftarrow \arg\min_j \|T\mathbf{p}_k - \mathbf{q}_j\|$ and (b) the closed-form
   alignment monotonically decreases the objective and converges to a local minimum.
   *Sketch:* both steps minimize the same joint cost $J(T, c)$ over one argument with the other
   fixed; $J$ is bounded below; monotone + bounded → convergent. Correspondence = E-step analog,
   alignment = M-step analog (echoes Ch. 10's EM for beam-model intrinsics).
   *Collapsible:* why convergence is only to a *local* minimum, with the two canonical failure
   geometries (rotational ambiguity in a corridor; picket-fence aliasing) worked as figures.
3. **Point-to-plane normal equations.**
   *Statement:* minimizing $\sum_k \big(\mathbf{n}_k^\top(\mathbf{R}\mathbf{p}_k + \mathbf{t} - \mathbf{q}_{c(k)})\big)^2$ has no closed form; with the
   small-angle substitution $\mathbf{R} \approx \mathbf{I} + \theta \mathbf{J}$, $\mathbf{J} = \begin{pmatrix}0 & -1\\ 1 & 0\end{pmatrix}$, it becomes linear least squares in $(\theta, \mathbf{t}) \in \mathbb{R}^3$.
   *Sketch:* substitute, collect the per-point row $\mathbf{a}_k^\top = (\mathbf{n}_k^\top \mathbf{J}\mathbf{p}_k, \; \mathbf{n}_k^\top)$ and scalar
   residual $b_k$; solve the $3{\times}3$ normal equations; iterate. One Gauss-Newton step of Ch. 15,
   specialized. Converges in far fewer iterations than point-to-point on structured (walls) scenes
   because the cost is flat *along* surfaces.
   *Collapsible:* exact SE(2)-tangent Jacobian via $\boxplus$, equivalence to Gauss-Newton on the manifold,
   and normal estimation by local PCA with its own noise analysis.
4. **NDT as a Gaussian-mixture measurement model.**
   *Statement:* representing the target scan by per-cell Gaussians $(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)$ and scoring
   $s(T) = \sum_k \exp\!\big(-\tfrac{1}{2}\mathbf{d}_k^\top \boldsymbol{\Sigma}_{i(k)}^{-1} \mathbf{d}_k\big)$ with $\mathbf{d}_k = T\mathbf{p}_k - \boldsymbol{\mu}_{i(k)}$ is (up to mixture
   weights) the log-likelihood of the scan under a Gaussian-mixture map — a smooth, correspondence-free
   objective with analytic gradient and Hessian.
   *Sketch:* each cell's density is a local Gaussian fit of target points; independence across beams
   (Ch. 10's assumption, same caveats) gives the product → sum of exponentials; Newton's method on
   $-\log s$; smoothness kills the discrete correspondence switching that makes ICP's cost piecewise.
   *Collapsible:* gradient/Hessian entries, cell-boundary discontinuity mitigation (overlapping grids),
   and the relation to likelihood-field matching from Ch. 10 (NDT = likelihood field with anisotropic, learned-per-cell kernels).
5. **The pose-graph residual on SE(2).**
   *Statement:* a loop or odometry measurement $Z_{ij}$ contributes $\mathbf{e}_{ij} = \log(Z_{ij}^{-1} T_i^{-1} T_j)^\vee$ with cost
   $\sum_{(i,j)} \rho\big(\mathbf{e}_{ij}^\top \mathbf{\Omega}_{ij} \mathbf{e}_{ij}\big)$ — a factor graph containing only pose variables.
   *Sketch:* relative-pose likelihood in the tangent space; $\log/\vee$ turns group error into a
   vector; Jacobians w.r.t. $T_i, T_j$ via the adjoint; plug into Ch. 15's optimizer unchanged.
   *Collapsible:* right-Jacobian expressions (reference Appendix C), and why $\mathbf{\Omega}_{ij}$ from ICP's
   Hessian is optimistic (correlated beams → inflate per Ch. 10 §practical).

### Named algorithms

| Algorithm | Signature | Complexity |
|---|---|---|
| `icp_point_to_point` | $(\mathcal{P}, \text{map}, T_0, \tau) \to (\hat T, \text{rmse}, \text{inliers})$ | $O(I \cdot N)$ with voxel-hash NN; $O(I N \log M)$ with a k-d tree |
| `icp_point_to_plane` | same, plus normals $\{\mathbf{n}_k\}$ | same order; ~3–5× fewer iterations $I$ in corridors |
| `build_ndt` / `ndt_align` | $(\mathcal{Q}, \text{cell}) \to \{(\boldsymbol{\mu}_i, \boldsymbol{\Sigma}_i)\}$; $(\mathcal{P}, \text{ndt}, T_0) \to \hat T$ | build $O(M)$; align $O(I \cdot N)$ Newton steps |
| `kiss_icp_register` | $(\mathcal{P}_t, \text{local map}, T_{t-1}, T_{t-2}) \to \hat T_t$ | constant-velocity predict → deskew → voxel downsample → adaptive-$\tau$ ICP; $O(N)$ per iteration |
| `detect_loop_candidates` | $(\text{graph}, t) \to \{j\}$ | covariance-gated radius search over past poses, $O(|V|)$ (descriptor pointers: Scan Context) |
| `verify_loop` | $(\mathcal{P}_t, \text{submap}_j, T_0) \to \text{Option}(Z_{tj}, \mathbf{\Omega}_{tj})$ | coarse correlative grid search then ICP refine; accept iff fitness ∧ χ² gate pass |
| `pose_graph_slam` | streaming loop: register → node/factor insert → detect/verify → Ch. 15 `optimize` on closure | optimize $O(|E| \cdot \text{fill})$ via sparse Cholesky (`faer`) |

Historical table, cited as ancestry only: `incremental_ML_mapping(o, a, s, m)` and
`incremental_ML_mapping_for_cycles` (Thrun draft Tables 14.1–14.2): gradient ascent on
$p(z_t \mid s_t)\,p(s_t \mid u_t, \hat s_{t-1})$ is scan-to-map matching with a likelihood-field cost;
"correcting poses backwards in time" is pose-graph optimization without the graph.

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w16.1: ICP Stepper** *(flagship; interactive sim)* — Two scans of an Apartment corner:
  target in **blue** (prior map), source in **orange** (predicted pose). Reader drags the source
  cloud's initial offset, then steps ICP iteration by iteration with a scrubber: correspondence
  lines draw in **green**, the aligned cloud settles into **purple**, ground truth outline in
  **gray dashed**. Toggles: point-to-point vs point-to-plane; show/hide the objective-value
  sparkline (`egui_plot`). Autoplays a well-conditioned convergence on load; one meaningful
  parameter (initial offset, via drag). *Misconception killed:* "ICP just works" — dragging the
  start beyond the basin makes it confidently converge to the wrong corner; point-to-plane visibly
  slides along walls instead of crabbing across them. Static fallback: 4-frame convergence filmstrip.
- **Widget w16.2: Loop Snap** *(flagship; interactive sim)* — Rusty drives a fixed lap of the
  Apartment. The estimated trajectory (**orange**, dead-reckoned by scan-to-scan ICP) drifts from
  ground truth (**gray dashed**); the pose graph draws as nodes and odometry factors. When Rusty
  re-enters the start room, a candidate loop factor appears in **green**; on "Close loop" (or
  autoplay timer) the Ch. 15 optimizer runs and the whole trajectory rubber-bands into **purple**,
  correction visibly propagating *backwards* through every past pose, and the stitched submap
  redraws sharp. Slider: odometry noise scale. *Misconception killed:* "loop closure fixes where I
  am now" — it re-writes the entire history, and the map with it. Fallback: before/after pair.
- **Widget w16.3: Score Terrain** *(supporting; animation + hover)* — The registration objective as
  a heatmap over $(x, y)$ translation offsets (θ fixed): ICP's piecewise, plateau-ridden cost next
  to NDT's smooth basin, same scan pair. Hover to read values; button flips to a corridor scene
  where both terrains develop a rank-deficient valley (degeneracy). *Misconception killed:* "the
  cost landscape is a nice bowl" — and it previews why degenerate corridors need motion priors.
- **Widget w16.4: RustSLAM-2D Control Room** *(chapter dashboard; interactive sim)* — the
  integration-lab widget. Layout sketch: left 2/3 = world view (submaps as translucent occupancy
  patches in their own frames, graph overlaid, Rusty live); right 1/3 stacked panes = current-scan
  registration inset (w16.1 in miniature), drift-vs-truth plot, and an event log ("loop 42→7
  verified, χ² = 4.1, optimizing… ΔATE −0.31 m"). Buttons: pause, kidnap-free replay, seed.
  Autoplays the full lap. *Misconception killed:* SLAM as monolith — the reader watches front-end
  and back-end take turns.

## 5. Practical (P) — Rust Implementation

Crates: `nalgebra` 0.35 (poses, $3{\times}3$ normal equations, SVD via `Matrix2::svd`), `parry2d` 0.30
(simulated LiDAR rays; point-in-map queries), `petgraph` 0.8 (pose-graph topology + candidate
radius search), `faer` 0.24 (sparse Cholesky inside the Ch. 15 optimizer), `rand`/`rand_distr`
0.9/0.6 seeded `Pcg64` (reproducible noise), `egui`/`eframe` 0.35 + `egui_plot` 0.34 (widgets),
`plotters` (static figures), `rerun` 0.26 *(optional, native-only: log the full SLAM run to `.rrd`
for the pinned-viewer replay embed)*.

Module plan: `crates/ch16_slam2d/` — `scan.rs`, `voxel_map.rs`, `icp.rs`, `ndt.rs`,
`frontend.rs` (KISS-style odometry), `submap.rs`, `loops.rs`, `slam.rs` (the system);
demo crates `demos/ch16-icp-stepper/`, `ch16-loop-snap/`, `ch16-rustslam2d/`.
Reuses: `sim` (Ch. 4 worlds + LiDAR), the SE(2) type from Ch. 3, `OccGrid` (Ch. 13), and the
graph optimizer crate from Ch. 15.

```rust
use nalgebra::{Matrix2, Point2, Vector2};

/// Built in `scan.rs` by projecting a `sim::Scan`'s ranges (Ch. 4) into robot-frame points.
pub struct PointCloud { pub points: Vec<Point2<f64>>, pub stamp: f64 }

/// KISS-ICP-style local map: voxel hash grid with bounded points per cell.
pub struct VoxelMap { cell: f64, max_per_cell: usize, cells: hashbrown::HashMap<(i32, i32), smallvec::SmallVec<[Point2<f64>; 8]>> }
impl VoxelMap {
    pub fn insert(&mut self, pts: impl Iterator<Item = Point2<f64>>);
    pub fn nearest(&self, p: &Point2<f64>, r_max: f64) -> Option<Point2<f64>>; // O(1) expected
}

pub struct IcpConfig { pub max_iters: usize, pub tau: f64, pub variant: IcpVariant, pub kernel: Kernel }
pub enum IcpVariant { PointToPoint, PointToPlane }
pub struct IcpResult { pub pose: SE2, pub rmse: f64, pub inliers: usize, pub trace: Vec<SE2> } // trace powers w16.1

pub fn icp(src: &PointCloud, map: &VoxelMap, init: SE2, cfg: &IcpConfig) -> IcpResult;
pub fn svd_align(src: &[Point2<f64>], tgt: &[Point2<f64>]) -> SE2;            // Derivation 1, verbatim

pub struct KissOdometry { map: VoxelMap, last: SE2, prev: SE2, tau: AdaptiveThreshold }
impl KissOdometry { pub fn register(&mut self, scan: &PointCloud) -> SE2 }    // predict→deskew→downsample→icp

pub struct Submap { pub origin_node: NodeIx, pub grid: OccGrid }               // Ch. 13 grid, local frame
pub struct Slam2d { odom: KissOdometry, graph: PoseGraph, submaps: Vec<Submap>, cfg: SlamConfig }
impl Slam2d {
    /// One SLAM tick: returns what the dashboard renders.
    pub fn step(&mut self, scan: &PointCloud) -> SlamReport;                   // odometry factor + maybe loop + maybe optimize
}
pub struct SlamReport { pub pose: SE2, pub loop_event: Option<LoopEvent>, pub ate_rmse: Option<f64> }
```

Worked end-to-end example (`cargo run --example rustslam2d_apartment`): seed `0xC0FFEE`, 240-beam
LiDAR at 10 Hz, one ~55 m lap. Expected output (reproduced by a unit test): raw odometry endpoint
error ≈ 1.8 m / 9°; KISS-style ICP odometry endpoint error ≈ 0.35 m; one loop closure verified
(χ² gate at 0.95 quantile); post-optimization ATE RMSE ≈ 0.06 m. The example writes the
before/after trajectory SVG via `plotters` and (natively) a `.rrd` replay. The WASM artifact is
w16.4: the same `Slam2d` struct, compiled to the browser, running the same lap live at 60 fps.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w16.1 | ICP Stepper | wasm-sim | eframe, egui_plot, ch16_slam2d | drag initial offset; step/scrub iterations; variant toggle | ICP mechanics, basins of convergence, point-to-plane advantage |
| w16.2 | Loop Snap | wasm-sim | eframe, ch16_slam2d, ch15 optimizer, faer | drive/replay lap; close-loop button; noise slider | drift, loop factors, correction propagating through history |
| w16.3 | Score Terrain | animation (precomputed) + hover | eframe, egui_plot | hover cost values; scene flip button | ICP vs NDT objective smoothness; corridor degeneracy |
| w16.4 | RustSLAM-2D Control Room | wasm-sim (dashboard) | eframe, egui_plot, ch16_slam2d full stack | pause, seed, replay; inspect event log | front-end/back-end architecture as one live system |
| f16.1 | Front-end/back-end architecture | static-svg | plotters | — | data flow: scans → odometry factors → graph → optimized map |

## 7. Exercises & Extensions

1. **[F]** Complete Derivation 1: prove $\mathrm{tr}(\mathbf{R}^\top\mathbf{W}) \le \sum_i s_i$ with equality at $\mathbf{R} = \mathbf{U}\mathbf{V}^\top$, and construct a 2-point scan pair where omitting the $\det$ correction returns a reflection.
2. **[F]** Derive the point-to-plane row $\mathbf{a}_k, b_k$ from Derivation 3 and show the $3{\times}3$ normal matrix is singular exactly when all normals are parallel (the infinite corridor). What does the null vector mean physically?
3. **[C]** In w16.1, predict the largest pure-rotation initial offset from which point-to-point ICP still converges on the corner scene; then measure it. Repeat for point-to-plane and explain the difference using w16.3's terrains.
4. **[C]** In w16.2, set odometry noise to maximum and predict whether the χ² gate will accept the loop candidate before optimization. Verify, then explain how gating uses the graph's marginal covariance (Ch. 15 §sparsity).
5. **[P]** Implement trimmed ICP (`cfg.trim: f64` dropping the worst fraction of correspondences) and show it survives 20% simulated dynamic-obstacle points where vanilla ICP biases.
6. **[P]** Add a second loop-detection channel: a 64-bin range histogram descriptor per scan with brute-force matching. Measure precision/recall against covariance-gated search on the Apartment lap; discuss why production systems (Scan Context, DBoW) use descriptors at scale.

## 8. Modernization Notes

- The 2005 book (and the 1999–2000 draft this project holds) has **no registration chapter**: the
  draft's Ch. 14 does scan-to-map alignment by gradient ascent on the Ch. 5/6 model likelihoods and
  patches loops with ad-hoc backwards correction. We keep its two *ideas* — incremental ML
  alignment and retrospective correction — and replace their machinery with ICP/NDT front-ends and
  the Ch. 15 factor-graph back-end, which is how every production 2D system (Cartographer, SLAM
  Toolbox) works today.
- KISS-ICP is chosen over LOAM-style feature pipelines deliberately: it is small, nearly
  parameter-free, state-of-the-art in spirit, and implementable by a reader in a weekend. LOAM,
  LIO-SAM, FAST-LIO2 are described in a half-page lineage box (FAST-LIO2 cross-referenced to
  Ch. 7's iterated/error-state EKF as "filtering's revenge").
- Dropped from the draft: Ch. 14's multi-robot mapping (deferred to a pointer; the fusion idea
  reappears in Ch. 15's information form) and its 3D structural mapping (superseded by Ch. 19).
  Dropped from the modern menu: branch-and-bound multi-resolution loop search (Cartographer's
  exact matcher) — we teach covariance-gated candidates + correlative verification instead and
  cite the exact method; deskewing is implemented in its simplest constant-velocity form.
- Honesty note carried through: ICP's per-point independence and the resulting overconfident
  $\mathbf{\Omega}_{ij}$ get the same "inflate and admit it" treatment the beam model got in Ch. 10.
