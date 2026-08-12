# Chapter 17 — FastSLAM and Rao-Blackwellization

> Part V — Mapping and SLAM · Estimated length: 12 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Chapters 14–16 attacked SLAM's joint pose-map posterior head-on — first with one big Gaussian, then
with one big least-squares problem. This chapter shows the third road: *don't estimate the joint at
all*. Conditioned on the robot's path, landmarks are independent of each other — so sample the path
with particles and solve each map in closed form. The "aha" is that this is not a hack but a
theorem: **Rao-Blackwellization provably never increases estimator variance**, and it turns an
intractable $(3 + 2N)$-dimensional particle filter into $M$ cheap path-particles each towing a bank
of tiny EKFs (or an occupancy grid). The chapter builds FastSLAM 1.0 and 2.0 for landmarks, then
the grid-based RBPF with improved proposals — the classic *gmapping* recipe — and closes honestly:
in 2026 pose graphs won 2D production SLAM, but per-particle data association and multimodal loop
hypotheses remain FastSLAM's unmatched party trick. A historical aside gives the road not taken:
the draft's Ch. 13 EM mapping, which solved data association in batch and lost to online
Rao-Blackwellization.

Story line:
1. Problem: Ch. 14's EKF SLAM is $O(N^2)$ and unimodal; Ch. 16's graph defers the map; can particles do SLAM directly? A naive PF over $(x_t, m)$ dies instantly (curse of dimensionality, shown numerically).
2. Intuition: "if I knew the path, mapping would be easy" (Ch. 13 said exactly this) — so let each particle *pretend to know* a path.
3. Formalism: the factorization theorem; Rao-Blackwell as variance reduction.
4. Algorithm: FastSLAM 1.0 → its proposal wastes particles → FastSLAM 2.0's measurement-aware proposal.
5. Grids: RBPF mapping with scan-matched proposals and selective resampling ($N_{\mathrm{eff}}$).
6. Implementation & lab: parallel universes racing around the Apartment; loop closure as survival of the fittest universe.
7. Verdict: what died (gmapping in production), what survived (the theorem, per-particle DA), and the comparison table against Ch. 16 on identical logs.

## 2. Prerequisites & Position

- Builds on: Ch. 8 (particle filter, low-variance resampler, deprivation), Ch. 9
  (`sample_motion_model_odometry`), Ch. 10 (landmark model, correspondence $c_t$; likelihood
  fields), Ch. 11 (EKF update per landmark, Mahalanobis gating), Ch. 13 (`OccGrid` + log odds),
  Ch. 14 (the joint-Gaussian failure that motivates factoring), Ch. 16 (scan matching, reused
  inside the improved proposal).
- Feeds into: Ch. 22 (POMCP reuses per-particle world hypotheses), Ch. 24 (exploration under map
  uncertainty), Ch. 26 (capstone's localization fallback modes).
- Baseline sources: **not in the draft PDF** — the 1999–2000 draft predates FastSLAM
  (Montemerlo et al. 2002/2003) entirely; sourced from the modernization set: FastSLAM 1.0/2.0
  (published-2005-edition lineage; Montemerlo & Thrun), grid RBPF with improved proposals and
  selective resampling (Grisetti, Stachniss, Burgard 2007 — *gmapping*), Doucet et al. 2000 for
  Rao-Blackwellized SMC. Historical note sourced from Thrun et al. draft Ch. 13 §13.3–13.5 (EM
  mapping: forward-backward E-step, ML map M-step) as the abandoned batch alternative. Particle
  machinery names from draft Ch. 4 Tables 4.3–4.4 and Ch. 5 Table 5.6.

## 3. Foundation (F) — Mathematical Core

### Definitions & notation introduced

| Symbol | Meaning |
|---|---|
| $x_{0:t}^{[i]}$ | path hypothesis of particle $i$ (the particle *is* a trajectory) |
| $\mu_{j,t}^{[i]}, \Sigma_{j,t}^{[i]}$ | mean/covariance of landmark $j$'s EKF inside particle $i$ (each $2{\times}2$) |
| $m^{[i]}$ | per-particle map (landmark set or occupancy grid) |
| $N_{\mathrm{eff}}$ | effective sample size, $N_{\mathrm{eff}} = 1 / \sum_i (\tilde w_t^{[i]})^2$ (Ch. 8's $M_{\mathrm{eff}}$ — written $N_{\mathrm{eff}}$ here per the gmapping literature) |
| $\pi(x_t \mid \cdot)$ | proposal distribution used to sample the next pose |
| $\hat x_t^{[i]}$ | scan-match optimum used as the improved proposal's center |

### Derivations

1. **The SLAM factorization theorem.**
   *Statement:* with known correspondences,
   $$p(x_{0:t}, m \mid z_{1:t}, u_{1:t}, c_{1:t}) \;=\; p(x_{0:t} \mid z_{1:t}, u_{1:t}, c_{1:t}) \prod_{j=1}^{N} p(m_j \mid x_{0:t}, z_{1:t}, c_{1:t}).$$
   *Sketch:* (i) condition on the full path $x_{0:t}$; (ii) each measurement $z_t$ depends only on
   the pose at its time and the one landmark $c_t$ indexes; (iii) hence the map posterior factors
   over landmarks (d-separation: paths block all landmark-landmark dependence); (iv) induction over
   $t$ exactly as in the Ch. 5 Bayes-filter derivation. The Ch. 14 moral inverted: correlations
   between landmarks exist *only through pose uncertainty* — freeze the path and they vanish.
   *Collapsible:* full induction with normalizers tracked, and the counterexample showing the
   factorization fails for the *online* marginal $p(x_t, m \mid \cdot)$ — why particles must carry paths.
2. **Rao-Blackwellization as a theorem, not a trick.**
   *Statement:* for any estimator $\varphi(X, Y)$ of $\mathbb{E}[\varphi]$, the conditional estimator
   $\mathbb{E}[\varphi \mid X]$ satisfies $\operatorname{Var}\big[\mathbb{E}[\varphi \mid X]\big] \le \operatorname{Var}[\varphi(X, Y)]$; applied to SMC:
   sampling only $x_{0:t}$ and integrating $m$ analytically (EKFs, log-odds grids) gives weights
   with no larger variance than sampling $(x_{0:t}, m)$ jointly.
   *Sketch:* law of total variance $\operatorname{Var}[\varphi] = \mathbb{E}[\operatorname{Var}[\varphi \mid X]] + \operatorname{Var}[\mathbb{E}[\varphi \mid X]]$; the first
   term is $\ge 0$; identify $Y$ with the map, $X$ with the path; conclude fewer particles for the
   same accuracy — the entire chapter in one inequality.
   *Collapsible:* the classical Rao-Blackwell statement via sufficient statistics, and the SMC-specific
   proof that the Rao-Blackwellized importance weights are the marginalized ones (Doucet et al. 2000).
3. **FastSLAM 1.0 weight.**
   *Statement:* sampling $x_t^{[i]} \sim p(x_t \mid x_{t-1}^{[i]}, u_t)$ gives
   $w_t^{[i]} \propto \left|2\pi \mathbf{S}_t^{[i]}\right|^{-1/2} \exp\!\big(-\tfrac{1}{2} \boldsymbol{\nu}^\top (\mathbf{S}_t^{[i]})^{-1} \boldsymbol{\nu}\big)$ with innovation
   $\boldsymbol{\nu} = z_t - \hat z_t^{[i]}$ and $\mathbf{S}_t^{[i]} = \mathbf{H}_t \Sigma_{c_t, t-1}^{[i]} \mathbf{H}_t^\top + \mathbf{Q}_t$.
   *Sketch:* weight = target/proposal (Ch. 8); everything cancels except the measurement
   *marginal* likelihood; integrating out the landmark's Gaussian yields the innovation Gaussian —
   the same $\mathbf{S}_t$ as Ch. 11's gate. Then the observed landmark gets a standard EKF update inside
   the particle; unobserved landmarks are untouched.
   *Collapsible:* the integral $\int p(z_t \mid x_t, m_j)\, p(m_j \mid \cdot)\, dm_j$ done explicitly with
   Appendix B's Gaussian-marginal identity; landmark initialization from the first sighting
   (inverse measurement + Jacobian-propagated covariance).
4. **FastSLAM 2.0's proposal (and why it matters).**
   *Statement:* sampling instead from $\pi = p(x_t \mid x_{1:t-1}^{[i]}, u_{1:t}, z_{1:t})$ — motion prior ×
   current measurement, a Gaussian computable by one EKF-style step — minimizes weight variance
   among proposals using the available information; weights become $w_t^{[i]} \propto \int p(z_t \mid x_t)\, p(x_t \mid x_{t-1}^{[i]}, u_t)\, dx_t$.
   *Sketch:* precise motion + sharp sensor ⇒ 1.0's prior-sampled particles nearly all miss the
   likelihood peak (illustrated in w17.3); linearize the measurement in the pose around the motion
   mean; complete the square → Gaussian proposal; the weight is the *predictive* likelihood, which
   no longer depends on where in the proposal the sample landed.
   *Collapsible:* full Gaussian algebra for proposal mean/covariance; the optimal-proposal theorem
   (Doucet) with proof; degeneracies when the measurement is multimodal.
5. **Improved proposals for grid RBPF (the gmapping recipe).**
   *Statement:* per particle: scan-match $z_t$ against $m^{[i]}_{t-1}$ from the odometry-predicted pose
   to get $\hat x_t^{[i]}$; sample $K$ poses $\{x_k\}$ around it; fit a Gaussian
   $(\mu_t^{[i]}, \Sigma_t^{[i]})$ weighted by $p(z_t \mid m^{[i]}, x_k)\, p(x_k \mid x_{t-1}^{[i]}, u_t)$; sample the new pose from it;
   multiply the weight by the normalizer $\eta^{[i]} = \sum_k (\cdot)$. Resample only when
   $N_{\mathrm{eff}} < M/2$.
   *Sketch:* same optimal-proposal logic as Derivation 4 but with no closed form — replace algebra
   with local sampling around the scan-match mode; selective resampling preserves universe
   diversity exactly when weights are healthy.
   *Collapsible:* derivation of $\eta^{[i]}$ as the weight increment; failure mode when the scan
   matcher's basin misses (fall back to raw motion sampling); parameter table from Grisetti et al.

### Named algorithms

| Algorithm | Signature | Complexity |
|---|---|---|
| `FastSLAM_1_0` | $(\mathcal{X}_{t-1}, u_t, z_t, c_t) \to \mathcal{X}_t$ | $O(M N)$ naive; $O(M \log N)$ with shared landmark trees (we teach trees, implement copy-on-write) |
| `FastSLAM_2_0` | same inputs; measurement-aware proposal | same order + one $2{\times}2$/$3{\times}3$ solve per particle |
| `FastSLAM_unknown_correspondence` | adds per-particle ML gating / new-landmark creation | $O(M N)$ with per-particle $c_t^{[i]}$ — DA becomes a *sampled* variable |
| `gmapping_step` (grid RBPF) | $(\mathcal{X}_{t-1}, u_t, \mathcal{P}_t) \to \mathcal{X}_t$ | $O(M \cdot (\text{ICP} + K \text{ evals} + \text{grid update}))$; memory $O(M \cdot \text{cells})$ unless shared |
| `effective_sample_size` | $\{\tilde w^{[i]}\} \to N_{\mathrm{eff}}$ | $O(M)$; resample iff $N_{\mathrm{eff}} < M/2$ via draft Table 4.4 `Low_variance_sampler` |

Reused by name from the draft: `Particle_filter` (Table 4.3), `Low_variance_sampler` (Table 4.4),
`sample_motion_model_odometry` (Table 5.6), `landmark_model_known_correspondence` (Table 6.4).
Historical: `EM_mapping(d)` (draft Table 13.1) — batch E-step smoothing over poses + M-step ML map;
presented in a half-page box as the road not taken (offline, local-minima-prone, but the first
system to treat correspondence as a latent variable).

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w17.1: Parallel Universes** *(flagship; interactive sim)* — A 3×4 grid of mini-map tiles,
  each one particle's private occupancy map with its path drawn on it (12 shown of $M = 30$;
  a strip shows the rest as thumbnails sorted by weight). Rusty drives the Apartment lap
  (autoplay). Each universe drifts *differently*; tile borders encode normalized weight
  (**green** = likely, fading to gray). At the loop, scans either agree with a universe's map
  (weight surges) or contradict it (fades); a resampling event flashes: dying universes dissolve,
  survivors clone (**purple** border pulse). One meaningful parameter: particle count slider
  $M \in \{5, 30, 100\}$. Ground truth map inset in **gray dashed**. *Misconception killed:* "the
  filter has a map" — there is no *the* map, only competing map-hypotheses, and loop closure is
  natural selection among them, not an optimization step. Static fallback: 12-tile snapshot pre/post loop.
- **Widget w17.2: Depletion Meter** *(supporting; interactive sim)* — $N_{\mathrm{eff}}/M$ as a live strip
  chart (`egui_plot`) under a small world view; resampling events marked as vertical ticks; toggle
  between "resample every step" and "selective ($N_{\mathrm{eff}} < M/2$)"; a diversity gauge counts distinct
  ancestors of the current population. *Misconception killed:* "resampling is free" — always-on
  resampling visibly collapses ancestry to one universe long before the loop, so the loop has
  nothing to select among (connects to Ch. 8's deprivation).
- **Widget w17.3: Proposal Quality** *(supporting; animation with one toggle)* — One prediction
  step, frozen: motion prior cloud in **orange**, measurement likelihood as **green** contours,
  posterior region **purple**. Toggle FastSLAM 1.0 (sample orange, weight by green — most samples
  wasted) vs 2.0/gmapping (sample from the overlap directly). A counter shows the fraction of
  samples landing in the 90% posterior mass. *Misconception killed:* "more particles fix
  everything" — a better proposal beats 10× more particles, measurably.

Dashboard note: w17.1 doubles as the chapter dashboard; w17.2's strip chart docks beneath it in
the integration lab so resampling events and universe extinctions line up on the same time axis.

## 5. Practical (P) — Rust Implementation

Crates: `nalgebra` 0.35 ($2{\times}2$ landmark EKFs as `SMatrix<f64, 2, 2>`, poses), `rand` 0.9 +
`rand_distr` 0.6 (seeded `Pcg64`; low-variance resampler), `statrs` 0.19 (innovation-likelihood
`ln_pdf` cross-checks in tests), `rayon` (native-only feature: per-particle parallelism — the same
code runs single-threaded on WASM), `egui`/`eframe` 0.35 + `egui_plot` 0.34 (w17.1–w17.3),
`plotters` (static fallbacks). Reuses `sim` (Ch. 4), the Ch. 8 particle machinery, Ch. 9 motion
samplers, Ch. 10 landmark/likelihood-field models, `OccGrid` (Ch. 13), and the Ch. 16 ICP matcher
inside the improved proposal.

Module plan: `crates/ch17_fastslam/` — `landmark_ekf.rs`, `fastslam.rs` (1.0/2.0 behind an enum),
`data_assoc.rs` (per-particle ML gating), `grid_rbpf.rs`, `neff.rs`, `cow_map.rs`
(copy-on-write map sharing); demos `demos/ch17-parallel-universes/`, `ch17-depletion/`.

```rust
use nalgebra::{Matrix2, Vector2};
use ch16_slam2d::PointCloud;   // Ch. 16's projected-scan type, fed to the ICP matcher

pub struct LandmarkEkf { pub mu: Vector2<f64>, pub sigma: Matrix2<f64> }

pub struct FsParticle {
    pub pose: SE2,
    pub path: Vec<SE2>,                       // the particle IS a path (factorization theorem)
    pub landmarks: Vec<LandmarkEkf>,
    pub log_w: f64,
}

pub enum Proposal { MotionPrior, MeasurementAware }   // FastSLAM 1.0 vs 2.0

pub struct FastSlam { particles: Vec<FsParticle>, proposal: Proposal, rng: rand_pcg::Pcg64 }
impl FastSlam {
    pub fn step(&mut self, u: &OdomDelta, z: &[Feature], c: Option<&[usize]>) -> FsReport;
    fn weight_1_0(p: &FsParticle, z: &Feature, j: usize) -> f64;        // Derivation 3, verbatim
    fn propose_2_0(p: &FsParticle, u: &OdomDelta, z: &Feature, j: usize) -> (SE2, f64);
}

/// Grid-based RBPF (gmapping recipe). Map shared copy-on-write:
/// clone-on-resample is O(1) until a particle actually writes.
pub struct GridParticle { pub pose: SE2, pub map: std::sync::Arc<OccGrid>, pub log_w: f64 }
pub struct GridRbpf { particles: Vec<GridParticle>, k_samples: usize, neff_threshold: f64 }
impl GridRbpf {
    pub fn step(&mut self, u: &OdomDelta, scan: &PointCloud) -> RbpfReport; // Derivation 5
    fn improved_proposal(&self, p: &GridParticle, u: &OdomDelta, scan: &PointCloud)
        -> (SE2, f64 /* log-weight increment log η */);
}

pub fn effective_sample_size(log_ws: &[f64]) -> f64;
```

The copy-on-write trick (`Arc::make_mut` on first write after cloning) replaces FastSLAM's classic
shared balanced trees with idiomatic Rust and gets the same asymptotic win for grids; the tree
version is described in the text and left as the extension exercise.

Worked end-to-end example (`cargo run --example fastslam_vs_graph`): seed `0xFA57`, two runs —
(a) landmark FastSLAM 2.0, 24-landmark field, $M{=}100$, unknown correspondence: final map RMSE
≈ 0.09 m, zero DA errors vs. Ch. 11's single-hypothesis EKF making 3 on the same log;
(b) grid RBPF, Apartment lap, $M{=}30$, 5 cm cells: loop closes by universe selection with ATE
RMSE ≈ 0.11 m vs. Ch. 16 RustSLAM-2D's ≈ 0.06 m on the identical log, at ~4× the memory — the
printed comparison table is generated by this example and reproduced by a unit test. The WASM
artifact is w17.1 running run (b) live.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w17.1 | Parallel Universes | wasm-sim (dashboard) | eframe, egui_plot, ch17_fastslam, sim | autoplay lap; particle-count slider; pause/scrub; tile inspect | per-particle maps; loop closure as resampling/selection |
| w17.2 | Depletion Meter | wasm-sim | eframe, egui_plot, ch17_fastslam | resampling-policy toggle | $N_{\mathrm{eff}}$, selective resampling, ancestry collapse |
| w17.3 | Proposal Quality | animation + toggle | eframe, ch17_fastslam | 1.0 vs 2.0 proposal toggle | why measurement-aware proposals dominate brute particle count |
| f17.1 | The factorization | static-svg | plotters | — | path samples unlock independent per-landmark posteriors |

## 7. Exercises & Extensions

1. **[F]** Prove the factorization theorem by induction on $t$ (Derivation 1), tracking normalizers, and exhibit where the proof breaks if a single measurement observes two landmarks at once. What does gmapping implicitly assume about scans here?
2. **[F]** Derive FastSLAM 2.0's proposal mean and covariance for the range-bearing model (Derivation 4 collapsible), starting from Appendix B's conditional-Gaussian identity.
3. **[C]** In w17.1 with $M{=}5$, predict whether the loop will close correctly on seed 2; run it, then explain the failure via w17.2's ancestry gauge. Find the smallest $M$ that closes the loop on 9 of 10 seeds.
4. **[C]** Using w17.3, estimate how many 1.0-particles match one 2.0-particle's effective coverage at the default noise settings; verify against the displayed 90%-mass counter.
5. **[P]** Implement per-particle unknown-correspondence DA (`data_assoc.rs`): Mahalanobis gating with per-particle new-landmark creation, and reproduce the "ambiguous corridor of identical doors" experiment where particle diversity solves an association the Ch. 11 EKF cannot.
6. **[P]** Replace `cow_map.rs` with the classic $O(\log N)$ shared landmark tree and benchmark both against $M \in \{10, 100, 1000\}$ on native + WASM; report where the constant factors cross.

## 8. Modernization Notes

- **Absent from the baseline PDF entirely:** the 1999–2000 draft predates FastSLAM; its slot was
  occupied by Ch. 13's batch EM mapping. We keep EM as a one-box history lesson (first explicit
  latent-correspondence treatment; its forward-backward E-step is smoothing, which returned in
  Ch. 15) and teach the online Rao-Blackwellized line that replaced it: FastSLAM 1.0/2.0
  (Montemerlo et al.) and gmapping (Grisetti et al. 2007).
- Contra the 2005 book's ordering, we place FastSLAM *after* graphs: in 2026 pose-graph systems
  (Ch. 16) are the production default and gmapping is legacy — but Rao-Blackwellization is
  permanent mathematics, per-particle DA is still the cleanest solution to ambiguous
  correspondence, and RBPF ideas live on in POMCP (Ch. 22) and multi-hypothesis loop closing.
  The chapter says all of this out loud, with the head-to-head numbers from the worked example.
- Followed the modernization guidance to compress FastSLAM 2.0's implementation minutiae (no
  tree-management pseudocode reproduction; copy-on-write shown instead) while *keeping* its
  derivation, which is the pedagogically valuable part (optimal proposals — reused by the
  gmapping recipe and later by Ch. 23's MPPI sampling story).
- Dropped: DP-SLAM and FastSLAM's landmark-existence evidence bookkeeping (pointer only);
  KLD-adaptive particle counts appear as a cross-reference to Ch. 8 rather than a re-derivation.
