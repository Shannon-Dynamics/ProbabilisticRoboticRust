# Chapter 14 — The SLAM Problem and EKF SLAM

> Part V — Mapping and SLAM · Estimated length: 9 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Ch. 12 localized against a known map; Ch. 13 mapped from known poses. Neither assumption survives
contact with a real building — that circularity *is* the SLAM problem, the field's crown jewel.
This chapter solves it the first way the field did: put the map *into* the state and run the EKF.
The reader's "aha" is that the resulting filter is not "localization plus mapping" — it is the
off-diagonal blocks of $\Sigma_t$. **Correlations are the map's memory**: observing one landmark
improves all of them, and a loop closure snaps the entire web tight in a single update. Then the
chapter performs an autopsy on its own hero: quadratic cost makes EKF SLAM unaffordable, and
linearization points frozen at first sight make it *provably overconfident* — the two fatal flaws
that motivate everything in Ch. 15–17. The reader leaves loving the idea and distrusting the filter.

Story line:
1. **Hook** — Rusty in an unknown apartment: localize with what map? Map with what poses? Show
   dead-reckoned mapping shearing the Apartment into a spiral (autoplay).
2. **Idea** — one state to rule them both: $y_t = (x_t, m)$; the EKF already knows what to do.
3. **Play** — w14.1 Correlation Web autoplay to first loop closure: the snap.
4. **Formalism** — online vs. full SLAM; prediction with projection matrices; landmark birth;
   the dense Kalman gain; why the off-diagonals do the work.
5. **Reality** — unknown correspondence: ML association, gating, provisional landmarks, map
   management.
6. **Autopsy** — w14.2: cost curve measured live (quadratic), NEES escaping its χ² band
   (inconsistency); the mechanism: Jacobians evaluated at wrong, frozen linearization points.
7. **Implementation & lab** — `EkfSlam` with dynamic state growth; covariance-movie export;
   integration lab on the Apartment landmark course.
8. **Bridge** — "the filter cannot revise its past linearizations; what if we kept the past and
   re-linearized it?" → Ch. 15.

## 2. Prerequisites & Position

- **Builds on:** Ch. 6 (KF machinery, covariance blocks); Ch. 7 (EKF, Jacobian discipline,
  on-manifold caveats revisited in the autopsy); Ch. 9 (velocity motion model and its Jacobians);
  Ch. 10 (range-bearing-signature landmark model, correspondence variable $c_t$); Ch. 11 (ML data
  association, Mahalanobis gating, validation regions — reused wholesale); Ch. 13 (its binary
  log-odds filter returns as landmark *existence* evidence).
- **Feeds into:** Ch. 15 (the two flaws motivate smoothing; the exported dataset is re-optimized
  there); Ch. 16 (pose-graph SLAM as the practical successor); Ch. 17 (FastSLAM factors the same
  posterior differently); Ch. 18 (MSCKF as EKF-SLAM's structureless descendant).
- **Baseline sources:** Thrun et al. (1999–2000 draft) Ch. 10 §10.1–10.6 — Tables 10.1
  (`EKF_SLAM_known_correspondences`), 10.2 (`EKF_SLAM`); Ch. 7 §7.5–7.6 (EKF localization +
  correspondence machinery this chapter extends); Ch. 6 §6.6 (landmark model). Modernization set:
  EKF-SLAM consistency literature (Julier & Uhlmann 2001; Bailey et al. 2006; Huang & Dissanayake
  observability analyses) and the modern verdict that EKF SLAM is pedagogy, not practice.

## 3. Foundation (F) — Mathematical Core

### 3.1 Notation introduced (chapter-scoped table)

| Symbol | Meaning |
|---|---|
| $y_t = (x_t^\top, m_1^\top, \dots, m_{N_t}^\top)^\top$ | joint SLAM state, $\dim = 3 + 2N_t$ (+1 per landmark if signatures kept) |
| $p(x_t, m \mid z_{1:t}, u_{1:t})$ | **online** SLAM posterior |
| $p(x_{0:t}, m \mid z_{1:t}, u_{1:t})$ | **full** SLAM posterior (star of Ch. 15) |
| $\Sigma_{xx}, \Sigma_{xm}, \Sigma_{mm}$ | pose, pose–map, map–map blocks of $\Sigma_t$ |
| $F_x, F_{x,j}$ | sparse projection matrices selecting the pose (resp. pose + landmark $j$) subspace |
| $N_t$ | current landmark count |
| $\pi_k$ | Mahalanobis association distance to landmark $k$ (note: unrelated to the policy $\pi$ of Part VI) |
| $\chi^2_{new}$ | new-landmark threshold (Thrun's Table-10.2 "$\alpha$", renamed to avoid clashing with motion noise $\alpha_{1..6}$) |
| $\epsilon_t$ | NEES: $(x_t - \mu_{t,x})^\top \Sigma_{xx}^{-1} (x_t - \mu_{t,x})$ — the consistency instrument |

### 3.2 Definitions

- **SLAM problem**, online and full variants; relation
  $p(x_t, m \mid z_{1:t}, u_{1:t}) = \int \cdots \int p(x_{0:t}, m \mid z_{1:t}, u_{1:t})\, dx_0 \cdots dx_{t-1}$:
  the online posterior is the full posterior with the past *marginalized out* — phrased this way
  deliberately so Ch. 15 can say "smoothing = refuse to marginalize."
- **Loop closure**: re-observation of a landmark after an excursion during which pose uncertainty
  grew — defined operationally here, dramatized by w14.1.
- **Consistency** of an estimator: $\mathbb{E}[\epsilon_t] = \dim(x_t)$; an inconsistent filter
  reports less uncertainty than its actual error warrants. NEES and its χ² acceptance band are
  defined here (modern addition; the draft baseline had no consistency instrument at all).
- **Correspondence** $c_t^i$, known vs. unknown; **provisional landmark** (candidate awaiting
  promotion); **landmark existence log odds** (Ch. 13's binary filter, retargeted).

### 3.3 Key derivations

**D1 — EKF SLAM prediction via projection matrices.**
*Statement:* motion changes only the pose block:
$\bar\mu_t = \mu_{t-1} + F_x^\top\, \delta(u_t, \mu_{t-1,\theta})$,
$\;G_t = I + F_x^\top g_t F_x$, $\;\bar\Sigma_t = G_t \Sigma_{t-1} G_t^\top + F_x^\top R_t F_x$,
and exploiting the block structure costs $O(N_t)$, not $O(N_t^2)$.
*Sketch (4 steps):* (1) lift the Ch. 9 velocity model to $y_t$ with $F_x = (I_3 \;\, 0)$;
(2) compute the Jacobian's block form; (3) multiply blockwise: only $\Sigma_{xx}$ and the strip
$\Sigma_{xm}$ change — the map–map block is untouched; (4) observe that the cross-covariances are
*rotated, not erased* by motion: the map remembers the robot through them. *Collapsible:* the full
blockwise expansion and the operation count.

**D2 — Landmark initialization.**
*Statement:* first observation of feature $(r, \phi)$ births
$\mu_{j} = \big(\bar\mu_{t,x} + r\cos(\phi + \bar\mu_{t,\theta}),\; \bar\mu_{t,y} + r\sin(\phi + \bar\mu_{t,\theta})\big)^\top$
with covariance propagated through the inverse measurement function.
*Sketch (3 steps):* inverse measurement function; Jacobians $G_x$ (w.r.t. pose) and $G_z$ (w.r.t.
measurement) give new blocks $\Sigma_{jj} = G_x \Sigma_{xx} G_x^\top + G_z Q_t G_z^\top$ and
$\Sigma_{jx} = G_x \Sigma_{xx}$ (plus cross-strips); note Thrun's Table 10.1 instead uses
mean-init + a huge diagonal prior and lets the first update do this — show both, prove they agree
in the limit. *Collapsible:* the limit argument.

**D3 — The dense gain, or why observing one landmark improves all.**
*Statement:* the update for landmark $j$ has $H_t = h_t F_{x,j}$ touching only 5 state dims, yet
$K_t = \bar\Sigma_t H_t^\top S_t^{-1}$ is **dense**: every landmark correlated with the pose moves.
*Sketch (4 steps):* (1) per-landmark $\hat z_t$, $S_t = H_t \bar\Sigma_t H_t^\top + Q_t$;
(2) $K_t$'s rows are $\bar\Sigma_t$'s columns through $H_t^\top$ — the cross-covariance strip
fans the innovation out to the whole state; (3) covariance update $\Sigma_t = (I - K_t H_t)\bar\Sigma_t$
touches *every* entry: $O(N_t^2)$ per observation, and there is no honest way around it in
moments form; (4) loop closure = this mechanism with a large, highly-informative innovation.
*Collapsible:* full derivation from the Ch. 6 KF equations lifted to the joint state.

**D4 — Correlations are the map (the Dissanayake-style structure theorems).**
*Statement, three facts for linear-Gaussian SLAM:* (i) the determinant of any landmark submatrix
of $\Sigma_{mm}$ is monotonically non-increasing; (ii) landmark–landmark correlations grow toward
1 as observations accumulate; (iii) in the limit, every landmark's covariance is bounded below by
the *initial* vehicle covariance — absolute accuracy is capped by how well you knew where you
started; only *relative* map structure becomes certain.
*Sketch (4 steps):* observations add information (never remove it); all information arrives
through the robot, so it is common-mode across landmarks; common-mode error is unobservable from
relative measurements; hence the shared floor and the correlation saturation. *Collapsible:* the
full 1D linear SLAM proof of (iii) — one robot coordinate, one landmark, exact KF algebra, four
lines; this is also the chapter's hand-checkable micro-example (§3.4).

**D5 — ML correspondence, gating, and map management (draft §10.3).**
*Statement:* for each observation compute $\pi_k = (z_t - \hat z_t^k)^\top (S_t^k)^{-1} (z_t - \hat z_t^k)$
over all $k \le N_t$; associate to $\arg\min_k \pi_k$ if $\min_k \pi_k < \gamma_{gate}$; create a
provisional landmark if $\min_k \pi_k > \chi^2_{new}$; promote provisionals after $n_{prom}$
consistent sightings; retire landmarks whose existence log odds sinks below a floor.
*Sketch (4 steps):* likelihood of $z_t$ under candidate $k$ is Gaussian → log-likelihood is
Mahalanobis + log-det; gate from the χ² inverse CDF (via `statrs`); provisional list keeps
clutter from ever entering $\Sigma_t$ (cheap insurance against the $O(N^2)$ cost of a mistake);
existence evidence is exactly a Ch. 13 binary Bayes filter per landmark. *Collapsible:* why
min-Mahalanobis is ML under equal priors, and the log-det correction when gates have unequal $S$.

**D6 — The two fatal flaws.**
*Statement (a), cost:* each observation update is $\Theta(N^2)$ time and $\Sigma_t$ is $\Theta(N^2)$
memory; a 10⁵-landmark city map needs ~80 GB for $\Sigma$ alone.
*Statement (b), inconsistency:* EKF SLAM is provably overconfident: Jacobians are evaluated at
estimated states, the heading error feeds a wrong rotation into every cross-covariance, and — the
structural sin — a linearization, once applied, is **frozen into $\Sigma_t$ forever**; the filter
has no mechanism to revise its past.
*Sketch (5 steps):* (1) count block operations in D3 → quadratic; (2) heading error $\tilde\theta$
rotates predicted landmark displacements by $\approx \tilde\theta$; (3) the filter's linear model
mistakes part of this rotation for fresh metric information → spurious covariance reduction;
(4) Monte Carlo NEES rises above the χ² band while reported $\Sigma$ shrinks — overconfident and
wrong (this is w14.2's script); (5) name the era-defining literature (Julier & Uhlmann 2001;
Bailey et al. 2006; observability analyses) and the two exits: better-invariant linearization
(Ch. 7's on-manifold/IEKF thread, partial fix) and re-linearizable smoothing (Ch. 15, real fix).
*Collapsible:* the standard stationary-robot counterexample where repeated observation of a single
landmark reduces heading uncertainty that is provably unobservable.

### 3.4 Named algorithms

| Algorithm (Thrun table) | Signature | Complexity |
|---|---|---|
| `EKF_SLAM_known_correspondences` (T10.1) | $(\mu_{t-1}, \Sigma_{t-1}, u_t, z_t, c_t) \to (\mu_t, \Sigma_t)$ | predict $O(N)$; $O(N^2)$ per observation update |
| `EKF_SLAM` (T10.2) | $(\mu_{t-1}, \Sigma_{t-1}, N_{t-1}, u_t, z_t) \to (\mu_t, \Sigma_t, N_t)$ | + association: $O(N)$ candidate gates per observation (2×2 solves) |
| `promote_provisional` / `retire_landmark` (§10.3.3, no table) | candidate list + existence log odds → state surgery (grow/delete blocks) | $O(N)$ amortized per event |

Numeric micro-example (unit-tested): pose known exactly, landmark first seen at $r = 2$, $\phi = 0$
with $\sigma_r = 0.1$, $\sigma_\phi = 0.05$: init at $(2, 0)$ with
$\Sigma_{jj} = \operatorname{diag}(0.01, 0.01)$ (since $r^2\sigma_\phi^2 = 0.01$); a second
identical observation halves it to $\operatorname{diag}(0.005, 0.005)$. Then switch on a pose
prior $\Sigma_{xx} \ne 0$ and watch the D4(iii) floor appear: $\Sigma_{jj}$ converges to
$G_x \Sigma_{xx} G_x^\top$, not to zero. All three numbers are hand-checkable.

## 4. Conceptual (C) — Intuition & Visual Design

One metaphor carried end-to-end: **the map as a web** — landmarks are knots, correlations are
threads, the robot is the spider that spun them; pull one knot (a loop closure) and the whole web
moves.

- **Widget w14.1: Correlation Web** *(flagship, TOC name)* — type: wasm-sim, linked-view
  dashboard. Layout: left = Apartment landmark course (Rusty, purple posterior ellipses per
  landmark, gray-dashed ground truth, green measurement rays); threads drawn between landmark
  pairs with opacity/width ∝ |correlation coefficient|; right = live $\Sigma_t$ heatmap with
  visible block grid; bottom = timeline scrubber. Hover a landmark ⇒ its row/column highlights in
  the heatmap (the same object in two views — the distill linked-view pattern). Autoplay: recorded
  run to the first loop closure — **the snap**: every ellipse contracts in one frame, threads pull
  taut, off-diagonal blocks flare. **One headline parameter:** measurement noise $\sigma_r$.
  Killer control: a **"diagonal-only ablation" toggle** that zeroes the off-diagonal blocks every
  step — with it on, the loop closure improves only the re-observed landmark and the map never
  heals. Misconception killed: "each landmark could have its own little filter" / "off-diagonals
  are bookkeeping" — the off-diagonals *are* the map's value. Static fallback: three stills
  (pre-closure, closure frame, post) + covariance heatmap pair.
- **Widget w14.2: Consistency Autopsy** *(flagship, TOC name)* — type: wasm-sim.
  Long-corridor loop scenario. Panels: map + trajectory (estimate vs. gray-dashed truth); NEES
  chart with 95 % χ² acceptance band; heading σ vs. true heading error. Autoplay: run until NEES
  escapes the band, then freeze on the "moment of death" with a scrubber to rewind. **One headline
  parameter:** loop length. Toggle: single run ↔ 50-run Monte Carlo mean NEES (precomputed,
  shipped as data — the honest instrument). Misconception killed: "small reported covariance means
  a good estimate" and "more observations always help" — watch the filter grow *confident and
  wrong simultaneously*. Static fallback: the NEES-escape figure, annotated.
- **Widget w14.3: Growth Meter** — type: wasm-sim (supporting). The state vector and $\Sigma_t$
  drawn as a growing block matrix as Rusty discovers landmarks; live per-update wall-time counter;
  plot of ms/update vs. $N$ with a quadratic fit overlaid; extrapolation readout ("at $N = 10^5$:
  ~80 GB, minutes per update"). Misconception killed: "computers are fast, $N^2$ is fine."
- **Widget w14.4: Poisoned Web** — type: wasm-sim (supporting). Unknown correspondence with a
  clutter-rate slider; watch one wrong association (overlapping gates, reusing Ch. 11's gate
  visuals) warp the entire web permanently; toggle the provisional list to see most false births
  prevented. Misconception killed: "association errors average out" — in a filter they are
  structural and forever.

Dashboard: the chapter integration lab is w14.1 running **live** on a driveable Rusty (arrow
keys), with w14.3's cost meter docked underneath — beauty and doom on one screen. All widgets
autoplay, seeded, one headline parameter each, static fallbacks rendered at build time.

## 5. Practical (P) — Rust Implementation

Crates:
- `nalgebra` 0.35 — `DVector<f64>`/`DMatrix<f64>` for the *growing* joint state. The text makes
  the design beat explicit: Ch. 6's `Kf<const N, const M>` const-generic sizing is impossible
  here because $N_t$ grows at runtime — the type system just taught us something about SLAM.
- `rand` 0.9 + `rand_distr` 0.6 — seeded `SmallRng` simulation noise.
- `statrs` 0.19 — χ² inverse CDF for $\gamma_{gate}$, $\chi^2_{new}$, and NEES bands.
- `crates/motion` (Ch. 9) and `crates/sensor` (Ch. 10) — models and Jacobians, reused not rewritten.
- `serde` + `bincode` — the **covariance movie** export the widgets replay.
- `egui`/`eframe` 0.35 + `egui_plot` 0.34 — demos; `plotters` — static NEES figures.

Module plan:

```text
crates/ch14_ekfslam/
  src/lib.rs        — EkfSlam, SlamConfig, state indexing helpers
  src/predict.rs    — D1 blockwise prediction (O(N))
  src/correct.rs    — D3 per-landmark updates; D5 association + gating
  src/manage.rs     — provisional list, promote/retire (state surgery)
  src/movie.rs      — CovMovie / CovSnapshot capture + serde
  src/nees.rs       — consistency instrumentation
  examples/loop_run.rs        examples/autopsy_mc.rs
  tests/micro.rs              tests/quadratic_cost.rs
demos/ch14-correlation-web/  — w14.1, w14.3 (replays movies; live mode drives EkfSlam in WASM)
demos/ch14-autopsy/          — w14.2, w14.4
```

Key types & signatures (compiles-in-spirit):

```rust
use nalgebra::{DMatrix, DVector, Matrix2, Matrix3};
use motion::VelocityCmd;       // Ch. 9
use sensor::Feature;           // Ch. 10: f = (r, φ[, s])

pub struct SlamConfig {
    pub r_noise: Matrix3<f64>,      // R_t
    pub q_noise: Matrix2<f64>,      // Q_t
    pub gate_chi2: f64,             // γ_gate  (statrs χ² quantile, 2 dof)
    pub new_lm_chi2: f64,           // χ²_new  (Thrun's α, renamed)
    pub promote_after: u32,
    pub retire_below: f64,          // existence log odds floor
}

pub struct EkfSlam {
    mu: DVector<f64>,               // [x y θ | m1x m1y | m2x m2y | …]
    sigma: DMatrix<f64>,
    n_landmarks: usize,
    provisional: Vec<Provisional>,
    existence: Vec<f64>,            // Ch. 13's binary filter, per landmark
    pub cfg: SlamConfig,
}

pub enum Association { Matched { lm: usize, nis: f64 },
                       Provisional(usize), New(usize), Rejected { nis: f64 } }

impl EkfSlam {
    pub fn predict(&mut self, u: &VelocityCmd);                             // Table 10.1 ll.2–5, O(N)
    pub fn correct(&mut self, obs: &[Feature]) -> Vec<Association>;         // Table 10.2, O(N²)/obs
    fn init_landmark(&mut self, z: &Feature) -> usize;                      // grows mu, sigma (D2)
    pub fn retire(&mut self, lm: usize);                                    // deletes blocks
    pub fn snapshot(&self, t: f64) -> CovSnapshot;                          // for the movie
    pub fn nees(&self, truth: &pr_core::geom::SE2) -> f64;
}

pub struct CovMovie { pub frames: Vec<CovSnapshot> }
pub struct CovSnapshot { pub t: f64, pub mu: Vec<f64>,
                         pub sigma_lower: Vec<f64>,      // packed lower triangle
                         pub n: usize, pub events: Vec<SlamEvent> }
pub enum SlamEvent { LandmarkBorn(usize), LoopClosure { lm: usize, nis: f64 },
                     Rejected { nis: f64 } }
```

Worked end-to-end example — `cargo run --example loop_run`: 12-landmark Apartment course, 600
steps, seed `42`. Output: `ch14_loop.covmovie` (the file both flagship widgets replay), plus three
plots — map before/after first closure (orange prediction vs. purple posterior vs. gray-dashed
truth), NEES with band, ms/update vs. $N$ with quadratic fit. Expected printed summary:
`N=12 · mean NEES(steps 1–200) ≈ 3.1 (3-dof bound 7.81: consistent) · after 3 laps ≈ 9.4
(inconsistent)`. `tests/micro.rs` reproduces the §3.4 landmark-init numbers exactly;
`tests/quadratic_cost.rs` asserts update time fits $aN^2 + bN + c$ with dominant $a$.

Runnable artifact: the native example produces the covariance movie and figures; the WASM demo
replays movies for the heavyweight Monte-Carlo content and runs `EkfSlam` live (same code) for
the small driveable course — the book's "the widget *is* the code" promise, with precomputation
where 50-run MC would melt a phone.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w14.1 | Correlation Web | wasm-sim | ch14_ekfslam + sim + eframe 0.35 | scrub, hover-link map↔Σ heatmap, σ_r slider, diagonal-ablation toggle, drive mode | correlations are the map; loop closure as global update |
| w14.2 | Consistency Autopsy | wasm-sim | ch14_ekfslam + egui_plot | loop-length slider, single↔MC toggle, death-scrubber | overconfidence; NEES as the honesty instrument |
| w14.3 | Growth Meter | wasm-sim | ch14_ekfslam + egui_plot | play/pause, N readouts | quadratic cost made visceral |
| w14.4 | Poisoned Web | wasm-sim | ch14_ekfslam + sensor + eframe | clutter slider, provisional-list toggle | association errors are permanent in a filter |

## 7. Exercises & Extensions

1. **(F)** Derive the blockwise prediction (D1); list exactly which blocks of $\Sigma_t$ change
   and count operations to establish $O(N)$.
2. **(F)** 1D linear SLAM (robot and one landmark on a line): run the KF by hand for two
   observe–move cycles, then prove the D4(iii) floor — the landmark variance never drops below
   the initial robot variance.
3. **(F)** Show that the covariance update in D3 costs $\Theta(N^2)$ and explain why no moments-
   form bookkeeping can avoid it (every entry of $\Sigma$ is touched through $K_t H_t \bar\Sigma_t$).
4. **(C)** *Predict-then-verify with w14.1:* with the diagonal-only ablation ON, predict what the
   loop closure will do to the far-side landmark ellipses; verify (nothing), and explain via
   $K_t = \bar\Sigma_t H_t^\top S_t^{-1}$ which matrix entries carried the healing before.
5. **(C)** *Predict-then-verify with w14.2:* find the loop length at which single-run NEES first
   escapes the band; explain why the 50-run Monte-Carlo mean is the honest metric and what a
   single lucky run can hide.
6. **(P)** Implement `retire_landmark` end-to-end (existence log odds → block deletion) and
   measure false-landmark count vs. clutter rate with and without the provisional list on the
   w14.4 scenario. **Stretch:** re-run the autopsy with the Ch. 7 error-state on-manifold EKF and
   report how far consistency improves — and that it does not fully heal, foreshadowing Ch. 15.

## 8. Modernization Notes

- **Baseline:** draft Ch. 10 ≈ published 2005 Ch. 10; the algorithmic content (Tables 10.1/10.2,
  provisional landmarks, map management) is preserved intact — this remains the cleanest first
  exposure to joint estimation anywhere in robotics.
- **Added beyond baseline:** the entire consistency methodology — NEES, χ² bands, Monte-Carlo
  averaging — and the post-2001 inconsistency literature (Julier & Uhlmann's counterexample,
  Bailey et al.'s empirical study, observability analyses), which the 1999–2000 draft predates;
  measured quadratic-cost evidence (Growth Meter) rather than a complexity remark; the explicit
  "online = full with the past marginalized" framing that makes Ch. 15 a one-sentence pivot; the
  on-manifold/invariant-EKF remark connecting to Ch. 7's modern thread.
- **Reframed:** per the modernization findings, EKF SLAM is taught as *pedagogy, not practice* —
  the chapter says explicitly that no serious 2026 system runs it, and that its two flaws are the
  founding charter of the factor-graph era.
- **Dropped:** EKF-SLAM scaling band-aids (submap/CEKF/ATLAS families) — one pointer paragraph
  only, since Ch. 15/16 supersede them honestly; MHT-flavored SLAM (historical, gets one line in
  the D5 discussion); landmark signatures $s$ carried as an optional type parameter rather than
  the default state extension.
- **Relocated from the draft:** draft Ch. 13 (EM mapping) and Ch. 14 (incremental ML mapping with
  cycle correction) are ancestors of Ch. 17's data-association discussion and Ch. 16's
  scan-matching/loop-closure pipeline respectively; noted here so readers of the old book can
  find where those ideas went.
