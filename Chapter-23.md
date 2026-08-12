# Chapter 23 — Stochastic Model Predictive Control: MPPI and Friends

> Part VI — Planning and Acting under Uncertainty · Estimated length: 8 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Chapters 21–22 solved decision making on coarse grids and tiny belief spaces; real Rusty needs a
continuous $(v, \omega)$ command every 50 ms, in clutter, around a moving obstacle, while roughly
following Ch. 20's path. The classical answer — track the path with a feedback controller (the
Spong/Craig world) — shatters on obstacles the path didn't know about. The receding-horizon answer:
re-solve a short-horizon optimal control problem every tick and execute only the first control. This
chapter builds that idea from its classical sampled baseline (DWA) to its modern probabilistic form,
MPPI — and lands the punchline the whole book has been setting up: **MPPI is importance sampling
over controls; it is the particle filter's twin.** Rollouts are particles, the cost is a negative
log-likelihood, the exponential weighting is the measurement update, the weighted mean is the
posterior estimate. The reader doesn't learn a new algorithm — they recognize an old friend pointed
at the future instead of the past.

Story line:
1. **Hook:** Rusty tracks the Ch. 20 path with a pure-pursuit controller (autoplay); a chair moved
   since planning — collision. Replanning with RRT* at 20 Hz — too slow, and the path jumps.
2. **Receding horizon (F):** the loop: measure (well — *believe*, via Ch. 12), optimize a horizon,
   execute one step, shift, repeat.
3. **DWA (F):** the classical baseline — search a window of *velocities*, not paths; why its
   one-arc greed dies in dead ends.
4. **Play (C):** Rollout Storm — hundreds of trajectories bloom and collapse into a command.
5. **MPPI derived (F):** free energy, the exponentiated-cost optimal distribution, importance
   sampling with the current plan as proposal — the particle-filter correspondence made exact,
   term by term, in the book color code.
6. **Engineering (F/P):** constraints, temperature, smoothing; `Mppi` in Rust with rayon rollouts;
   ESDF costs from Ch. 19.
7. **Integration lab:** Rusty threads moving clutter at real time in the browser; DWA vs. MPPI in
   the dead-end arena.
8. **Bridge:** when gradients beat samples — an honest MPPI vs. gradient-MPC scorecard; and the
   cliffhanger that the *cost* can include information terms (Ch. 24 makes goals out of them).

## 2. Prerequisites & Position

- **Builds on:** Ch. 8 (importance sampling — the derivation is a re-instantiation), Ch. 9
  (velocity motion model = rollout dynamics; $\alpha$ noise = exploration noise), Ch. 12 (the MCL
  pose estimate the controller consumes), Ch. 19 (ESDF as the obstacle-cost field), Ch. 20
  (reference path), Ch. 21 (cost-to-go/terminal-cost view).
- **Feeds into:** Ch. 24 (the explorer executes its chosen frontier via this controller), Ch. 26
  (capstone control layer).
- **Baseline sources:** entirely modernization set — Williams et al. (ICRA 2016 / T-RO 2018,
  information-theoretic MPC / MPPI); Fox, Burgard & Thrun 1997 (DWA — probabilistic-robotics
  lineage, absent from the draft PDF); Nav2's MPPI controller as deployment evidence. Classical
  context: Spong et al. Ch. 8 (independent joint control, PID; §8.5–8.6 feedforward/computed
  torque) and Ch. 9 (nonlinear control) as "feedback along a given trajectory"; Craig,
  *Introduction to Robotics*, linear/nonlinear control chapters as the cross-reference for the same
  material; Lynch & Park Ch. 13.3.4 (nonholonomic feedback control) for why smooth stabilization of
  diff-drives is delicate (Brockett obstruction, one box).

## 3. Foundation (F) — Mathematical Core

**Notation introduced:**

| Symbol | Meaning |
|---|---|
| $H$ | planning horizon (steps); $\Delta t$ control period |
| $U = (u_0, \dots, u_{H-1})$ | control sequence over the horizon; $u_k = (v_k, \omega_k)^\top$ |
| $x_{k+1} = f(x_k, u_k)$ | nominal rollout dynamics (Ch. 9 velocity model, noise-free) |
| $c(x_k, u_k)$, $\phi(x_H)$ | stage cost, terminal cost |
| $S(U) = \phi(x_H) + \sum_k c(x_k, u_k)$ | trajectory cost of a rollout |
| $\epsilon_k^{(i)} \sim \mathcal{N}(0, \Sigma_u)$ | control perturbation, sample $i$, step $k$ |
| $\lambda$ | temperature (the chapter's one headline parameter) |
| $w^{(i)}$, $\eta$ | rollout weight, normalizer (the book's $\eta$, reused deliberately) |
| $\mathcal{F}(S)$ | free energy $-\lambda \log E_p[\exp(-S/\lambda)]$ |

**Definitions:** receding-horizon control; the dynamic window $V_d$ (velocities reachable under
acceleration limits within $\Delta t$); admissible velocity (stoppable before collision);
temperature; effective sample size of rollouts (imported from Ch. 8, same formula).

**Key derivations:**

1. **DWA as constrained one-step search.** *Statement:* DWA maximizes
   $\sigma(w_h \cdot \text{heading} + w_c \cdot \text{clearance} + w_v \cdot \text{velocity})$ over
   the admissible dynamic window, each candidate $(v, \omega)$ scored along its constant-curvature
   arc. *Sketch (3 steps):* window from acceleration limits; admissibility from stopping distance
   vs. ESDF clearance; grid-evaluate and pick. *Collapsible:* the original admissibility inequality
   and the myopia analysis — constant-curvature arcs cannot represent "back out of a dead end,"
   which the lab demonstrates. Complexity $O(n_v n_\omega H)$.
2. **The optimal control distribution (free energy).** *Statement:* over control sequences with
   prior $p(U)$ (current plan + Gaussian perturbations), the free energy bounds expected cost:
   $\mathcal{F}(S) \le E_q[S(U)] + \lambda\, KL(q \,\|\, p)$ for every distribution $q$, with
   equality iff $q^*(U) = \frac{1}{\eta}\, p(U) \exp(-S(U)/\lambda)$. *Sketch (4 steps):* write
   $\mathcal{F}$ as $-\lambda \log E_p[e^{-S/\lambda}]$; insert $q$ by importance-sampling identity;
   apply Jensen to the log; identify the KL term and the equality condition. *Collapsible:* the
   full information-theoretic derivation (Williams et al. 2018, discrete-time — we deliberately
   avoid the continuous-time Girsanov machinery and say so, with a pointer).
3. **MPPI update = importance sampling toward $q^*$ (the twin theorem).** *Statement:* sampling
   $U^{(i)} = U^{nom} + \epsilon^{(i)}$ from the proposal and weighting with
   $w^{(i)} = \frac{1}{\eta} \exp\!\big(-\tfrac{1}{\lambda}\big(\tilde S^{(i)} - \tilde S_{min}\big)\big),
   \qquad
   \tilde S^{(i)} = S(U^{(i)}) + \lambda \sum_k (u_k^{nom})^\top \Sigma_u^{-1} \epsilon_k^{(i)},$
   the update $u_k \leftarrow u_k^{nom} + \sum_i w^{(i)} \epsilon_k^{(i)}$ is the self-normalized
   importance-sampling estimate of $E_{q^*}[U]$. *Sketch (5 steps):* target $q^*$, proposal $p$
   centered on the nominal; likelihood ratio $q^*/p$ yields the exponentiated cost *plus the
   cross-term* $\lambda\, u^\top \Sigma_u^{-1}\epsilon$ (the term everyone forgets — derived, not
   asserted); subtract $\tilde S_{min}$ for numerical stability (invariance of self-normalized
   weights); form the weighted mean; shift the horizon (receding step = the predict step of the
   twin). *Collapsible:* the full likelihood-ratio algebra, plus **the twin table** set in the book
   color code — nominal plan (prior, blue) → perturbed rollouts (prediction, orange) → cost
   evaluation (likelihood, green) → weighted update (posterior, purple) — one row per particle-filter
   correspondence (sample/weight/estimate/resample-analogue), the book's color semantics carrying
   the argument.
4. **Constraint handling.** *Statement + recipes:* (a) input constraints by clamping samples to
   $[v_{min}, v_{max}] \times [-\omega_{max}, \omega_{max}]$ (clamp-then-rollout keeps dynamics
   honest); (b) obstacles as costs — indicator-with-huge-constant vs. smooth ESDF barrier
   $c_{obs}(x) = w_o \exp(-d_{ESDF}(x)/\sigma_o)$, and why smooth costs reduce weight degeneracy
   (ESS argument, imported intact from Ch. 8); (c) chance-constraint flavor: inflate by belief —
   the MCL covariance widens the footprint (one boxed equation, pointer to belief-space MPC).
   *Collapsible:* colored-noise / smoothed perturbations (sampling $\epsilon$ with temporal
   correlation) and Savitzky–Golay smoothing of the updated plan.
5. **MPPI vs. gradient MPC — the honest scorecard.** *Statement (comparison, not theorem):* MPPI
   needs only a *simulator* of $f$ and pointwise costs — no differentiability, so raw occupancy
   costs, hard contact, and multimodal cost landscapes are fine; it parallelizes embarrassingly;
   it degrades with dimension (sample complexity) and jitters near constraint boundaries. Gradient
   MPC (iLQR/DDP, SQP) exploits smoothness for fast local convergence and tight constraint
   handling, but needs $\nabla f$, $\nabla c$ and a smooth world (ESDF, not occupancy), and commits
   to one homotopy class. Rule of thumb boxed: *sampling for contact-rich/discontinuous/multimodal,
   gradients for smooth/high-dimensional/certified.* iLQR gets a half-page sketch (backward
   Riccati-like pass), not an implementation; factor-graph MPC gets a pointer (Ch. 15 lineage).

**Named algorithms:**

| Algorithm | Signature | Complexity |
|---|---|---|
| `dwa_plan` | `(x, x_goal, v_prev, esdf) -> (v, ω)` | $O(n_v n_\omega H)$ |
| `mppi_step` | `(x0, U_nom, K, λ, Σ_u) -> U_new` | $O(K H)$ dynamics+cost evals; embarrassingly parallel over $K$ |
| `rollout` | `(x0, U) -> (states, S)` | $O(H)$ |
| `smooth_plan` (Savitzky–Golay) | `(U) -> U` | $O(H)$ |

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w23.1: Rollout Storm** *(flagship, interactive sim)* — type: wasm-sim, full-width. The
  Apartment with draggable clutter and one moving obstacle. Every frame: $K$ candidate trajectories
  fan out from Rusty — perturbed rollouts drawn in prediction-orange with opacity $\propto$ weight
  $w^{(i)}$, the executed plan in posterior-purple, the previous nominal in prior-blue, cost field
  (ESDF barrier) as a faint green underlay — the book's palette *is* the legend, matching the twin
  table in F.3 term for term. Autoplays at K=256. The *one meaningful parameter*: the temperature
  slider $\lambda$. At $\lambda \to 0$: winner-takes-all — the purple plan snaps to the single best
  orange rollout and jitters. At large $\lambda$: the plan averages over everything and cuts
  corners toward obstacles. Secondary (disclosure): $K$, noise $\sigma$, horizon $H$, pause+scrub
  one control cycle (the "collapse" animated in slow motion: orange storm → green costing →
  purple collapse). ESS meter always visible — the Ch. 8 gauge, back for controls. *Misconception
  killed:* "MPC requires convexity/gradients," and "the chosen trajectory is one of the samples"
  (it is a weighted *blend* — visibly smoother than any single rollout).
- **Widget w23.2: Dynamic Window Bench** *(interactive sim, supporting)* — split view. Left: the
  $(v, \omega)$ plane; the dynamic window rectangle, inadmissible region shaded, objective heatmap,
  chosen velocity starred. Right: the world with the corresponding arc. Drag obstacles and watch
  the window get carved. Button: teleport the scene into the dead-end arena — DWA freezes at the
  wall (its best arc is "stop"), then a toggle swaps in MPPI, which discovers the reverse-and-turn
  escape because its horizon is $H$ steps, not one arc. *Misconception killed:* "reactive obstacle
  avoidance is enough" — myopia is structural, not a tuning problem.
- **Widget w23.3: Twin Vision** *(animation, supporting)* — split screen, synchronized stepping.
  Left: the Ch. 8 particle filter on the Hallway (states, weights, resample). Right: one MPPI cycle
  (rollouts, weights, update). The same four sub-steps light up in lockstep with the same colors;
  captions name the correspondence (sample ↔ rollout, likelihood ↔ exp(−cost/λ), posterior mean ↔
  updated plan, resampling ↔ horizon shift/re-centering). No parameters — a pure watching widget,
  scrub-only. *Misconception killed:* "MPPI is an ad-hoc heuristic" — it is the same importance-
  sampling mathematics the reader already proved correct in Ch. 8.

Dashboard: w23.1 is the chapter dashboard (storm + ESS meter + cost readout + λ slider in one
frame); w23.2/w23.3 are section-inline half-width. Static fallbacks: three-phase filmstrip of one
MPPI cycle (storm/costed/collapsed) for w23.1; final-frame SVGs for the others.

## 5. Practical (P) — Rust Implementation

Crates:
- `nalgebra` 0.35 — `SVector<f64, 3>` states, `SVector<f64, 2>` controls, `SMatrix` $\Sigma_u$;
  const generics make horizon and dimensions compile-time.
- `rand` 0.9 + `rand_distr` 0.6 — seeded Gaussian perturbations (`Pcg64`; every storm reproducible).
- `rayon` — parallel rollouts natively; the same code compiles single-threaded to WASM behind
  `#[cfg(target_arch = "wasm32")]` (the book shows the two-line cfg — a Rust-pedagogy moment).
- `eframe`/`egui` 0.35 + `egui_plot` 0.34 — widgets.
- Depends on `motion` (the Ch. 9 velocity model as `Dynamics`), `ch19_maps` (ESDF), `localize`
  (Ch. 12 MCL pose belief input in the lab), `ch20_planning` (reference path).

Module plan: `crates/ch23_mppi/` with `src/dynamics.rs`, `cost.rs`, `mppi.rs`, `dwa.rs`,
`smooth.rs`, `examples/clutter_run.rs`, `examples/deadend_duel.rs`.

```rust
use nalgebra::{SMatrix, SVector};

pub trait Dynamics<const NX: usize, const NU: usize> {
    fn step(&self, x: &SVector<f64, NX>, u: &SVector<f64, NU>, dt: f64) -> SVector<f64, NX>;
    fn clamp(&self, u: SVector<f64, NU>) -> SVector<f64, NU>;   // input constraints, F.4(a)
}

/// Rusty: unicycle with accel limits — the Ch. 9 velocity model, noise-free.
pub struct DiffDrive { pub v_lim: (f64, f64), pub w_max: f64, pub a_max: f64 }
impl Dynamics<3, 2> for DiffDrive { /* ... */ }

pub trait CostFn<const NX: usize, const NU: usize> {
    fn stage(&self, x: &SVector<f64, NX>, u: &SVector<f64, NU>, k: usize) -> f64;
    fn terminal(&self, x: &SVector<f64, NX>) -> f64;
}
/// path-tracking + ESDF barrier + control effort; weights are public and slider-bound in the widget
pub struct TrackAndClear<'a> { pub path: &'a [SVector<f64, 3>], pub esdf: &'a ch19_maps::Esdf2,
                               pub w_track: f64, pub w_obs: f64, pub sigma_o: f64 }

pub struct Mppi<D, C, const NX: usize, const NU: usize, const H: usize>
where D: Dynamics<NX, NU> + Sync, C: CostFn<NX, NU> + Sync {
    pub lambda: f64,
    pub sigma_u: SMatrix<f64, NU, NU>,
    pub k_samples: usize,
    nominal: [SVector<f64, NU>; H],
    dynamics: D, cost: C, rng: rand::rngs::SmallRng,
}

impl<D, C, const NX: usize, const NU: usize, const H: usize> Mppi<D, C, NX, NU, H> {
    /// One control cycle: sample K rollouts (rayon on native), weight, update, shift.
    /// Returns the first control and a diagnostics struct (ESS, S_min, weights) the widget renders.
    pub fn plan(&mut self, x0: &SVector<f64, NX>) -> (SVector<f64, NU>, MppiDiag);
}

pub struct Dwa { pub n_v: usize, pub n_w: usize, pub weights: DwaWeights }
impl Dwa { pub fn plan(&self, x: &SVector<f64,3>, goal: &SVector<f64,3>,
                       v_prev: (f64, f64), esdf: &ch19_maps::Esdf2) -> (f64, f64); }
```

Worked end-to-end example (`examples/clutter_run.rs`, seed 23): Rusty tracks the Ch. 20 RRT* path
through the cluttered Apartment with one moving obstacle; MCL (Ch. 12) supplies the pose;
$K{=}512$, $H{=}30$ ($1.5$ s at 20 Hz), $\lambda{=}0.1$, $\Sigma_u = \mathrm{diag}(0.15^2,
0.4^2)$. Printed table (unit-tested on the fixed seed): mean tracking error 6 cm, min clearance
0.19 m, 0 collisions in 50 runs; native throughput ~2.4 M rollout-steps/s with rayon; WASM at
$K{=}256$ holds 20 Hz. `deadend_duel` prints the scorecard: DWA 3/50 escapes, MPPI 50/50 —
the F.1 myopia claim, measured.

Runnable artifact: `cargo run --release --example clutter_run` (+ `deadend_duel`); the WASM demo is
w23.1, running this exact `Mppi` with `MppiDiag` driving the storm rendering.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w23.1 | Rollout Storm | wasm-sim | ch23_mppi + ch19_maps + localize (Ch. 12 MCL) + eframe 0.35 | λ slider; disclosure: K, σ, H; drag clutter; pause+scrub one cycle; ESS meter | MPPI as importance sampling over controls; temperature trade-off |
| w23.2 | Dynamic Window Bench | wasm-sim | ch23_mppi + ch19_maps + eframe | drag obstacles, dead-end teleport, DWA↔MPPI toggle | velocity-space search; structural myopia of one-arc reactive control |
| w23.3 | Twin Vision | animation | ch23_mppi + ch08_particles + eframe | step/scrub synchronized PF and MPPI cycles | the particle-filter ↔ MPPI correspondence, sub-step by sub-step |
| f23.4 | MPPI cycle filmstrip (storm/cost/collapse) | static-svg | plotters (build-time) | — | anatomy of one control cycle (w23.1 fallback) |

## 7. Exercises & Extensions

1. **(F)** Derive the likelihood-ratio cross-term $\lambda \sum_k (u_k^{nom})^\top \Sigma_u^{-1}
   \epsilon_k^{(i)}$ from $q^*/p$ (F.3), and show that dropping it biases the update toward large
   perturbations. Verify the bias experimentally by toggling the crate's `debug_drop_cross_term`
   feature on the clutter run.
2. **(F)** Show that as $\lambda \to 0$ the MPPI update converges to best-of-$K$ selection, and as
   $\lambda \to \infty$ to the unweighted mean of perturbations (i.e., the nominal plan plus zero
   in expectation). Relate both limits to the ESS meter's readings in w23.1.
3. **(C, predict-then-verify)** In w23.1, set $\sigma$ to one quarter of the default. Predict:
   will Rusty still discover the gap between the two chairs, and how will the ESS change? Verify,
   then explain using the proposal-coverage language of Ch. 8's importance sampling.
4. **(C)** Use w23.2 to construct the *smallest* obstacle arrangement that defeats DWA but not
   MPPI. What horizon $H$ does MPPI need for your arrangement? (Find out by shrinking $H$.)
5. **(P)** Implement temporally correlated (colored) noise sampling in `mppi.rs` and measure: plan
   smoothness (mean $|\Delta u|$) and success rate vs. the white-noise default on the clutter run.
6. **(P, stretch)** Implement one-step iLQR on the same `Dynamics`/`CostFn` traits (smooth costs
   only) and race it against MPPI on (a) the smooth-ESDF world and (b) a raw-occupancy cost world.
   Reproduce the F.5 scorecard's two headline rows.

## 8. Modernization Notes

- This chapter has no 2005/1999 baseline to modernize — the Thrun draft contains no control
  chapter, and the classical texts (Spong Chs. 8–9, Craig's control chapters, Lynch & Park Ch. 11)
  cover trajectory-tracking control of *known* trajectories in *obstacle-free* task space. They are
  cited as the classical context (and their PID/computed-torque worldview frames the hook), but the
  content — receding-horizon control, DWA, and MPPI — is built from the modernization set: DWA (Fox,
  Burgard & Thrun 1997, the probabilistic-robotics lineage the draft never absorbed) and MPPI
  (Williams et al. 2016–2018), now shipping as a stock Nav2 controller — evidence the chapter cites
  when claiming this is the field's working default for diff-drive local control.
- Deliberate omissions, each with a pointer box: continuous-time path-integral control (Girsanov
  machinery — replaced by the discrete-time information-theoretic derivation, which is exact and
  honest at our level); iLQR/DDP as implemented methods (sketched only; the trait design leaves the
  socket); tube/robust MPC and formal chance constraints (one boxed equation on belief-inflated
  footprints, pointer to belief-space MPC literature); learned dynamics/terminal costs (Ch. 25's
  territory — its differentiable-filter machinery pairs naturally with gradient MPC).
- A scoping decision made explicit: MPPI here controls a *kinematic* diff-drive at 2D-book scale.
  The text notes (one paragraph, with citations) that the same algorithm drives aggressive
  off-road racing at 40+ dimensions of state on GPUs — the reader's `rayon` loop is the small end
  of a real production continuum, not a toy.
