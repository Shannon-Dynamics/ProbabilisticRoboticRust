# Chapter 21 — Decision Making I: MDPs and Value Iteration

> Part VI — Planning and Acting under Uncertainty · Estimated length: 7 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Chapter 20 ended on a cliffhanger: a beautiful RRT* path, and Rusty sliding off it, because plans
assume execution is perfect and Ch. 9 spent thirty pages proving it is not. This chapter makes the
conceptual jump the whole Part pivots on: from a *plan* (a curve through space) to a *policy* (an
answer for every state). The vehicle is the Markov Decision Process — the fully-observable half of
the decision-theoretic story — and its workhorse, value iteration. The "aha" the reader leaves with:
a plan is a line, a policy is a vector field; and the value function is the principled version of
Ch. 20's wave-front — a potential field that *cannot* have local minima because it is defined by
expected cost-to-go rather than geometry. The closing bridge asks the destabilizing question Ch. 22
answers: what if the state itself is uncertain?

Story line:
1. **Hook:** the gridworld that refuses to walk straight — Rusty on slippery wheels executes the
   Ch. 20 path, drifts into the cliff cell three runs out of ten (autoplay).
2. **Play (C):** Policy Painter — paint rewards, crank the noise, watch arrows re-converge.
3. **Formalize (F):** MDP tuple, return, discounting, policies, value functions.
4. **The Bellman equation** derived, then value iteration with its contraction-mapping convergence
   proof; policy extraction; policy iteration and asynchronous variants.
5. **Stochastic shortest paths:** the undiscounted, goal-absorbing formulation that navigation
   actually is; the wave-front planner unmasked as its deterministic special case.
6. **Practical (P):** `mdp` module; a gridworld compiled from the Apartment occupancy grid, with a
   slip model discretized honestly from the Ch. 9 velocity motion model.
7. **Integration lab:** Rusty with slippery wheels navigating by policy; compare policy vs. replanned
   RRT* under increasing noise.
8. **Bridge:** the kidnapped-Rusty run where the policy is perfect and useless — state uncertainty
   as the next enemy (feeds Ch. 22).

## 2. Prerequisites & Position

- **Builds on:** Ch. 2 (expectation), Ch. 5 (Markov assumption; $p(x_t \mid x_{t-1}, u_t)$ is the
  same object the Bayes filter predicts with), Ch. 9 (velocity motion model → discrete transition
  probabilities), Ch. 13 (occupancy grid → state space), Ch. 20 (the plan-vs-policy contrast; the
  wave-front planner as the deterministic seed).
- **Feeds into:** Ch. 22 (belief-MDP: the same machinery over beliefs; QMDP reuses this chapter's
  $Q$-values), Ch. 23 (MPPI's terminal cost is a value-function stand-in), Ch. 24 (exploration as
  decision making), Ch. 26.
- **Baseline sources:** Thrun et al. (draft) Ch. 15 §15.1–15.3 (motivation, uncertainty in action
  selection, goals and payoff, value iteration, illustration; algorithm `MDP_value_iteration()`,
  Table 15.1). Supplementary rigor (contraction proof, policy iteration, SSP) is standard dynamic-
  programming material absent from the draft, following Bertsekas' treatment; Choset App. H.4
  ("Optimal Plans") as the graph-search cross-reference.

## 3. Foundation (F) — Mathematical Core

**Notation introduced** (per the TOC table: $\pi$, $r(x,u)$, $\gamma$, $V$ are book-reserved for
Part VI):

| Symbol | Meaning |
|---|---|
| $(\mathcal{X}, \mathcal{U}, p(x' \mid x, u), r(x, u), \gamma)$ | MDP: states, controls, transition model, reward, discount $\gamma \in [0,1)$ |
| $\pi : \mathcal{X} \to \mathcal{U}$ | (deterministic, stationary) control policy |
| $R_T = E\!\left[\sum_{\tau=0}^{T} \gamma^{\tau} r_{\tau}\right]$ | expected cumulative discounted payoff |
| $V^{\pi}(x)$, $V^{*}(x)$ | value of $\pi$; optimal value function |
| $Q^{*}(x, u)$ | optimal state–action value (introduced for Ch. 22's QMDP) |
| $\mathcal{T}$ | Bellman backup operator |

States are grid cells of the (inflated) Apartment map; controls are the eight compass moves plus
*stay*; the transition model is *derived, not decreed*: integrate the Ch. 9 velocity model over one
cell-crossing and bin the outcome — a slip parameter $s$ emerges (intended cell w.p. $1 - 2s$, each
lateral neighbor w.p. $s$), and the text shows the integral that produced it.

**Definitions:** MDP; policy; return; discounting (and why $\gamma < 1$ buys convergence);
value function; optimal policy; greedy policy; proper policy (for SSP); asynchronous backup.

**Key derivations:**

1. **Bellman optimality equation.** *Statement:* $V^{*}(x) = \max_{u} \big[ r(x, u) + \gamma
   \sum_{x'} p(x' \mid x, u)\, V^{*}(x') \big]$. *Sketch (4 steps):* write $R_T$ recursively as
   first reward + discounted remainder; condition on the first control and transition; optimal
   substructure (the tail must itself be optimal — argued, not hand-waved); take the max over the
   first control. *Collapsible:* full finite-horizon induction $V_T \to V_{T+1}$ and the limit
   argument to the stationary infinite-horizon equation, matching Thrun §15.3's payoff framing.
2. **Value iteration converges (contraction).** *Statement:* the backup operator
   $(\mathcal{T}V)(x) = \max_u [\, r(x,u) + \gamma \sum_{x'} p(x' \mid x,u) V(x')\,]$ is a
   $\gamma$-contraction in $\|\cdot\|_\infty$; hence $V_{k+1} = \mathcal{T}V_k$ converges to the
   unique fixed point $V^*$ geometrically, with the stopping-rule bound $\|V_k - V^*\|_\infty \le
   \frac{\gamma}{1-\gamma}\|V_k - V_{k-1}\|_\infty$. *Sketch (5 steps):* bound
   $|\max_u a_u - \max_u b_u| \le \max_u |a_u - b_u|$; push the expectation through; extract
   $\gamma$; Banach fixed-point theorem; derive the stopping bound. *Collapsible:* the full
   $\epsilon$-argument and the error bound on the *greedy policy's* value,
   $\|V^{\pi_k} - V^*\|_\infty \le \frac{2\gamma}{1-\gamma}\|V_k - V^*\|_\infty$ — the reason a
   loosely-converged $V$ still yields a nearly-optimal policy (the fact Policy Painter exploits to
   animate mid-convergence arrows honestly).
3. **Greedy policy extraction is optimal at the fixed point.** *Statement:* $\pi^*(x) = \arg\max_u
   [\,r(x,u) + \gamma \sum_{x'} p(x' \mid x,u) V^*(x')\,]$ attains $V^{\pi^*} = V^*$. *Sketch
   (3 steps):* greedy w.r.t. $V^*$ makes $V^*$ a fixed point of the *policy's* (linear) backup;
   uniqueness of that fixed point; conclude equality. *Collapsible:* the linear-operator version.
4. **Policy iteration terminates finitely.** *Statement:* alternating exact policy evaluation
   (a linear solve) with greedy improvement reaches $\pi^*$ in at most $|\mathcal{U}|^{|\mathcal{X}|}$
   steps, in practice a handful. *Sketch:* improvement is monotone; finitely many policies; strict
   improvement until fixed point. *Collapsible:* evaluation as the sparse linear system
   $(I - \gamma P_{\pi}) V^{\pi} = r_{\pi}$ and modified/optimistic policy iteration interpolating
   between VI and PI.
5. **Stochastic shortest paths.** *Statement:* with $\gamma = 1$, absorbing zero-reward goal, and
   $r = -1$ per step, value iteration still converges provided a *proper* policy exists (goal
   reached w.p. 1) and improper policies have infinite cost; $-V^*(x)$ is then the minimal expected
   number of steps. *Sketch:* cite-and-sketch (Bertsekas), no full proof. Payoff: the numeric
   micro-example below, and the punchline that **the Ch. 20 wave-front planner is exactly SSP value
   iteration with deterministic transitions** — one boxed equation showing the max collapse.

**Named algorithms:**

| Algorithm | Signature | Complexity |
|---|---|---|
| `MDP_value_iteration` (Thrun Table 15.1) | `(p, r, γ, ε) -> (V, π)` | $O(|\mathcal{X}|^2 |\mathcal{U}|)$ per sweep dense; $O(|E||\mathcal{U}|)$ sparse; $O(\log_\gamma \epsilon)$ sweeps |
| `policy_evaluation` | `(π, p, r, γ) -> V^π` | one sparse linear solve or iterated linear backups |
| `policy_iteration` | `(p, r, γ) -> (V*, π*)` | few outer iterations × evaluation cost |
| `gauss_seidel_vi` / `prioritized_sweeping` | in-place backups, priority by Bellman residual | same worst case, large constant-factor wins (measured in P) |

**Numeric micro-example** (kalmanfilter.net discipline; unit-tested): the 4-cell hallway SSP —
states $\{A, B, C, G\}$, action *right* succeeds w.p. $0.8$, stays w.p. $0.2$, $r = -1$ per step,
$G$ absorbing. Closed-form fixed point: $V(C) = -1.25$, $V(B) = -2.5$, $V(A) = -3.75$ (each cell
costs $1/0.8$ expected steps). The chapter shows the first three sweeps converging toward these
numbers; the reader can verify every line by hand.

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w21.1: Policy Painter** *(flagship, interactive sim)* — type: wasm-sim. A coarse
  gridworld over the Apartment floorplan. The reader *paints*: left-drag deposits reward (goal
  cells, purple-positive), right-drag deposits hazard (cliff cells, negative); the value function
  renders as a heatmap and the policy as an arrow field, both re-converging live (VI sweeps run
  ~60/frame in WASM — convergence is visible as a wave, not instant). The *one meaningful
  parameter*: the slip slider $s \in [0, 0.4]$. Secondary (behind disclosure): $\gamma$ slider,
  sweep-speed, sweep-order toggle (sweep vs. prioritized — watch the wave change shape). Autoplay
  default: goal at the charging dock, hazard at the stairwell, $s = 0.2$. Observes: raising $s$
  bends the arrow field away from the hazard corridor — the *detour margin grows with noise*;
  lowering $\gamma$ makes the far goal invisible (myopia). *Misconception killed:* "the optimal
  route is the shortest route" — and, deeper, "a policy is just a stored path."
- **Widget w21.2: Bellman Stepper** *(animation, supporting)* — one cell magnified. A single backup
  is drawn as machinery: the candidate actions' $Q$ bars assemble from neighbor values (each
  neighbor's contribution shown as $\gamma \cdot p \cdot V$ slabs stacking), the max gate lights up,
  the cell's value updates. Step buttons: one backup / one sweep / run. Then zoom out: the value
  wave propagating backward from the goal, visually rhyming with w20.3's wave-front — the chapter's
  central visual echo. *Misconception killed:* "value iteration is opaque linear algebra" — it is
  local, mechanical bookkeeping.
- **Widget w21.3: Cliff Run** *(interactive sim, supporting; predict-then-verify)* — the classic
  two-route world: short path along a cliff, long safe path. One slider: slip $s$. A vertical marker
  shows the *analytically computed* critical $s^*$ where the optimal policy flips routes — but only
  after the reader commits a prediction (the widget asks first, pedagogy-by-wager). Ten Rusty runs
  animate per setting, tallying realized return vs. $V^*$. *Misconception killed:* "optimize the
  best case" — expectation is the only honest objective under noise, and variance is visible.

Color code: value heatmap uses the book's posterior-purple ramp (value *is* the distilled posterior
of future reward); hazard cells in measurement-green would clash, so hazards use the standard
alert treatment outside the belief palette; ground-truth trajectories in gray dashed. Equations in
F color $r(x,u)$, $\gamma\sum p V$, and $V^*$ to match w21.2's slabs, bzarg-style.

## 5. Practical (P) — Rust Implementation

Crates:
- `nalgebra` 0.35 — dense value vectors; `faer` 0.24 optional feature for the policy-evaluation
  sparse solve (reusing the Ch. 15 sparse Cholesky path).
- `rand` 0.9 (`SmallRng`, seeded) — Monte Carlo rollouts validating $V^*$ empirically.
- `eframe`/`egui` 0.35 + `egui_plot` 0.34 — Policy Painter and friends.
- Depends on workspace crates `sim` (worlds), `motion` (the Ch. 9 velocity model for the slip
  derivation), `ch13_occgrid` (OccGrid → state space).

Module plan: `crates/ch21_mdp/` with `src/mdp.rs` (model), `vi.rs`, `pi.rs`, `gridworld.rs`
(OccGrid → MDP compiler, incl. the motion-model discretizer), `examples/hallway_ssp.rs`,
`examples/slippery_gridworld.rs`.

```rust
/// Sparse transition row: p(x' | x, u) as (state, prob) pairs. Probabilities sum to 1.
pub struct SparseDist(pub Vec<(u32, f32)>);

/// A finite MDP with A actions. States are dense indices (grid cells).
pub struct Mdp<const A: usize> {
    pub n_states: usize,
    pub trans: Vec<[SparseDist; A]>,   // indexed [state][action]
    pub reward: Vec<[f64; A]>,          // r(x, u)
    pub gamma: f64,                     // γ ∈ [0, 1); γ = 1 allowed iff `absorbing` non-empty (SSP)
    pub absorbing: Vec<u32>,
}

pub struct ViResult { pub v: Vec<f64>, pub policy: Vec<u8>, pub sweeps: usize }

/// Thrun Table 15.1, with the §3.2 stopping rule ‖V_k − V_{k−1}‖∞ < ε(1−γ)/γ.
pub fn value_iteration<const A: usize>(mdp: &Mdp<A>, eps: f64) -> ViResult;

/// One in-place Gauss–Seidel sweep; returns the max Bellman residual (drives the widget's wave).
pub fn sweep_in_place<const A: usize>(mdp: &Mdp<A>, v: &mut [f64]) -> f64;

pub fn policy_iteration<const A: usize>(mdp: &Mdp<A>) -> ViResult;

/// Compile an MDP from an occupancy grid + the Ch. 9 velocity model (slip s estimated by
/// integrating sample_motion_model_velocity over one cell transit; see §F notes).
pub fn gridworld_from_occgrid(
    grid: &ch13_occgrid::OccGrid, motion: &motion::VelocityModel, cell_m: f64,
) -> Mdp<9>;
```

Worked end-to-end example (`examples/slippery_gridworld.rs`, seed 21): (1) `hallway_ssp` reproduces
the F micro-example exactly — the unit test asserts $V = [-3.75, -2.5, -1.25, 0]$ to $10^{-9}$;
(2) the Apartment at 0.25 m cells (~3400 free states), $s = 0.2$, $\gamma = 0.98$: VI converges in
217 sweeps (41 with prioritized sweeping — the benchmark table is printed), the policy routes Rusty
through the wide corridor, and 1000 seeded rollouts land within 1% of $V^*(x_0)$ — Monte Carlo
meeting dynamic programming, on the page.

Runnable artifact: `cargo run --example slippery_gridworld` prints the sweep/benchmark table and
writes the value-heatmap + arrow-field figure via `plotters` (w21.1's static fallback). The WASM
demo is Policy Painter running these exact functions; `sweep_in_place`'s residual drives its
convergence-wave animation.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w21.1 | Policy Painter | wasm-sim | ch21_mdp + sim + eframe 0.35 + egui_plot 0.34 | paint rewards/hazards, slip slider, γ slider, sweep-order toggle | value functions & policies re-converging; noise bends optimal behavior |
| w21.2 | Bellman Stepper | animation | ch21_mdp + eframe | step one backup / one sweep / run | the backup as local machinery; the value wave |
| w21.3 | Cliff Run | wasm-sim | ch21_mdp + sim + eframe | slip slider, predict-then-verify prompt, run tally | expectation vs. best case; policy flips at a critical noise |
| f21.4 | Plan vs. policy under slip (3-panel) | static-svg | plotters (build-time) | — | why the Ch. 20 path fails and the vector field doesn't |

## 7. Exercises & Extensions

1. **(F)** Prove the max-inequality $|\max_u a_u - \max_u b_u| \le \max_u |a_u - b_u|$ used in the
   contraction proof, and exhibit vectors where it is tight.
2. **(F)** For the 4-cell hallway with success probability $p$, derive $V(x)$ in closed form and
   verify the $s = 0.2$ numbers; then compute the expected number of *sweeps* VI needs to reach
   $\|V_k - V^*\| < 0.01$ using the geometric bound, and compare with the crate's actual count.
3. **(C, predict-then-verify)** In w21.3, predict the critical slip $s^*$ for the default cliff
   geometry before touching the slider; then find it by bisection with the widget. Explain the
   mismatch between $s^*$ and where *realized* runs start preferring the safe route.
4. **(C)** Use w21.1 to construct a reward painting where lowering $\gamma$ *changes the policy's
   topology* (routes through a different doorway). Screenshot both arrow fields and explain.
5. **(P)** Implement `prioritized_sweeping` with a binary heap keyed on Bellman residual; reproduce
   the chapter's 217→41 sweep reduction and plot residual vs. wall-clock against plain VI.
6. **(P, stretch)** Add an SSP mode ($\gamma = 1$) with proper-policy detection (flag states from
   which the goal is unreachable) and use it to auto-generate the Ch. 20 wave-front from the MDP
   compiler — one assert: identical distance fields on a deterministic MDP.

## 8. Modernization Notes

- The draft's Ch. 15 is the thinnest chapter in the baseline: motivation, payoff, value iteration,
  one illustration — no convergence proof, no policy iteration, no complexity discussion. This
  chapter keeps its narrative (and the `MDP_value_iteration` algorithm name) but adds the modern
  standard kit: contraction-mapping convergence with stopping bounds, greedy-policy loss bound,
  policy iteration and evaluation-as-linear-solve, asynchronous/prioritized backups, and the
  stochastic-shortest-path formulation that grid navigation actually is.
- Deliberately *not* added: reinforcement learning. Q-learning/actor-critic solve the same Bellman
  equations with unknown models; that thread belongs to Ch. 25's "learning in the loop" framing and
  gets a two-sentence pointer here. Similarly, continuous-state MDPs (LQR as the Gaussian-quadratic
  special case) get a boxed aside pointing to Ch. 23, where receding-horizon control takes over the
  continuous world in practice.
- Dropped from the 2005-era framing: nothing substantive — instead the chapter *re-grounds* the
  gridworld: rather than an abstract slip constant, the transition model is compiled from the Ch. 9
  velocity model, keeping the book's promise that every probability has a pedigree.
