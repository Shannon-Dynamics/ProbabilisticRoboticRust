# Chapter 22 — Decision Making II: POMDPs and Belief-Space Planning

> Part VI — Planning and Acting under Uncertainty · Estimated length: 10 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Chapter 21 ended with a perfect policy rendered useless by a kidnapped robot: the policy answers
"what to do in state $x$," but Rusty only ever has $bel(x)$. This chapter completes the book's
central circle: the belief that Parts II–IV taught the reader to *maintain* becomes the thing to
*plan over*. The POMDP is where estimation and decision-making fuse — and where the book's most
delightful emergent behavior appears: agents that pay to sense, hug walls to stay localized, and
take detours to disambiguate, none of it hand-coded. The reader plays the tiger problem honestly
(losing money to the tiger before seeing any math), then climbs the ladder: belief MDP →
$\alpha$-vectors and piecewise-linear-convex value functions → exact value iteration and its
combinatorial explosion → point-based methods (PBVI) → the modern practical path, online sparse-tree
solvers (POMCP, DESPOT) running on the book's own particle machinery → AMDP/coastal navigation. The
"aha": *the value of information is not a bolted-on heuristic — it falls out of the Bellman equation
the moment the state is a belief.*

Story line:
1. **Hook:** two identical doors in the Hallway; behind one, Rusty's charging dock; behind the
   other, a stairwell drop. A noisy sensor may be consulted for a price. The reader plays (w22.1
   autoplays a taunting demo first).
2. **Formalize (F):** POMDP tuple; the belief update *is* the Ch. 5 Bayes filter; the belief MDP.
3. **Structure (F):** finite-horizon value functions are piecewise-linear-convex; $\alpha$-vectors;
   the exact backup and why it explodes; pruning.
4. **Tame it (F):** QMDP (the cheap baseline that never buys information), PBVI, and online search —
   POMCP/DESPOT as the field's working answer.
5. **Behavior (C):** Coastal Navigator — uncertainty-aware paths emerge from AMDP planning.
6. **Practical (P):** exact finite-world solver; POMCP over Ch. 8 particles; Rusty relocalizes
   before committing to a corridor.
7. **Bridge out:** POMDPs are the *conceptual* frame for Chs. 23–24 — MPPI and active SLAM are both
   tractable specializations of "act on the belief."

## 2. Prerequisites & Position

- **Builds on:** Ch. 2 (entropy), Ch. 5 (Bayes filter = belief transition function), Ch. 8
  (particle representation, importance sampling — POMCP's substrate), Ch. 12 (MCL beliefs that the
  lab plans over), Ch. 21 (MDP machinery; $Q^*$ feeds QMDP; value iteration reused twice).
- **Feeds into:** Ch. 23 (receding-horizon control as the tractable belief-space workhorse), Ch. 24
  (active SLAM formulated as a POMDP, then approximated), Ch. 26 (capstone decision layer).
- **Baseline sources:** Thrun et al. (draft) Ch. 16 — §16.1–16.2 (motivation, finite environments,
  the two-state illustrative example, value iteration in belief space, linear-programming pruning;
  algorithm `finite_world_POMDP(T)`, Table 16.1), §16.3 (`POMDP(T)`, Table 16.2), §16.5 (Augmented
  MDPs, `Augmented_MDP_value_iteration()`, coastal navigation). §16.4 (`MCPOMDP_*`, Tables
  16.3–16.4) is treated as history (see §8). Modernization set: Kaelbling/Littman/Cassandra 1998
  (tiger, $\alpha$-vector formalism), Pineau et al. 2003 (PBVI), Kurniawati et al. (SARSOP,
  mention), Silver & Veness 2010 (POMCP), Somani et al. 2013 / Ye et al. 2017 (DESPOT).

## 3. Foundation (F) — Mathematical Core

**Notation introduced:**

| Symbol | Meaning |
|---|---|
| $(\mathcal{X}, \mathcal{U}, \mathcal{Z}, p(x' \mid x, u), p(z \mid x), r(x, u), \gamma, b_0)$ | POMDP tuple |
| $b$, $b(x)$ | belief state — exactly $bel(x)$ of Ch. 5, now a *planning* state |
| $b' = \tau(b, u, z)$ | belief update (Bayes filter step), $\tau$ the belief transition |
| $p(z \mid b, u) = \sum_{x'} p(z \mid x') \sum_x p(x' \mid x,u) b(x)$ | observation likelihood under a belief |
| $\rho(b, u) = \sum_x b(x)\, r(x, u)$ | belief reward |
| $\Gamma_T = \{\alpha^{(k)}\}$, $V_T(b) = \max_k \langle \alpha^{(k)}, b \rangle$ | $\alpha$-vector set; PWLC value function |
| $\Delta^{|\mathcal{X}|-1}$ | the belief simplex (a segment for two states) |

**Notation collision, flagged in a warning box:** $\alpha$-vectors are standard POMDP vocabulary
and unrelated to the motion-noise parameters $\alpha_1..\alpha_6$ of Ch. 9. We write vectors
$\alpha^{(k)}$ with superscripts, noise with subscripts, and never use both in one equation.

**The tiger problem, fully specified** (the chapter's numeric micro-example): $\mathcal{X} =
\{x_L, x_R\}$ (tiger left/right), $\mathcal{U} = \{\text{listen}, \text{open}_L, \text{open}_R\}$,
$\mathcal{Z} = \{z_L, z_R\}$ with hearing accuracy $0.85$; $r(\cdot, \text{listen}) = -1$, correct
door $+10$, tiger door $-100$; $\gamma = 0.95$. Worked numbers: from $b(x_L) = 0.5$, hearing $z_L$
gives $b'(x_L) = \frac{0.85 \cdot 0.5}{0.85 \cdot 0.5 + 0.15 \cdot 0.5} = 0.85$; a second
consistent $z_L$ gives $0.9698$; one contradicting $z_R$ drops it back to $0.5$. Every number is
hand-checkable and unit-tested.

**Key derivations:**

1. **The belief MDP.** *Statement:* a POMDP is an MDP over $\Delta^{|\mathcal{X}|-1}$ with
   transition $p(b' \mid b, u) = \sum_z p(z \mid b, u)\, \mathbf{1}[b' = \tau(b,u,z)]$ and reward
   $\rho(b,u)$; its optimal policy is the POMDP's optimal policy. *Sketch (4 steps):* belief is a
   sufficient statistic of the history (this is Ch. 5's completeness argument, re-cited not
   re-proved); rewrite the expected return conditioning on histories; collapse histories to beliefs;
   read off the induced MDP. *Collapsible:* the sufficient-statistic proof and why *deterministic*
   belief dynamics + stochastic observations give the branching structure.
2. **PWLC by induction (the chapter's centerpiece).** *Statement:* for every finite horizon $T$,
   $V_T(b) = \max_{\alpha \in \Gamma_T} \langle \alpha, b\rangle$ for a finite set $\Gamma_T$.
   *Sketch (6 steps):* base case $\Gamma_1 = \{r(\cdot, u)\}_{u}$ — for tiger, the three vectors
   $(-1,-1)$, $(-100, 10)$, $(10, -100)$; inductively substitute the PWLC form into the Bellman
   backup; the $z$-expectation of a max of linear functions is a max (over *choices of one $\alpha$
   per observation*) of linear functions — the normalizer $\eta$ in $\tau$ cancels against
   $p(z \mid b,u)$, the derivation's one beautiful trick; enumerate: $|\Gamma_{T+1}| \le
   |\mathcal{U}| \cdot |\Gamma_T|^{|\mathcal{Z}|}$; convexity as max of linear maps. *Collapsible:*
   the full cross-sum algebra (Thrun §16.2.3), and the cancellation displayed term-by-term in the
   book color code (prior-blue $b$, measurement-green $p(z\mid x)$, posterior-purple $b'$).
3. **Why exact VI explodes.** *Statement:* the backup recursion gives doubly-exponential growth in
   the horizon before pruning; pruning to the minimal set requires solving one LP per candidate
   vector; finite-horizon POMDPs are PSPACE-hard, infinite-horizon undecidable (statements with
   citations, no proofs). *Sketch:* count tiger: $|\Gamma_1| = 3$, then $|\mathcal{U}| \cdot
   |\Gamma_1|^{|\mathcal{Z}|} = 27$ candidates at $T{=}2$, ~$2000$ at $T{=}3$ un-pruned — the
   widget w22.3 displays the live counts our solver actually produces. *Collapsible:* the LP
   formulation of dominance testing (Thrun §16.2.4) and why we implement pairwise + witness-point
   pruning instead (see P).
4. **QMDP — the revealing baseline.** *Statement:* $Q_{MDP}(b, u) = \sum_x b(x) Q^*(x, u)$ (with
   $Q^*$ from Ch. 21) is exact *if all uncertainty vanished after one step*; consequently a QMDP
   agent never takes actions whose only value is information. *Sketch (3 steps):* substitute the
   full-observability assumption into the belief Bellman equation; note the expectation moves inside
   the max; conclude information-gathering actions are never selected — in tiger, QMDP never
   listens. *Collapsible:* QMDP as the $\Gamma = \{Q^*(\cdot,u)\}$ one-backup approximation and its
   upper-bound property (used by DESPOT's bounds).
5. **PBVI.** *Statement:* maintain $\Gamma$ only at a finite belief set $B$; the point-based backup
   keeps $\le |B|$ vectors per horizon; error bounded by
   $\frac{(r_{max}-r_{min})\, \epsilon_B}{(1-\gamma)^2}$ where $\epsilon_B$ is the density of $B$ in
   the reachable simplex. *Sketch (4 steps):* per-point backup picks the best cross-sum for that
   $b$; polynomial per-iteration cost; reachable-belief expansion of $B$; error statement
   (proof pointer, not reproduced).
6. **POMCP — planning with the particle machinery.** *Statement:* Monte-Carlo tree search over
   histories with UCB1 action selection, beliefs represented by the particles that reach each node,
   converges to the optimal value in the limit of simulations. *Sketch (5 steps):* simulate from a
   state sampled from $b$; the generative model $(x', z, r) \sim G(x,u)$ is the *simulator the
   reader built in Ch. 4*; bandit view of action selection, UCB1 exploration bonus $c\sqrt{\log
   N/N_u}$; value backup along the visited path; the root's particle filter is literally Ch. 8 code.
   *Collapsible:* convergence statement, and particle-deprivation caveats deep in the tree
   (transplanting the Ch. 8 lesson). **DESPOT** follows as a 1-page contrast: $K$ determinized
   scenarios shared across the tree, regularized lower/upper bounds, anytime guarantees — described
   precisely, implemented optionally.
7. **AMDP / coastal navigation.** *Statement:* compress $b \mapsto \bar b = (\arg\max_x b(x),
   H(b))$ with entropy $H(b) = -\sum_x b(x)\log b(x)$; learn/estimate transitions in the augmented
   space by simulation; run Ch. 21 value iteration over the $(pose, H)$ grid. *Sketch:* the
   compression, the simulated-transition estimation (`Augmented_MDP_value_iteration()`), and the
   emergent behavior: paths that trade length for low predicted entropy — wall-hugging. Honest
   caveat box: $H$ alone cannot distinguish *which* mode a bimodal belief is in — the exact failure
   w24.2 will exploit.

**Named algorithms:**

| Algorithm | Signature | Complexity |
|---|---|---|
| `finite_world_POMDP` (Thrun Table 16.1) | `(pomdp, T) -> Γ_1..Γ_T` | $O(|\mathcal{U}||\Gamma|^{|\mathcal{Z}|})$ vectors/backup before pruning |
| `prune` | `(Γ) -> Γ_min` | pairwise dominance $O(|\Gamma|^2 |\mathcal{X}|)$ + witness search |
| `qmdp` | `(mdp_q: &[[f64; A]]) -> Γ_{QMDP}` | one Ch. 21 solve |
| `pbvi` | `(pomdp, B, n_iter) -> Γ` | $O(|B| \cdot |\mathcal{U}| \cdot |\mathcal{Z}| \cdot |\Gamma|)$ per iteration |
| `pomcp_search` | `(b: ParticleSet, n_sims, c_ucb) -> u` | $O(n_{sims} \cdot \text{depth})$; anytime |
| `Augmented_MDP_value_iteration` (Thrun Table 16.5 ctx) | `(sim, grid) -> V(pose, H)` | Ch. 21 VI over augmented grid |

## 4. Conceptual (C) — Intuition & Visual Design

- **Widget w22.1: Tiger Door Console** *(flagship, interactive sim)* — type: wasm-sim. Top: the two
  doors, a Listen button (plays a positional growl cue), Open buttons, and a running score. Bottom,
  always visible: the belief segment $[0,1]$ with the current belief as a draggable point
  (prior-blue), and beneath it the $\alpha$-vector envelope — every vector a line over the segment,
  the upper envelope bold, intervals colored by optimal action. Listening visibly walks the belief
  point along the segment *and the reader watches which linear piece they're standing on*. Modes:
  **You play** (autoplay demo first: the widget plays one greedy game and loses -100, then hands
  over control) / **Optimal pilot** (policy plays, annotated) / **QMDP pilot** (never listens —
  watch it coin-flip doors from $b = 0.5$). One meaningful parameter: hearing accuracy slider
  (0.5–1.0); at 0.5 the envelope's listen-interval vanishes — sensing that teaches nothing is worth
  nothing. *Misconception killed:* "sensing is free / more sensing is always optimal," and "you can
  act on the MAP state" (QMDP mode is the executioner).
- **Widget w22.2: Coastal Navigator** *(flagship, interactive sim)* — a hall in the Apartment: a
  featureless open center, walls with texture the LiDAR can localize against. Rusty must cross to a
  goal doorway. Two ghost runs race (autoplay): shortest-path pilot (gray dashed, Ch. 20) vs. AMDP
  pilot (purple). Along each trajectory an *entropy ribbon* (width $\propto H(b)$, MCL running
  live underneath) shows the shortest path ballooning with uncertainty mid-hall and the AMDP path
  hugging the wall, staying thin, and arriving *later but reliably* — success tally over seeded
  runs displayed. One parameter: sensor range slider (long range dissolves the effect — coastal
  behavior is an artifact of *when* sensing is informative). *Misconception killed:* "the optimal
  path under uncertainty is the shortest collision-free path."
- **Widget w22.3: Alpha Forge** *(interactive sim, supporting)* — exact VI on tiger, one backup per
  click: candidate vectors flash in (count readout: 3 → 27 → …), dominated ones fade under pruning
  (toggle pruning off to feel the explosion; counts from the real solver). Switch to **PBVI mode**:
  drag a handful of belief points onto the segment; only their argmax vectors survive; watch the
  envelope degrade gracefully as you remove points. *Misconception killed:* "exact POMDP solving is
  a scaling problem" — it is a combinatorial one; approximation is structural, not lazy.
- **Widget w22.4: POMCP Tree Peek** *(animation, supporting)* — the search tree grows from a root
  belief (particle cloud drawn in the root node) during 2000 simulations; node size = visit count;
  the tree visibly concentrates on promising action branches; hover a node to see its particle
  belief. Scrub slider over simulation count. *Misconception killed:* "online planning means
  exhaustive lookahead" — it means *sampled, bandit-guided* lookahead.

Dashboard note: w22.1 and w22.3 share the belief-segment component (built once in the widget
framework); the F section's PWLC equations color-match the envelope rendering. All widgets autoplay,
one headline parameter each, static SVG fallbacks generated from the same code at build time.

## 5. Practical (P) — Rust Implementation

Crates:
- `nalgebra` 0.35 — `SVector`/`SMatrix` with const generics: the POMDP's dimensions are types.
- `rand` 0.9 + `rand_distr` 0.6 — seeded simulation, POMCP rollouts.
- `rayon` (native only) — parallel POMCP simulation batches; single-thread on WASM.
- `eframe`/`egui` 0.35 + `egui_plot` 0.34 — the four widgets.
- Depends on `ch08_particles` (particle sets, low-variance resampler), `localize` (Ch. 12 MCL for
  the coastal lab), `ch21_mdp` (VI for QMDP and AMDP).

Module plan: `crates/ch22_pomdp/` with `src/model.rs`, `exact.rs` ($\alpha$-vector VI + pruning),
`qmdp.rs`, `pbvi.rs`, `pomcp.rs`, `amdp.rs`, `tiger.rs` (the canonical instance),
`examples/tiger.rs`, `examples/coastal.rs`, `examples/corridor_commit.rs`.

```rust
use nalgebra::{SMatrix, SVector};

/// A finite POMDP with S states, A actions, Z observations — dimensions checked at compile time.
pub struct FinitePomdp<const S: usize, const A: usize, const Z: usize> {
    pub t: [SMatrix<f64, S, S>; A],   // t[u][x'][x] = p(x' | x, u)
    pub o: [SMatrix<f64, Z, S>; A],   // o[u][z][x'] = p(z | x', u)
    pub r: SMatrix<f64, S, A>,        // r(x, u)
    pub gamma: f64,
}

pub type Belief<const S: usize> = SVector<f64, S>;

pub struct AlphaVec<const S: usize> { pub v: SVector<f64, S>, pub action: usize }

/// Bayes filter step in matrix form; returns (b', p(z | b, u)). Ch. 5, reborn as τ.
pub fn belief_update<const S: usize, const A: usize, const Z: usize>(
    m: &FinitePomdp<S, A, Z>, b: &Belief<S>, u: usize, z: usize,
) -> (Belief<S>, f64);

/// Thrun Table 16.1: exact value iteration; returns Γ_1..Γ_T with per-horizon pre/post-prune counts
/// (the numbers w22.3 displays).
pub fn exact_vi<const S: usize, const A: usize, const Z: usize>(
    m: &FinitePomdp<S, A, Z>, horizon: usize,
) -> Vec<PrunedSet<S>>;

/// Online planner over the book's particle machinery. `G` is the generative model —
/// for the labs, literally the Ch. 4 simulator.
pub struct Pomcp<X: Clone, G: Fn(&X, usize, &mut SmallRng) -> (X, u32, f64)> {
    pub c_ucb: f64, pub max_depth: usize, gen: G, tree: Tree<X>,
}
impl<X: Clone, G: ...> Pomcp<X, G> {
    pub fn search(&mut self, root: &ch08_particles::ParticleSet<X>, n_sims: usize) -> usize;
    pub fn advance(&mut self, u: usize, z: u32);  // prune tree to the reached node, re-root belief
}
```

Pruning honesty note (stated in the text): full LP pruning would drag in a MILP/LP dependency; for
the book's problem sizes we implement pointwise dominance plus Lark's witness-point test with a
simplex-corner search, and the text says exactly what this misses versus the LP formulation.

Worked end-to-end example: (1) `cargo run --example tiger` — exact solve to horizon 50; prints
per-horizon vector counts (3, then 27→pruned, …), the converged envelope's action intervals (the
listen band and open thresholds, asserted in a regression test to $10^{-3}$ and matching the
structure in Kaelbling et al. 1998), and a 10,000-game tournament: optimal vs. QMDP vs. always-open
(QMDP's average return is catastrophically negative — printed, not claimed). (2)
`examples/corridor_commit.rs` — the chapter's flagship behavior: Rusty at a T-junction with a
bimodal MCL belief (two symmetric corridors); POMCP over 300 particles, generative model = the
Ch. 4 simulator; the chosen policy *drives past a door to disambiguate first*, then commits;
tournament vs. QMDP-style commit shows +38% success. This is the TOC's promise "Rusty choosing to
relocalize before committing to a corridor," delivered as a reproducible seeded experiment.

Runnable artifact: the two examples above; the WASM demos are w22.1/w22.3 (running `exact.rs`) and
w22.4 (running `pomcp.rs`).

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w22.1 | Tiger Door Console | wasm-sim | ch22_pomdp + eframe 0.35 + egui_plot 0.34 | listen/open buttons, drag belief, accuracy slider, pilot modes (you/optimal/QMDP) | belief-space policies; value of information; PWLC envelope |
| w22.2 | Coastal Navigator | wasm-sim | ch22_pomdp + localize (Ch. 12 MCL) + sim + eframe | play race, sensor-range slider, seed re-roll | uncertainty-aware paths emerge from planning in belief space |
| w22.3 | Alpha Forge | wasm-sim | ch22_pomdp + eframe | step backups, pruning toggle, drag PBVI belief points | exact VI's explosion; point-based approximation |
| w22.4 | POMCP Tree Peek | animation | ch22_pomdp + ch08_particles + eframe | scrub simulation count, hover nodes | sampled bandit-guided lookahead over particle beliefs |
| f22.5 | Belief update as door-listening (3-step strip) | static-svg | plotters (build-time) | — | τ = the Ch. 5 Bayes filter, numerically |

## 7. Exercises & Extensions

1. **(F)** Compute $\Gamma_2$ for the tiger problem by hand (all 27 candidates, then prune to the
   survivors); check your surviving set against `exact_vi`'s horizon-2 output.
2. **(F)** Prove that $V_T(b)$ is convex in $b$ directly from the definition (without the
   $\alpha$-vector induction), and explain in one paragraph why convexity means "certainty is
   worth money."
3. **(C, predict-then-verify)** In w22.1, predict the hearing-accuracy threshold below which the
   optimal policy never listens; verify by bisection with the slider. Then explain why the
   threshold depends on $\gamma$.
4. **(C)** In w22.2, find the sensor range at which the coastal detour stops paying. Relate what
   you observe to the AMDP entropy ribbon — and describe a belief state where the AMDP compression
   $(\arg\max, H)$ would lie to the planner.
5. **(P)** Implement QMDP (`qmdp.rs` is a stub in the repo) on top of Ch. 21's `value_iteration`
   and reproduce the chapter's tournament table.
6. **(P, stretch)** Implement DESPOT's determinized scenarios ($K$ shared random-seed streams) as
   an alternative `search` in `pomcp.rs`; compare regret vs. simulations against POMCP on tiger and
   on the corridor-commit world.

## 8. Modernization Notes

- Kept from the draft baseline: the finite-world exact VI (Table 16.1) with its PWLC derivation —
  still the field's intellectual foundation and unmatched pedagogy — and §16.5's AMDP/coastal
  navigation, which the published 2005 edition also carried and which remains the cleanest
  demonstration that uncertainty-aware behavior *emerges* rather than being scripted.
- Dropped: §16.4's Monte-Carlo POMDP (nearest-neighbor value learning over particle beliefs) — an
  honest 1999 idea superseded on every axis by the POMCP/DESPOT line, which this chapter adds as
  the practical path (both post-2005: Silver & Veness 2010; Somani et al. 2013). The LP-based
  pruning machinery (§16.2.4) is compressed to a derivation note with dominance-based pruning in
  code. General continuous-space value iteration (`POMDP(T)`, Table 16.2) is summarized in one
  boxed paragraph as the limit object, not implemented.
- Added beyond both draft and 2005 edition: QMDP as the diagnostic baseline; PBVI with its error
  bound (2003 — post-dates the draft, contemporaneous with the published edition but absent from
  it); SARSOP as a pointer; belief-space planning vocabulary (reachable beliefs) that Ch. 24's
  active-SLAM formulation will reuse. Deliberately excluded: continuous-observation POMDP solvers
  (POMCPOW et al.) and belief-space trajectory optimization — one pointer paragraph each; Ch. 23
  covers the practical continuous-control ground.
