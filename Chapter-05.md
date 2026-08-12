# Chapter 5 — The Bayes Filter: Recursive State Estimation

> Part II — The Bayes Filter Family · Estimated length: 9 web pages · Difficulty: Foundational

## 1. Purpose & Story Arc

This is the keystone chapter: every algorithm in the remaining 21 chapters is the Bayes filter wearing
a different representation. The reader arrives having driven Rusty (Ch. 4) and watched encoders drift;
the hook is that neither data stream alone suffices — odometry diverges without bound, and a single
door sighting is ambiguous on its own — but *fused recursively* they localize a robot that never knew
where it started. The "aha": belief updating is exactly two alternating operations — **a convolution
that smears and a pointwise product that sharpens** — and that two-step recursion, derived honestly by
induction, is the entire field in embryo. Reader path follows the book rhythm: Hook → C (play) →
F (rigor) → P (Rust) → Integration lab → Exercises.

Story line:
1. **Problem** — Rusty dead-reckons down the Hallway; the pose estimate walks off the map (autoplay
   failure widget). One sense event alone can't fix it: three identical doors, three hypotheses.
2. **Play** — w5.1 Hallway Belief Machine, the Ch. 1 preview widget returned in full: sense/move,
   belief morphing, entropy falling and rising.
3. **Intuition** — "sensing sharpens, moving smears": product vs. convolution, made visible (w5.2);
   controls and measurements as two opposing information streams.
4. **Formalism** — state, completeness, the Markov assumption; generative laws; $bel(x_t)$ and
   $\overline{bel}(x_t)$; the recursion derived by induction (Thrun 2.4.3, done fully).
5. **Algorithm** — `Bayes_filter` as a schema, not an algorithm: what must be supplied (models,
   representation) and what is inherited (correctness).
6. **Implementation** — the `BayesFilter` Rust trait the whole book implements; the discrete hallway
   filter as first `impl`; unit test reproduces the chapter's 3-step numeric example digit for digit.
7. **Experiment** — Integration lab in the Hallway world; deliberately break the Markov assumption
   (an unmodeled pedestrian trips the sensor) and watch the posterior go confidently wrong; repair
   by state augmentation. Close with the family tree of Part II–V filters.

## 2. Prerequisites & Position

- **Builds on:** Ch. 2 (Bayes rule with background knowledge, conditional independence, entropy;
  widgets w2.1/w2.2 are the vocabulary here), Ch. 3 (poses as states, lightly), Ch. 4 (Rusty, the
  Hallway world, encoder/sensor noise phenomenology, widget framework).
- **Feeds into:** Ch. 6–8 (KF/EKF/UKF/histogram/particle filters are `impl BayesFilter`), Ch. 11–12
  (localization = Bayes filter + map), Ch. 13 (binary static-state variant), Ch. 14/17 (SLAM filters),
  Ch. 15 (the *non*-recursive alternative: full-trajectory smoothing), Ch. 21–22 (beliefs as the
  states of decision problems), Ch. 25 (the recursion as a differentiable computation graph).
- **Baseline sources:** Thrun et al. (1999–2000 draft) Ch. 2, specifically §2.3 (state, environment
  interaction, generative laws, beliefs), §2.4 (Bayes filter algorithm, example, full induction
  derivation §2.4.3, Markov assumption §2.4.4), §2.5 (representation and computation). The §2.2
  probability refresher was already covered in our Ch. 2. Pedagogy: Labbe Ch. 2 (discrete Bayes),
  roboticsbook.org §4.4, Udacity CS373 hallway numerics.

## 3. Foundation (F) — Mathematical Core

**Notation introduced** (chapter-scoped table at top of F section; all per TOC.md):
$x_t$ (state), $u_t$ (control), $z_t$ (measurement), $x_{0:t}, u_{1:t}, z_{1:t}$ (histories),
$bel(x_t)$, $\overline{bel}(x_t)$, $\eta$, $p(x_t \mid x_{t-1}, u_t)$, $p(z_t \mid x_t)$,
$H[bel]$ (belief entropy, bits — our instrumentation extension).

**Definitions (each gets a display box):**
- *State & completeness*: $x_t$ is **complete** if no variables prior to $x_t$ influence the future
  once $x_t$ is known — the Markov property. Discussion: pose alone vs. pose+velocity vs. "state of
  the universe"; completeness is a modeling *choice* with a price (dimension).
- *Two data streams*: controls $u_t$ (information about state **change**; tends to increase
  uncertainty) vs. measurements $z_t$ (information about the state **now**; tends to decrease it).
  Timing convention fixed: $u_t$ then $z_t$ within step $t$.
- *Probabilistic generative laws*: motion model $p(x_t \mid x_{t-1}, u_t)$, measurement model
  $p(z_t \mid x_t)$ — the two distributions a practitioner must supply (Chs. 9–10 build them).
- *Belief and predicted belief*:
  $bel(x_t) = p(x_t \mid z_{1:t}, u_{1:t})$, $\overline{bel}(x_t) = p(x_t \mid z_{1:t-1}, u_{1:t})$.
- *Initial belief* $bel(x_0)$: point mass (tracking), uniform (global localization), or anything
  between — the taxonomy Ch. 11 formalizes.

**Derivations** (skeleton inline; full algebra in collapsible blocks):

1. **Measurement update.** *Statement:* $bel(x_t) = \eta\, p(z_t \mid x_t)\, \overline{bel}(x_t)$.
   *Sketch (4 steps):* (i) expand $bel(x_t)$ by Bayes rule conditioning on $z_t$ with background
   $z_{1:t-1}, u_{1:t}$; (ii) invoke the Markov assumption: $p(z_t \mid x_t, z_{1:t-1}, u_{1:t}) =
   p(z_t \mid x_t)$; (iii) recognize the denominator as $\eta^{-1}$, independent of $x_t$;
   (iv) recognize the remaining factor as $\overline{bel}(x_t)$. *Collapsible:* the full conditional-
   Bayes expansion with every conditioning set written out, plus why $\eta$ can be computed by
   normalization after the fact.
2. **Prediction (control update).** *Statement:*
   $\overline{bel}(x_t) = \int p(x_t \mid x_{t-1}, u_t)\, bel(x_{t-1})\, dx_{t-1}$.
   *Sketch (3 steps):* (i) law of total probability over $x_{t-1}$; (ii) Markov: given $x_{t-1}$ and
   $u_t$, older data is irrelevant to $x_t$; (iii) drop $u_t$ from the conditioning of
   $bel(x_{t-1})$ — a *future* control carries no information about a *past* state (subtle; gets a
   call-out box because readers always ask). *Collapsible:* full expansion, discrete-sum variant.
3. **Correctness by induction** (the chapter's centerpiece, Thrun §2.4.3). *Statement:* if
   $bel(x_0) = p(x_0)$, the state is complete, and the models are correct, then the two-step
   recursion yields $bel(x_t) = p(x_t \mid z_{1:t}, u_{1:t})$ for all $t$. *Sketch (5 steps):*
   base case $t=0$; assume for $t-1$; apply derivation 2 to get $\overline{bel}(x_t)$ exact; apply
   derivation 1 to get $bel(x_t)$ exact; conclude by induction. *Collapsible:* the fully-annotated
   chain with each equality tagged by the rule that licenses it (Bayes / total probability / Markov)
  — tags color-matched to the equation panel in w5.1.
4. **When Markov lies.** Not a theorem — a structured taxonomy with one example each: unmodeled
   dynamics (people moving through the Hallway), model error, representation/approximation error
   (foreshadows Chs. 6–8), software abstraction variables. Repair: **state augmentation**, with the
   pedestrian-position augmentation worked as the example. Links forward to Ch. 12 §dynamic
   environments.

**Named algorithm:**

- `Bayes_filter(bel(x_{t-1}), u_t, z_t) → bel(x_t)` — Thrun Table 2.1. Two lines: predict
  (integral/sum), correct (product + normalize). Presented explicitly as a **schema**: it becomes
  executable only once (a) a belief representation and (b) the two models are chosen — the book's
  layered-instantiation device, stated here once and reused in every later chapter.
- Finite-state instantiation (preview of Ch. 8's `Discrete_Bayes_filter`): with $K$ states, predict
  is $O(K^2)$ in general, $O(K \cdot W)$ for a banded motion kernel of width $W$; correct is $O(K)$.

**Numeric micro-example** (the chapter's contract with the unit test; kalmanfilter.net discipline):
cyclic 8-cell hallway, doors at cells $\{1,2,6\}$ (0-indexed), uniform prior $0.125$; sensor
$p(z{=}\text{door} \mid \text{door}) = 0.75$, $p(z{=}\text{door} \mid \text{wall}) = 0.15$; motion
"right 1" kernel $[0.1, 0.8, 0.1]$ (undershoot/exact/overshoot). Three steps, full 8-vector printed
at each step: **sense** → doors $0.25$, walls $0.05$ ($H$: $3.000 \to 2.580$ bits); **move** →
$[0.07, 0.07, 0.23, 0.23, 0.07, 0.05, 0.07, 0.21]$ ($H \to 2.739$: smearing); **sense** → peak
$bel(2) \approx 0.4637$ ($H \to 2.335$: only the door-pair alignment at cells 1–2 survives both
sightings). Every number checkable by hand; the Rust test asserts all three vectors to $10^{-4}$.

**Representation & computation trade-offs** (closing F subsection): the family tree. Axes:
parametric (Gaussian: Chs. 6–7) vs. nonparametric (grid/particles: Ch. 8); exact vs. approximate;
per-step cost; multimodality. Table + forward pointers, rendered as widget w5.3.

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor (one, carried end-to-end): **"sensing sharpens, moving smears."** Book color code
throughout: prior **blue**, prediction **orange**, measurement/likelihood **green**, posterior
**purple**, ground truth **gray dashed**. All widgets autoplay a seeded default run, expose one
headline parameter, and ship a build-time static SVG fallback (plotters) per Appendix D.

- **Widget w5.1: Hallway Belief Machine** — *flagship, interactive wasm-sim* (full version of the
  Ch. 1 preview). **Manipulates:** `sense` / `move` buttons; autoplay toggle (default: on, looping a
  12-step recorded run); a time scrubber over that run; headline slider — sensor reliability
  $p(z{=}\text{door}\mid\text{door})$; secondary sliders behind a "more" disclosure — motion kernel
  spread, hallway cyclic on/off; `kidnap` button; **"invisible pedestrian" toggle that injects
  unmodeled door-like readings, breaking the Markov assumption on purpose**; seed re-roll.
  **Observes:** the belief histogram morphing (blue → orange after move; green likelihood strip
  glows over door cells during sense; purple posterior); Rusty's true cell as a gray dashed marker;
  a linked entropy sparkline; the equation panel highlighting whichever term (product vs. integral)
  is currently executing, colors matched. **Misconceptions killed:** "the robot must know where it
  starts" (uniform prior converges anyway); "sensing always disambiguates" (three doors → three
  peaks; only motion + sensing resolves aliasing); "a confident belief is a correct belief" (the
  pedestrian mode produces a sharp, wrong posterior).
- **Widget w5.2: Sharpen & Smear Scope** — *interactive animation.* One belief curve stepped through
  exactly one predict and one correct in slow motion. **Manipulates:** drag the motion-kernel width;
  drag the likelihood sharpness; a "η" step button that shows the un-normalized product being scaled
  up. **Observes:** convolution visibly transporting and flattening mass (entropy counter ticks up);
  pointwise product carving the curve (entropy ticks down); $\eta$ visibly changing *nothing* about
  the shape. **Misconceptions killed:** "both filter steps reduce uncertainty"; "$\eta$ is a fudge
  factor with physical meaning."
- **Widget w5.3: The Filter Family Tree** — *static-svg with hover reveals.* Map of Part II–V:
  representation axis × exactness, nodes = KF (Ch. 6), EKF/UKF (Ch. 7), histogram/particle (Ch. 8),
  their localization/SLAM descendants (Chs. 11–17), and the smoothing branch (Ch. 15) drawn
  deliberately *off* the recursion trunk. **Misconception killed:** "the Bayes filter is one
  algorithm" — it is a schema with a family of instantiations.

Dashboard layout: w5.1 full-width at top of the C section; entropy sparkline docked beneath it;
color-coded recursion equations in a side panel that stays sticky while the reader scrolls the F
derivations (equation ↔ widget ↔ derivation share the same term colors).

## 5. Practical (P) — Rust Implementation

**Crates:** `nalgebra` 0.35 (fixed-size `SVector` beliefs, const-generic cell count), `rand` 0.9 +
`rand_distr` 0.6 (seeded `Pcg64` for the simulated sensor/motion draws — reproducible demos, WASM-
clean), `egui`/`eframe` 0.35 + `egui_plot` 0.34 (widgets), `plotters` (static SVG fallbacks). No
filtering crate: the point is to write it.

**Module plan:**
- `crates/bayes_core/` — **introduced by this chapter, depended on by every later chapter**: the
  `BayesFilter` trait, `BeliefLike` trait, entropy helpers.
- `crates/ch05_hallway/` — `HallwayFilter`, the worked example as both `example` and unit test.
- `demos/ch05-demo/` — eframe app hosting w5.1/w5.2 (one crate, tabbed, lazy-iframe embedded).

**Key types & signatures** (compiles-in-spirit):

```rust
// crates/bayes_core/src/lib.rs
/// The contract every filter in this book implements (Chs. 6–8, 11–12, 14, 17).
pub trait BayesFilter {
    type State;
    type Control;
    type Measurement;
    type Belief: BeliefLike<Self::State>;

    /// bel¯(x_t) ← ∫ p(x_t | x_{t-1}, u_t) bel(x_{t-1}) dx_{t-1}   [prediction: orange]
    fn predict(&mut self, u: &Self::Control);
    /// bel(x_t) ← η p(z_t | x_t) bel¯(x_t)                          [correction: purple]
    fn correct(&mut self, z: &Self::Measurement);
    fn belief(&self) -> &Self::Belief;
}

pub trait BeliefLike<S> {
    fn mode(&self) -> S;      // MAP point estimate
    fn entropy(&self) -> f64; // bits; powers every entropy sparkline in the book
}

// crates/ch05_hallway/src/lib.rs
pub struct HallwayBelief<const K: usize>(pub nalgebra::SVector<f64, K>);

pub struct HallwayFilter<const K: usize> {
    pub bel: HallwayBelief<K>,
    pub doors: [bool; K],
    pub p_hit: f64,               // p(z=door | door)
    pub p_false: f64,             // p(z=door | wall)
    pub kernel: [f64; 3],         // [undershoot, exact, overshoot]
    pub cyclic: bool,
}

pub enum Shift { Left, Right }
pub struct DoorSense(pub bool);

impl<const K: usize> BayesFilter for HallwayFilter<K> {
    type State = usize;
    type Control = Shift;
    type Measurement = DoorSense;
    type Belief = HallwayBelief<K>;
    fn predict(&mut self, u: &Shift) { /* banded convolution, O(K·3) */ }
    fn correct(&mut self, z: &DoorSense) { /* pointwise product + normalize, O(K) */ }
    fn belief(&self) -> &HallwayBelief<K> { &self.bel }
}
```

**Worked end-to-end example:** `cargo run --example three_steps -p ch05_hallway` builds the 8-cell
filter above, executes sense → move → sense, and prints the three belief vectors plus entropies —
exactly the F-section table. `#[test] fn reproduces_worked_example()` asserts every entry to
$10^{-4}$ (this test is the template for the book-wide "numeric example = unit test" rule).
A second example, `global_localization`, runs 40 autoplay steps from a uniform prior and prints the
step at which $bel$ concentrates $> 0.9$ on the true cell.

**Runnable artifact:** the WASM demo is w5.1 itself — compiled from `ch05_hallway`, making the
book's signature claim ("the widget is the chapter's code") true from the first algorithmic chapter.
The Integration lab drops the same filter into the Ch. 4 `sim` Hallway with Rusty's real noisy
encoders/door sensor, then flips on the pedestrian to demonstrate Markov violation and its
state-augmentation repair.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w5.1 | Hallway Belief Machine | wasm-sim | ch05_hallway + sim + eframe/egui_plot, seeded Pcg64 | sense/move, scrub, reliability slider, kidnap, Markov-breaker, seed re-roll | the full recursion; sensing sharpens / moving smears; aliasing; confident-but-wrong beliefs |
| w5.2 | Sharpen & Smear Scope | interactive animation | ch05_hallway + eframe | drag kernel width & likelihood sharpness; step η | convolution vs. product; entropy bookkeeping; η is cosmetic |
| w5.3 | The Filter Family Tree | static-svg (hover) | plotters build-time + CSS hover | hover nodes | Bayes filter as schema; map of Part II–V |
| — | Dead-reckoning drift hook | animation (autoplay) | sim + eframe | none (replay) | why fusion is necessary (chapter hook) |

## 7. Exercises & Extensions

1. **(F)** Re-derive the recursion when the data order within a step is $z_t$ *before* $u_t$
   (measurement-first convention). Show which conditional-independence claims change and which
   don't, and write the resulting two-step algorithm.
2. **(F)** A door sensor's errors are correlated in time (sticky misreads). Show the Markov
   assumption fails for $x_t = $ pose alone; propose an augmented state that restores it and write
   the new generative laws.
3. **(C)** Predict-then-verify in w5.1: with reliability $0.75/0.15$ and doors $\{1,2,6\}$, where
   will the belief peak after sense–move–sense, and with what mass? Verify against the widget and
   the printed example. Then find a door layout for which this sequence *cannot* disambiguate.
4. **(C)** Enable the invisible pedestrian in w5.1 and measure (entropy sparkline) how the wrong
   posterior's confidence compares with the correct run's. Explain why "sharp" ≠ "right" using the
   induction theorem's hypotheses.
5. **(P)** Extend `HallwayFilter` with `Shift::Stay` and a control-dependent kernel; verify with a
   property test that predict never decreases entropy for any symmetric kernel (and find the
   asymmetric counterexample the test suggests).
6. **(P)** Implement `BeliefLike::entropy` for a `Gaussian` newtype (closed form) ahead of Ch. 6,
   and add a criterion micro-benchmark comparing predict cost for $K \in \{10^2, 10^3, 10^4\}$ —
   the empirical seed of the representation trade-off table.

## 8. Modernization Notes

- **Baseline is timeless; this chapter modernizes packaging, not content.** The 1999–2000 draft's
  Ch. 2 recursion, induction proof, and Markov discussion survive intact into 2026 practice
  (Barfoot 2nd ed. still opens with them); we keep the derivation complete rather than condensing.
- **Added vs. baseline:** (i) explicit two-streams framing and entropy instrumentation ($H[bel]$
  is not in Thrun; it powers our widgets and later exploration Ch. 24); (ii) the trait-based
  "schema + instantiation" pattern stated as a first-class design device (Thrun does this
  implicitly across algorithm tables); (iii) the family tree extended with branches the 2005 book
  couldn't draw — smoothing/factor graphs (Ch. 15) and differentiable filters (Ch. 25); (iv) the
  Markov-violation demo made interactive and tied to state augmentation.
- **Dropped/condensed:** the draft's §2.2 probability refresher (lives in our Ch. 2); the draft's
  door-manipulator example replaced by the Hallway so the worked numbers, the flagship widget, and
  the Ch. 1 hook are one artifact; bibliographical-remarks prose becomes a further-reading box.
- **No 2005-vs-draft gap here:** published Ch. 2 ≈ draft Ch. 2; nothing needed sourcing from the
  modernization set beyond framing.
