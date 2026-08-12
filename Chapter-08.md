# Chapter 8 — Nonparametric Filters: Histograms and Particles

> Part II — The Bayes Filter Family · Estimated length: 11 web pages · Difficulty: Intermediate

## 1. Purpose & Story Arc

Gaussians buy speed by betting everything on unimodality — and Part II's opening problem (three
identical doors) is multimodal by construction. This chapter closes the filter family with the
representations that can say "it's *either* here *or* there": chop the state space into cells
(histogram filter), or better, let the samples go where the probability is (particle filter). The
"aha" sequence: (i) a weighted random sample is a legitimate representation of a posterior, and
importance sampling — derived from first principles, not asserted — is the license to sample from
the wrong distribution on purpose; (ii) resampling is not bookkeeping but a survival-of-the-fittest
operator with its own variance pathology, and the low-variance comb fixes in one random number what
the roulette wheel botches in $M$; (iii) even resampling done right kills diversity (particle
deprivation), and adapting $M$ to the belief's true spread (KLD sampling) is the modern discipline.
Along the way, a five-line special case — the binary static-state filter in log odds — is planted
and explicitly labeled as the seed that grows into occupancy grid mapping in Ch. 13.

Story line:
1. **Problem** — the Hallway posterior after one door sighting is three peaks; Ch. 6/7's single
   ellipse must choose one and be two-thirds wrong (autoplay hook: an EKF collapsing onto the
   wrong door while the true posterior sits elsewhere).
2. **Play** — w8.1 Particle Survival Arena autoplaying global localization: dust everywhere →
   three clouds → one cloud.
3. **Intuition** — particles as "hypotheses with mass"; the weight–resample rhythm; the wheel vs.
   the comb (w8.2).
4. **Formalism** — discrete Bayes filter; grid decomposition; binary log-odds filter; importance
   sampling from first principles; the particle filter derivation; resampling variance; deprivation;
   KLD-adaptive $M$.
5. **Implementation** — `HistogramFilter`, `LogOdds`, `ParticleFilter<S>` with pluggable
   proposal/likelihood; the 15-line low-variance resampler; rayon-parallel weights, same code
   single-threaded on WASM.
6. **Experiment** — histogram vs. particles on identical Hallway logs; deprivation lab at small
   $M$; KLD mode watching $M_t$ collapse as the belief condenses.

## 2. Prerequisites & Position

- **Builds on:** Ch. 2 (sampling as representation, seeded RNG discipline), Ch. 5 (`BayesFilter`
  trait; the discrete instantiation previewed there is formalized here), Ch. 6–7 (the parametric
  foils; comparison table), Ch. 4 (Hallway world and sensors).
- **Feeds into:** Ch. 12 (MCL/AMCL = this particle filter + Ch. 9/10 models; KLD returns there at
  full scale), Ch. 13 (occupancy grid mapping = the binary log-odds filter per cell — stated here,
  built there), Ch. 17 (FastSLAM/Rao-Blackwellization rides this machinery), Ch. 22 (POMCP uses
  particle beliefs), Ch. 23 (MPPI as importance sampling over controls — the particle filter's
  twin), Ch. 25 (differentiable resampling pointer).
- **Baseline sources:** Thrun et al. (draft) Ch. 4 in full — §4.1 (histogram filter, Table 4.1;
  continuous-state decomposition §4.1.2–4.1.3; binary static-state filter §4.1.4, Table 4.2),
  §4.2 (particle filter Table 4.3, importance sampling §4.2.2, derivation §4.2.3, properties and
  low-variance sampler Table 4.4, §4.2.4). **KLD-adaptive sampling is not in the draft** (it is
  2005-edition material): sourced from the modernization set — Fox, "Adapting the Sample Size in
  Particle Filters Through KLD-Sampling" (IJRR 2003). Pedagogy: Thrun's Udacity resampling wheel;
  Labbe's particle chapters.

## 3. Foundation (F) — Mathematical Core

**Notation introduced:** grid belief $\{p_{k,t}\}$ over cells $\mathbf{x}_k$ with representative
points $\hat{x}_k$; log odds $\ell_t = \log\frac{p}{1-p}$ (per-cell form $\ell_{t,i}$ reserved for
Ch. 13); particle set $\mathcal{X}_t = \{x_t^{[i]}, w_t^{[i]}\}_{i=1}^{M}$; target $f$ and proposal
$g$ densities; effective sample size $M_{\mathrm{eff}}$; KLD bound parameters $(\varepsilon,
\delta, k)$.

**Definitions:**
- *Histogram/discrete belief*: piecewise-constant density over a finite decomposition; models
  coarsened via representative points ($p(\mathbf{x}_j \mid u_t, \mathbf{x}_k) \approx$ model
  evaluated at $\hat{x}_j, \hat{x}_k$, normalized).
- *Importance weight*: for $x \sim g$, $w(x) = f(x)/g(x)$; requires $f > 0 \Rightarrow g > 0$.
- *Particle representation*: $bel(x_t) \approx \sum_i w_t^{[i]} \delta_{x_t^{[i]}}(x_t)$ —
  a distribution, not a point cloud; expectations are weighted sums.
- *Effective sample size*: $M_{\mathrm{eff}} = 1 / \sum_i (w_t^{[i]})^2$ (normalized weights) — the
  resampling trigger.

**Derivations:**

1. **Discrete Bayes filter is exact on finite spaces.** *Statement:* Ch. 5's recursion with sums:
   $\bar{p}_{k,t} = \sum_j p(\mathbf{x}_k \mid u_t, \mathbf{x}_j)\, p_{j,t-1}$;
   $p_{k,t} = \eta\, p(z_t \mid \mathbf{x}_k)\, \bar{p}_{k,t}$. *Sketch (2 steps):* instantiate the
   integral as a sum; note no approximation entered. *Collapsible:* the histogram *approximation*
   argument for continuous state (density ≈ average over cell; error → 0 with resolution), Thrun
   §4.1.2.
2. **Binary Bayes filter with static state, in log odds** — **the occupancy seed** (Thrun §4.1.4).
   *Statement:* for static binary $x$ with inverse model $p(x \mid z_t)$:
   $\ell_t = \ell_{t-1} + \log\frac{p(x \mid z_t)}{1 - p(x \mid z_t)} - \ell_0$, recovery
   $bel_t(x) = 1 - \frac{1}{1+\exp \ell_t}$. *Sketch (4 steps):* Bayes rule for $x$ and $\neg x$;
   divide the two posteriors (η cancels — the trick); take logs (products → sums); telescope the
   recursion, exposing the prior-correction $\ell_0$. *Collapsible:* full manipulation + why the
   *inverse* model $p(x \mid z)$ is the natural parameterization here. **Explicit forward pointer,
   verbatim in the text:** "run one of these per grid cell and you have occupancy grid mapping —
   Ch. 13 is this box, tiled."
3. **Importance sampling from first principles.** *Statement:* for any event/statistic,
   $\mathbb{E}_f[\phi(x)] = \mathbb{E}_g[w(x)\phi(x)]$ with $w = f/g$; the weighted empirical
   distribution of $g$-samples converges to $f$. *Sketch (3 steps):* write the expectation under
   $f$; multiply and divide by $g$; recognize an expectation under $g$ (support condition flagged).
   *Collapsible:* convergence discussion, variance of the estimator, and why weight degeneracy is
   the price of a poor $g$.
4. **The particle filter targets the posterior** (Thrun §4.2.3). *Statement:* propagating each
   particle through the motion model (proposal $g = p(x_t \mid x_{t-1}, u_t)\, bel(x_{t-1})$) and
   weighting by the likelihood $w_t^{[i]} = \eta\, p(z_t \mid x_t^{[i]})$ yields weighted samples of
   $bel(x_t)$. *Sketch (4 steps):* state target = posterior over trajectories $x_{0:t}$; compute
   target/proposal ratio; watch everything cancel except the likelihood; marginalize histories to
   the filter posterior. *Collapsible:* the full trajectory-space induction, plus the honest
   caveats — finite-$M$ bias, the $\eta$-free implementation via normalize-after.
5. **Resampling and its variance.** *Statement:* resampling converts weights into offspring counts;
   multinomial (roulette) resampling has offspring variance $M w^{[i]}(1-w^{[i]})$; the
   low-variance/systematic comb (single draw $r \sim U[0, M^{-1}]$, pointers $r + (i-1)/M$)
   achieves the minimal integer spread $\lfloor M w^{[i]} \rfloor$ or $\lceil M w^{[i]} \rceil$
   while staying unbiased. *Sketch (4 steps):* offspring expectation under both schemes (both
   $= M w^{[i]}$ — unbiasedness); variance computation for multinomial; comb's deterministic
   stratification bounds each count within 1 of its expectation; note $O(M)$ vs. naive $O(M \log M)$.
   *Collapsible:* full variance algebra; stratified resampling as the intermediate scheme.
6. **Particle deprivation & when not to resample.** *Statement + taxonomy (measurable claims, each
   a widget preset):* resampling with high-variance weights deletes diversity; repeated resampling
   without informative measurements is pure diffusion of support; remedies — resample only when
   $M_{\mathrm{eff}} < M/2$, low-variance sampler, more particles, (forward pointer) injection of
   random particles in Ch. 12's Augmented MCL. *Sketch:* the "no-measurement" thought experiment
   (Thrun §4.2.4) run live in w8.1.
7. **KLD-adaptive sample size** (modernization payload; Fox 2003). *Statement:* to guarantee, with
   probability $1-\delta$, KL divergence $\le \varepsilon$ between the sample-based MLE histogram
   (over $k$ occupied bins) and the true posterior, it suffices that
   $M \ge \frac{k-1}{2\varepsilon}\Big[1 - \frac{2}{9(k-1)} + \sqrt{\tfrac{2}{9(k-1)}}\, z_{1-\delta}\Big]^3$.
   *Sketch (4 steps):* likelihood-ratio statistic for a multinomial is asymptotically
   $\chi^2_{k-1}$; bound KL by the statistic; apply the Wilson–Hilferty $\chi^2$ quantile
   approximation; read off $M$. Consequence: $M$ tracks the *support* of the belief — thousands of
   particles during global uncertainty, dozens once converged. *Collapsible:* Wilson–Hilferty
   details; bin-bookkeeping implementation notes.

**Named algorithms (signatures + complexity):**
- `Discrete_Bayes_filter({p_{k,t-1}}, u_t, z_t) → {p_{k,t}}` — Thrun Table 4.1. Predict $O(K^2)$
  general / $O(K W)$ banded; correct $O(K)$. Curse of dimensionality stated with the $K = r^d$
  arithmetic.
- `binary_Bayes_filter(ℓ_{t-1}, z_t) → ℓ_t` — Thrun Table 4.2. $O(1)$; clamping bounds
  $\ell \in [\ell_{\min}, \ell_{\max}]$ noted as the practical guard Ch. 13 inherits.
- `Particle_filter(𝒳_{t-1}, u_t, z_t) → 𝒳_t` — Thrun Table 4.3. $O(M)$ per step (given $O(1)$
  model evaluations).
- `Low_variance_sampler(𝒳_t, 𝒲_t) → 𝒳_t'` — Thrun Table 4.4. $O(M)$, one random number.
- `KLD_sample_size(k, ε, δ) → M` — from the modernization set (Fox 2003; `KLD_Sampling_MCL` in the
  2005 edition). $O(1)$ given the bin count; integrated into the PF loop as an adaptive stop.

**Numeric micro-example** (unit-test contract): $M = 5$ particles with normalized weights
$[0.10, 0.30, 0.05, 0.40, 0.15]$. (a) $M_{\mathrm{eff}} = 1/\sum w^2 = 1/0.285 = 3.509$ — below
the $M/2{=}2.5$? No: above, so no resample under the threshold rule; the text then forces one for
the exercise. (b) Low-variance resample with $r = 0.15$: pointers $\{0.15, 0.35, 0.55, 0.75,
0.95\}$ against the CDF $[0.10, 0.40, 0.45, 0.85, 1.00]$ select offspring counts
$[0, 2, 0, 2, 1]$ — particles 1 and 3 die, none drawn twice-plus-one. Every arrow drawn in a
figure; w8.2 replays exactly these numbers as its default; the Rust test asserts the counts.

## 4. Conceptual (C) — Intuition & Visual Design

Chapter metaphor: **survival of the fittest** — weights are fitness, resampling is selection, and
selection pressure has side effects. Color code: prior/previous particle cloud **blue**, motion-
propagated cloud **orange**, likelihood field / weight glow **green**, post-resample cloud
**purple**, Rusty's true pose **gray dashed**. All widgets: autoplay seeded defaults, one headline
parameter, static SVG fallback.

- **Widget w8.1: Particle Survival Arena** — *flagship, interactive wasm-sim.* The Hallway with
  $M$ particles as ticks (radius/opacity ∝ weight). **Manipulates:** headline slider — particle
  count $M$ (10 … 5,000, log scale); play/pause + step buttons separating the three phases
  (propagate → weight → resample) so each is visible alone; sensor-sharpness slider; `kidnap`
  button; **KLD mode toggle** (adds an $M_t$ trace and lets the arena shrink its own population);
  seed re-roll. **Observes:** blue → orange smear on propagate; green glow scaling ticks on weight;
  purple survivors cloning/dying on resample; the $M_{\mathrm{eff}}$ meter dipping before each
  resample; **deprivation preset** ($M{=}30$, razor-sharp sensor): every particle near the true
  pose dies in one unlucky weighting and the filter confidently converges elsewhere — the arena's
  version of "confident but wrong." In KLD mode, $M_t$ falls from thousands to dozens as the cloud
  condenses. **Misconceptions killed:** "particles are the trajectory of guesses" (they are a
  *distribution*; the weighted mean readout ≠ any particle); "more resampling = faster
  convergence" (step-mode shows pure-resample diffusion); "more particles is always the fix"
  (KLD mode shows *where* they're needed: spread, not accuracy).
- **Widget w8.2: Resampling Wheel** — *flagship, interactive wasm-sim.* Left: Thrun's roulette
  wheel, arc lengths ∝ the micro-example's weights, spun $M$ times. Right: the same wheel with the
  low-variance comb — $M$ equally spaced spokes dropped after one spin. **Manipulates:** `spin`
  (single trial, animated); `spin ×1000` (batch); drag any weight arc to reshape the distribution;
  $M$ slider. **Observes:** per-trial offspring counts; after batch runs, side-by-side offspring-
  count histograms with the measured variance printed — multinomial's spread vs. the comb's
  within-1-of-expectation bars; expectation line identical on both (unbiasedness, seen not
  asserted). Default weights are exactly the F-section micro-example, so the text, the widget, and
  the test share numbers. **Misconceptions killed:** "resampling is deterministic bookkeeping";
  "the comb is biased because it's less random" (same expectation, smaller variance — variance was
  the enemy).
- **Widget w8.3: Grid Resolution Ladder** — *interactive animation.* The Hallway histogram filter
  at $K \in \{8, 32, 128, 512\}$ side by side on one log. **Manipulates:** step through the run;
  highlight one cell to inspect its coarsened models. **Observes:** localization accuracy vs. a
  flops-per-step meter; the $K{=}8$ filter aliasing two doors the $K{=}128$ filter separates.
  **Misconception killed:** "finer is free" — cost is linear-to-quadratic in $K$ per step and
  exponential in dimension (the meter does the arguing; Ch. 12's grid localization inherits the
  trade).
- **Widget w8.4: Log-Odds Flip-Flop** — *interactive animation* (the occupancy seed made playable).
  One binary cell ("is this Hallway cell occupied?") fed a stream of noisy detections.
  **Manipulates:** sensor hit/false rates; a hand-crank to feed evidence one $z_t$ at a time; toggle
  probability-view ↔ log-odds-view; clamp on/off. **Observes:** in probability view the updates are
  a messy product; in log-odds view the same evidence is literal addition of signed increments;
  with clamp off, 50 consistent readings then 5 contradicting ones barely move a saturated cell —
  with clamp on it recovers. **Misconception killed:** "a static state needs no filter" — and the
  reader leaves already knowing Ch. 13's update rule.

Dashboard: w8.1 heads the chapter and returns in the Integration lab; w8.2 docks beside derivation
5; w8.3 beside derivation 1's collapsible; w8.4 beside derivation 2 with the Ch. 13 pointer as its
caption.

## 5. Practical (P) — Rust Implementation

**Crates:** `nalgebra` 0.35 (state vectors; the Hallway PF uses plain `f64` states to show `S` is
generic), `rand` 0.9 + `rand_distr` 0.6 (seeded `Pcg64`; `Normal`/`Uniform` for propagation),
`statrs` 0.19 (likelihoods; `z_{1-δ}` quantile for KLD), `rayon` (parallel weight evaluation,
**native only** — the identical code runs single-threaded on `wasm32` via `cfg`, per the book's
WASM checklist), `egui`/`eframe` 0.35 + `egui_plot` 0.34, `plotters` (fallbacks).

**Module plan:** `crates/ch08_particles/` (library: `HistogramFilter`, `LogOdds`, `ParticleFilter`,
resamplers, `kld`); demos in `demos/ch08-demo/`. Depends on `bayes_core`, `sim`; consumed later by
`localize` (Ch. 12 MCL), `ch13_occgrid`, `ch17_fastslam`.

**Key types & signatures:**

```rust
use rand::rngs::SmallRng; // seeded; Pcg64 alias used book-wide via `sim::rng`

/// Dynamic-K histogram filter (K is data, not type: maps come in all sizes).
pub struct HistogramFilter {
    pub bel: Vec<f64>,
    pub kernel: Vec<f64>,          // banded motion kernel
}
impl bayes_core::BayesFilter for HistogramFilter { /* sums; O(K·W) / O(K) */ }

/// The occupancy seed: one static binary cell in log odds. Ch. 13 tiles this.
#[derive(Clone, Copy)]
pub struct LogOdds(pub f64);
impl LogOdds {
    pub fn update(&mut self, l_meas: f64, l_prior: f64) { self.0 += l_meas - l_prior; }
    pub fn clamp(&mut self, lo: f64, hi: f64);
    pub fn prob(self) -> f64 { 1.0 - 1.0 / (1.0 + self.0.exp()) }
}

pub struct ParticleSet<S> {
    pub states: Vec<S>,
    pub log_w: Vec<f64>,           // log weights: the numerical-stability habit, taught here
}
impl<S> ParticleSet<S> {
    pub fn normalize(&mut self);                    // log-sum-exp
    pub fn ess(&self) -> f64;                       // M_eff = 1/Σw²
    pub fn weighted_mean(&self) -> S where S: Mean;
}

pub trait Proposal<S> {          // pluggable: Ch. 9's samplers slot in for Ch. 12
    type Control;
    fn sample(&self, x: &S, u: &Self::Control, rng: &mut SmallRng) -> S;
}
pub trait Likelihood<S> {        // pluggable: Ch. 10's models slot in for Ch. 12
    type Measurement;
    fn log_lik(&self, z: &Self::Measurement, x: &S) -> f64;
}

pub struct ParticleFilter<S, P: Proposal<S>, L: Likelihood<S>> {
    pub particles: ParticleSet<S>,
    pub proposal: P,
    pub likelihood: L,
    pub resample_threshold: f64,   // resample iff M_eff < threshold · M
    pub rng: SmallRng,
}
impl<S: Clone + Send, P, L> bayes_core::BayesFilter for ParticleFilter<S, P, L> {
    /* predict: propagate (rayon par_iter on native, plain iter on wasm32)
       correct: log-weights += log_lik; normalize; conditional low_variance_resample */
}

/// Thrun Table 4.4 — the whole algorithm, 15 lines in the book listing.
pub fn low_variance_resample<S: Clone>(set: &ParticleSet<S>, rng: &mut SmallRng) -> ParticleSet<S>;

/// Fox 2003: bins occupied → number of particles that suffices for (ε, δ).
pub fn kld_sample_size(k_bins: usize, epsilon: f64, z_one_minus_delta: f64) -> usize;
```

**Worked end-to-end example:** `cargo run --example hallway_duel -p ch08_particles` runs the
histogram filter ($K{=}128$) and the particle filter ($M{=}1{,}000$) on the *same* seeded Hallway
log (continuous state, noisy shift motion, door sensor): prints step-by-step $M_{\mathrm{eff}}$,
resample events, and both filters' MAP estimates converging to the same cell; then reruns the PF at
$M{=}50$ ten times and reports the deprivation failure rate (expected: a nonzero, seed-stable
count — the honest number the Arena's preset dramatizes). `#[test]`s: the 5-particle resampling
micro-example (offspring counts `[0, 2, 0, 2, 1]` at $r{=}0.15$); unbiasedness of
`low_variance_resample` over $10^5$ trials (mean offspring within CI of $Mw^{[i]}$);
`kld_sample_size` against three tabulated values from Fox's paper.

**Runnable artifact:** WASM demos w8.1/w8.2 compile from this crate — the resampler spinning the
wheel is the 15-line listing. Native `--example` runs use rayon and print a weights-evaluation
throughput comparison (the "fearless parallelism" aside), while the WASM build of the identical
source stays single-threaded — demonstrating, not just claiming, portable determinism per seed.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w8.1 | Particle Survival Arena | wasm-sim | ch08_particles + sim + eframe/egui_plot, SmallRng | M slider, phase-step buttons, sharpness, kidnap, KLD mode, seed re-roll | weight–resample rhythm; ESS; degeneracy & deprivation; adaptive M |
| w8.2 | Resampling Wheel | wasm-sim | ch08_particles + eframe | spin / spin×1000, drag weights, M slider | multinomial vs. low-variance: same mean, different variance |
| w8.3 | Grid Resolution Ladder | interactive animation | ch08_particles + sim + eframe | step run, K comparison, cell inspect | resolution/cost trade; aliasing; curse of dimensionality |
| w8.4 | Log-Odds Flip-Flop | interactive animation | ch08_particles + eframe | hand-crank evidence, rate sliders, view & clamp toggles | binary static filter; additivity in log odds; saturation → the Ch. 13 seed |
| — | Three-peaks hook | animation (autoplay) | ch07_nonlinear + sim | none (replay) | why unimodal filters fail the Hallway; motivates nonparametrics |

## 7. Exercises & Extensions

1. **(F)** Derive $M_{\mathrm{eff}} = 1/\sum_i (w^{[i]})^2$ as the variance-matching equivalent
   sample size, and compute it for the chapter's 5-particle example. Show it hits $M$ exactly when
   weights are uniform and 1 when one particle holds all mass.
2. **(F)** Prove both resampling schemes are unbiased ($\mathbb{E}[\text{offspring}_i] = M
   w^{[i]}$) and that the comb's offspring count is always $\lfloor Mw^{[i]} \rfloor$ or
   $\lceil Mw^{[i]} \rceil$. Conclude the comb's per-particle variance bound and compare with
   multinomial's $Mw(1{-}w)$.
3. **(F)** Telescope the binary log-odds recursion to show order of evidence doesn't matter for a
   static state, and exhibit why the same claim fails the moment the state can change (connect to
   Ch. 13's static-map assumption).
4. **(C)** Predict-then-verify in w8.2: reshape the weights to $[0.96, 0.01, 0.01, 0.01, 0.01]$ and
   predict both histograms for $M{=}5$ before pressing `spin ×1000`. Which scheme ever lets
   particle 1 die, and with what probability?
5. **(C)** In w8.1's deprivation preset, find (by bisection on the $M$ slider) the smallest $M$
   with under 10% failure rate across 20 seeds; then check whether KLD mode's steady-state $M_t$
   agrees with your number, and explain the gap.
6. **(P)** Implement `stratified_resample` (one draw per comb interval) between the two taught
   schemes; extend the wheel widget's batch mode to include it and rank the three measured
   variances. Then implement the `resample_threshold` sweep and plot deprivation rate vs.
   threshold — the empirical version of "when not to resample."

## 8. Modernization Notes

- **Kept whole from the baseline:** draft Ch. 4's complete arc — histogram filter, decomposition
  discussion, binary static-state filter, importance-sampling-first particle filter derivation, and
  the low-variance sampler — this material is genuinely timeless (MCL/AMCL built on it still ships
  as Nav2's default localizer in 2026, an argument the text makes explicitly with a forward pointer
  to Ch. 12).
- **Added beyond the draft:** KLD-adaptive sampling (2005-edition material absent from our draft
  PDF; rebuilt from Fox, IJRR 2003, with the Wilson–Hilferty sketch); ESS-triggered resampling and
  log-domain weights as standard post-2005 hygiene (the draft resamples every step and works in
  linear probabilities); the explicit "occupancy seed" framing of the binary filter (the draft
  buries it as a subsection; we promote it because Ch. 13 tiles it); measured-variance comparison
  of resamplers (the draft states the preference, we instrument it); rayon/WASM parallelism
  discipline as a Rust-specific contribution.
- **Condensed:** the draft's tree/selective-updating decomposition techniques (§4.1.3) to one
  paragraph plus pointers — density trees matter less in 2026 than the grid (Ch. 12) and particle
  descendants that survived; sonar-era illustrations replaced by the Hallway's LiDAR-style door
  sensor per the modernization guidance.
- **Deliberately deferred:** Rao-Blackwellization (Ch. 17, as a theorem with FastSLAM as payoff),
  Augmented-MCL recovery particles (Ch. 12, where the kidnapped-robot problem is formally posed),
  differentiable/soft resampling (Ch. 25 pointer only).
