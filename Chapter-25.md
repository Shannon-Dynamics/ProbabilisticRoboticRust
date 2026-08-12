# Chapter 25 — Learning in the Loop

> Part VII — Frontiers and Integration · Estimated length: 9 web pages · Difficulty: Advanced

## 1. Purpose & Story Arc

Every model in this book so far was written by hand: the beam mixture's four densities, the
$\alpha_1..\alpha_6$ motion noise, the likelihood field's smoothing kernel. This chapter asks the
2026 question honestly: where does machine learning belong in a probabilistic robot? The answer
that organizes everything: **learning replaces the boxes inside the Bayes filter, never the filter
itself** — and a learned $p(z_t \mid x_t, m)$ is only admissible if it is *calibrated*, because the
Bayes filter is an engine that turns stated confidence into decisions. The reader leaves with two
"aha"s: (1) a model can be *accurate* and still poison a filter if it is overconfident — calibration
is a measurable, improvable property, not a vibe; (2) the Bayes filter is itself a differentiable
program, so "tuning $R_t$, $Q_t$" and "training a network" are the same act: gradient descent on a
proper scoring rule through the filter equations. Diffusion policies close the arc: even *acting*
can be a probability distribution you learn to sample.

Story line:

1. Problem: the hand-tuned beam model from Ch. 10 misfits a new sensor; EM refits intrinsics, but
   the functional form itself is wrong (hook widget: reliability diagram of the mis-specified model).
2. Intuition: what "the sensor model is honest" means — reliability diagrams, PIT histograms; an
   overconfident model starves MCL of particles exactly when it is wrong.
3. Formalism: proper scoring rules, calibration vs. sharpness, tempered likelihoods; the
   innovation/evidence decomposition that lets us train noise models *without ground truth*.
4. Algorithm: learned observation models (Thrun's Ch. 9.3 idea, modernized with `candle`);
   post-hoc temperature/variance scaling; differentiable KF (backprop through the gain);
   differentiable PF (soft resampling).
5. Implementation: `crates/ch25_learning` — train a small learned beam model from simulator data;
   differentiate through the Ch. 6 KF; benchmark against the hand-tuned Ch. 10 models on identical
   MCL logs.
6. Experiment: Calibration Clinic and Differentiable Filter Trainer; a closing section that draws
   the principled/alchemy line explicitly, including where diffusion policies sit on it.

## 2. Prerequisites & Position

- **Builds on:**
  - Ch. 2 (entropy, KL divergence, expectation — needed for scoring rules),
  - Ch. 5 (Bayes filter as the frame being filled),
  - Ch. 6 (KF: the differentiable-filter substrate; innovation covariance $S_t$),
  - Ch. 8 (particle filter, resampling — the non-differentiable step we repair),
  - Ch. 9 (motion models: the other learnable box),
  - Ch. 10 (beam model + `learn_intrinsic_parameters` — the baseline we out-learn),
  - Ch. 12 (MCL — the downstream consumer used in every benchmark),
  - Ch. 13 (learned inverse sensor models — this chapter modernizes that idea),
  - Ch. 22–23 (policies and sampling-based control — context for diffusion policies).
- **Feeds into:** Ch. 26 (the capstone exposes a "calibrated learned sensor model" toggle and cites
  this chapter's benchmarks in its retrospective).
- **Baseline sources:**
  - Thrun et al. (1999–2000 draft) Ch. 6 §6.3.2, Table 6.2 `learn_intrinsic_parameters` (ML/EM
    fitting of beam-mixture weights) — recapped as "learning, 1999 style".
  - Thrun et al. draft Ch. 9 §9.3 "Learning Inverse Measurement Models" (§9.3.2 sampling from the
    forward model, §9.3.3 the error function) — the chapter's direct ancestor: train a function
    approximator on simulator-generated $(x, z)$ pairs.
  - Thrun et al. draft Ch. 13 (EM mapping: correspondence as hidden variable) and Ch. 16 §16.4
    (MC-POMDP learned value functions) — cited as historical seeds, not re-derived.
  - Modernization set: Jonschkowski et al., "Differentiable Particle Filters" (RSS 2018);
    differentiable/soft resampling and normalizing-flow DPFs (arXiv:2107.00488); differentiable
    ensemble KF (arXiv:2308.09870); DiffPF (arXiv:2507.15716); DnD filter (arXiv:2503.01274);
    Diffusion Policy (Chi et al., RSS 2023 / IJRR 2025); post-hoc calibration via temperature
    scaling (Guo et al. 2017).

## 3. Foundation (F) — Mathematical Core

**Notation introduced this chapter** (per-chapter table; all other symbols per TOC):

| Symbol | Meaning |
|---|---|
| $\theta$ | learnable parameters (net weights, $\log r$, $\log q$, temperature) |
| $\hat p_\theta(z_t \mid x_t, m)$ | learned observation model (a proper density in $z_t$) |
| $F_\theta(z \mid x)$ | its CDF; $v = F_\theta(z \mid x)$ the PIT value |
| $\mathrm{ECE}$, $\mathrm{acc}(b)$, $\mathrm{conf}(b)$ | expected calibration error; per-bin accuracy/confidence |
| $\kappa$ | likelihood tempering exponent (deliberately **not** $\gamma$, reserved for discount) |
| $\lambda$ | soft-resampling mixture weight (deliberately **not** $\alpha$, reserved for motion noise) |
| $\mathcal{L}(\theta)$ | training loss (negative log score unless stated) |
| $\beta_k$, $\epsilon_\theta$ | diffusion noise schedule; learned noise predictor |

**Definitions** (each gets a formal display box):

- **D25.1 Proper scoring rule.** A score $S(q, z)$ is proper iff
  $\mathbb{E}_{z \sim p}[S(p, z)] \le \mathbb{E}_{z \sim p}[S(q, z)]$ for all $q$; strictly proper if
  equality holds only at $q = p$. The log score $S(q, z) = -\log q(z)$ is strictly proper.
- **D25.2 Calibration (density forecast).** $\hat p_\theta$ is calibrated iff the PIT values
  $v_i = F_\theta(z_i \mid x_i)$ are distributed $\mathcal{U}(0,1)$ under the data distribution.
  Discrete/binary form: $P(\text{hit} \mid \mathrm{conf} = p) = p$ for all $p$.
- **D25.3 Expected calibration error.**
  $\mathrm{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \mathrm{acc}(b) - \mathrm{conf}(b) \right|$
  over $B$ confidence bins (we use $B = 15$ book-wide; reliability diagram = the per-bin plot).
- **D25.4 Tempered measurement update.**
  $bel(x_t) = \eta \, p(z_t \mid x_t, m)^{\kappa} \, \overline{bel}(x_t)$; $\kappa > 1$ is
  overconfidence, $\kappa < 1$ underconfidence. Beam-independence violations act like $\kappa > 1$.
- **D25.5 Differentiable filter.** A Bayes filter whose motion/measurement models (and noise
  parameters) are $\theta$-parameterized differentiable functions, trained by gradient descent on a
  loss over logged trajectories, with gradients propagated through the filter recursion (BPTT).
- **D25.6 Diffusion policy.** An action model $\pi_\theta(u_{t:t+H} \mid o_t)$ represented by a
  learned reverse-diffusion process; sampling an action sequence = iteratively denoising Gaussian
  noise conditioned on the observation. It is a *sampler for a multimodal action distribution* —
  the control-side sibling of the particle filter's "represent by samples" creed.

**Key derivations** (name · statement · sketch · collapsible content):

1. **Propriety of the log score.** *Statement:*
   $\mathbb{E}_{z \sim p}[-\log q(z)] = H(p) + D_{KL}(p \,\|\, q) \ge H(p)$, minimized iff $q = p$.
   *Sketch (4 steps):* write the expected score gap; recognize it as $D_{KL}$; Gibbs' inequality via
   Jensen; equality condition. *Collapsible:* full Jensen argument and the continuous-density
   measure-theoretic caveat. *Payoff:* justifies NLL as the one honest training loss for models
   that will live inside a Bayes filter.
2. **Calibration–sharpness decomposition.** *Statement:* the Brier score decomposes as
   reliability $-$ resolution $+$ uncertainty (Murphy decomposition); NLL admits the analogous
   calibration + refinement split. A model can buy a better score by sharpness only if it stays
   calibrated. *Sketch (5 steps):* bin predictions; add-and-subtract per-bin means; expand the
   square; identify the three terms; map "reliability" to what the reliability diagram plots.
   *Collapsible:* full algebra plus the estimator's binning bias.
3. **Overconfidence poisons the filter.** *Statement:* under a Gaussian measurement model, the
   tempered update D25.4 yields posterior information $\Omega = \Omega_0 + \kappa H_t^\top Q_t^{-1} H_t$
   — confidence grows linearly in $\kappa$ while accuracy does not; in MCL, effective sample size
   collapses as $\kappa$ grows, and recovery from a wrong mode becomes impossible. *Sketch (5
   steps):* Gaussian tempering = precision scaling; posterior covariance shrinks by $\kappa$;
   particle weights $w^{[i]} \propto p^{\kappa}$ sharpen the weight distribution; ESS
   $= 1/\sum (w^{[i]})^2$ falls; connect to Ch. 10's beam-independence overconfidence and Ch. 12's
   particle deprivation. *Collapsible:* the Gaussian algebra and an ESS-vs-$\kappa$ lemma.
4. **Evidence via innovations (train without ground truth).** *Statement:* for the (E)KF,
   $\log p(z_{1:T} \mid u_{1:T}, \theta) = \sum_t \log \mathcal{N}\!\left(z_t;\, H_t \bar\mu_t,\, S_t\right)$
   with $S_t = H_t \bar\Sigma_t H_t^\top + Q_t$ — the marginal likelihood factorizes over
   one-step-ahead prediction errors, so $R_t, Q_t$ (or nets producing them) can be trained by
   maximizing evidence on raw logs, no ground-truth states needed. *Sketch (4 steps):* chain rule
   over $t$; each factor is the predictive density of $z_t$ given the past; that predictive is the
   filter's pre-update Gaussian; sum the log terms. *Collapsible:* full derivation + why this is
   exactly "prediction-error system identification".
5. **Gradient of the Kalman filter.** *Statement:* every KF quantity is a smooth function of
   $\theta = (\log r, \log q, \ldots)$; in particular
   $\partial K_t / \partial \theta$ flows through $S_t^{-1}$ via
   $\mathrm{d}S^{-1} = -S^{-1}(\mathrm{d}S)S^{-1}$, giving a well-defined BPTT recursion.
   *Sketch (5 steps):* write predict/update as a composed map; differentiate $S_t$, then $K_t$,
   then $(\mu_t, \Sigma_t)$; note the recursion carries sensitivities $(\partial\mu_t/\partial\theta,
   \partial\Sigma_t/\partial\theta)$; observe autodiff does this mechanically; state the 1D analytic
   gradients used by the in-browser trainer. *Collapsible:* full matrix-calculus derivation of
   $\partial \mathcal{L} / \partial \log q$ for the NLL and evidence losses (the exact formulas the
   WASM widget implements — the derivation *is* the widget's source).
6. **Soft resampling carries gradient.** *Statement:* multinomial resampling is piecewise-constant
   in the weights (gradient zero a.e.). Sampling indices from the mixture
   $q(i) = \lambda w_t^{[i]} + (1-\lambda)/M$ and reweighting
   $w'^{[i]} = \dfrac{w_t^{[i]}}{\lambda w_t^{[i]} + (1-\lambda)/M}$
   keeps the estimator unbiased while $w'$ depends smoothly on the weights, so gradients reach the
   measurement model through surviving particles. *Sketch (4 steps):* resampling as categorical
   draw; why reparameterization fails for categoricals; the importance-sampling identity that moves
   weight dependence into $w'$; the $\lambda \to 1$ (unbiased-but-blind) vs $\lambda \to 0$
   (high-variance-but-transparent) trade. *Collapsible:* unbiasedness proof.
7. **Diffusion policy objective (statement-level).** *Statement:* with forward noising
   $q(u^k \mid u^{k-1}) = \mathcal{N}(\sqrt{1-\beta_k}\, u^{k-1}, \beta_k I)$, training minimizes
   $\mathbb{E}\left[ \| \epsilon - \epsilon_\theta(u^k, k, o_t) \|^2 \right]$, a bound on the NLL of
   the demonstrated action distribution; multimodality survives because we learn a sampler, not a
   mean. *Sketch (4 steps):* noising destroys structure; the net learns to reverse one step;
   chaining reversals samples the data distribution; contrast with a unimodal Gaussian policy that
   averages the two ways around an obstacle into the obstacle. *Collapsible:* pointer-level ELBO
   sketch only — the full diffusion derivation is out of the book's scope, and we say so.

**Named algorithms** (signature · complexity):

| Algorithm | Signature | Complexity |
|---|---|---|
| `learn_intrinsic_parameters(Z, X, m)` | Thrun Table 6.2; returns $z_{hit}, z_{short}, z_{max}, z_{rand}, \sigma_{hit}, \lambda_{short}$ | $O(\lvert Z\rvert)$ per EM iteration |
| `learn_observation_model(sim, cfg)` | sample $(x, z)$ from the forward model (Thrun §9.3.2), fit $\hat p_\theta$ by NLL descent | $O(N_{samples})$ per epoch |
| `reliability_diagram(scores, outcomes, B)` | returns bins + ECE | $O(N + B)$ |
| `pit_histogram(model, pairs, B)` | PIT values + uniformity check ($\chi^2$) | $O(N)$ |
| `fit_temperature(model, val)` | 1-D optimization of $\kappa$ (equivalently variance scale) on validation NLL | $O(N)$ per step |
| `train_dkf(traj_log, loss)` | BPTT through the Ch. 6 KF; `loss` ∈ {state-NLL (needs ground truth), evidence (does not)} | $O(T (n^3 + m^3))$ per trajectory per epoch |
| `soft_resample(X_t, lambda, rng)` | low-variance sampler over the mixture; returns particles + $w'$ | $O(M)$ |
| `train_dpf(traj_log)` | end-to-end PF training with soft resampling | $O(T \cdot M \cdot c_{net})$ per epoch |

## 4. Conceptual (C) — Intuition & Visual Design

Book color code throughout: likelihood/measurement quantities **green**, posterior **purple**,
prediction **orange**, prior **blue**, ground truth **gray dashed**. Every widget autoplays a
sensible default, has one headline parameter, and ships a static SVG fallback.

- **Widget w25.1: Calibration Clinic** *(flagship — TOC name)* — interactive sim. Three sensor
  models of the same simulated LiDAR (overconfident hand-tuned / Ch. 10 EM-fitted / learned +
  temperature-scaled) drive a live reliability diagram and PIT histogram (green bars, perfect-
  calibration diagonal in gray dashed). Headline control: **one variance-scale slider** that sweeps
  the middle model from overconfident to underconfident; a model picker and a "send to MCL" button
  run 20 seconds of Ch. 12 global localization with the chosen model, showing ESS and a
  particle-deprivation event counter. Reader manipulates: variance scale, model choice. Observes:
  the diagram bending away from the diagonal, ECE readout, and — the punchline — MCL failing with
  the *sharpest-looking* model. **Misconception killed:** "lower variance = better sensor model";
  accuracy without calibration is a lie the filter will act on. Autoplay: slow sweep of the slider
  with the diagram animating.
- **Widget w25.2: Differentiable Filter Trainer** *(flagship — TOC name)* — interactive sim.
  A 1D cart (the Ch. 6 world) tracked by a KF whose $\log r, \log q$ are trainable. Loss curve
  (evidence loss, so no ground-truth cheating) falls live as gradient steps run; left pane shows
  the filter tracking with its purple posterior band adapting from too-tight/too-loose to honest.
  Reader manipulates: **play/pause training** (headline), loss picker (evidence vs. state-NLL),
  learning rate, a "perturb $\theta$" button to kick the parameters and watch re-convergence.
  Observes: loss curve, $(r, q)$ trajectory plotted over the true values (gray dashed), NEES
  approaching 1. **Misconception killed:** "training a filter is alchemy" — the gradient is
  calculus through the same five KF equations, visible in a linked equation panel where the terms
  being differentiated highlight as gradients flow.
- **Widget w25.3: Tempering Lab** — supporting sim. The Ch. 12 MCL Theater with one new slider:
  $\kappa$. Observes ESS, weight histogram, and kidnap-recovery success rate over auto-repeated
  trials. Kills: "multiplying in more/stronger evidence is free"; shows Ch. 10's independence
  warning as a measurable failure.
- **Widget w25.4: Resampling Gradient Microscope** — animation. The PF unrolled as a computation
  graph (three timesteps); gradient flow rendered as animated pulses from loss back to
  measurement-model parameters. Toggle hard vs. soft resampling: hard cuts the pulses at every
  resample node; the $\lambda$ slider revives them at the cost of a visible weight-variance meter.
  Kills: "you can't differentiate through sampling" (and its converse, "soft resampling is free").
- **Widget w25.5: Multimodal Action Sampler** — animation (weights trained offline, replayed).
  Rusty faces an obstacle it can pass on either side. Left: unimodal Gaussian policy trained on
  two-sided demonstrations — its mean drives into the obstacle. Right: diffusion policy — 32
  action-trajectory samples denoise from noise into two clean modes (denoising steps scrubbable).
  Kills: "a mean action is a good action"; shows why a *distribution over actions* is the
  probabilistic-robotics answer on the control side too.

Dashboard note: w25.1 and w25.3 share the MCL substrate and are one eframe app with two tabs;
w25.2 and w25.4 are a second app. No full-page dashboard in this chapter — that is Ch. 26's job.

## 5. Practical (P) — Rust Implementation

**Crates** (versions per TOC stack):

- `candle-core` / `candle-nn` (pinned minor) — small MLPs + autodiff for the learned beam model and
  for native BPTT training; chosen over `burn` for smaller API surface (book records the
  alternative and the TOC allows either).
- `nalgebra` 0.35 — KF math (`SMatrix`), analytic-gradient path.
- `rand` 0.9 / `rand_distr` 0.6 — seeded `SmallRng` everywhere (reproducible training runs).
- `statrs` 0.19 — reference densities/CDFs for PIT computation.
- `egui`/`eframe` 0.35 + `egui_plot` 0.34 — the two demo apps.

**Design decision (recorded for the implementer):** training the learned beam model happens
*natively* with candle; the browser never trains a net. The Differentiable Filter Trainer widget
trains only the 2-parameter 1D KF using the *analytic* gradients derived in F5/F4 (hand-rolled,
~40 lines, no candle in WASM) — the same numbers, a tiny payload, and the derivation doubles as
the source code. The learned beam model ships to the Calibration Clinic as a serialized weight
blob (`safetensors`) evaluated by a dependency-free forward pass (~30 lines: two matmuls + GELU).

**Module plan:** `crates/ch25_learning/`

```
src/
  calib.rs        // reliability diagrams, ECE, PIT, temperature fitting
  learned_beam.rs // LearnedBeamModel: candle training (native) + no-deps inference (wasm)
  diff_kf.rs      // DiffKf1d: analytic-gradient trainer; candle cross-check test (native)
  soft_resample.rs// soft resampler over the Ch. 8 ParticleSet
  policy_demo.rs  // offline-trained diffusion-policy replay data generation (native only)
demos/ch25-calibration-clinic/   // w25.1 + w25.3 (repo-root demos/)
demos/ch25-dkf-trainer/          // w25.2 + w25.4
```

**Key types & signatures:**

```rust
use nalgebra::SVector;

/// Anything that can serve the Bayes filter as p(z | x, m). Implemented by the
/// Ch. 10 BeamModel and LikelihoodField, and by this chapter's learned model.
pub trait ObservationModel {
    fn log_likelihood(&self, z: &Scan, x: &SE2, m: &OccGrid) -> f64;
}

/// Calibration report over a labeled evaluation set.
pub struct CalibrationReport {
    pub bins: [ReliabilityBin; 15],
    pub ece: f64,
    pub mean_nll: f64,
    pub pit_chi2: f64,
}
pub fn calibrate<Mdl: ObservationModel>(
    model: &Mdl, eval: &[(SE2, Scan)], m: &OccGrid,
) -> CalibrationReport;

/// Learned per-beam range density: MLP (expected_range, measured_range) -> mixture params,
/// temperature-scaled post hoc. Inference is dependency-free (runs in WASM).
pub struct LearnedBeamModel { weights: BeamNetWeights, pub temperature: f64 }
impl ObservationModel for LearnedBeamModel { /* forward pass, log-sum-exp mixture */ }

/// Differentiable 1D KF with analytic gradients (the widget's engine).
pub struct DiffKf1d { pub log_r: f64, pub log_q: f64 }
pub enum DkfLoss { Evidence, StateNll }
impl DiffKf1d {
    /// One epoch of BPTT over a logged trajectory; returns (loss, d/d log_r, d/d log_q).
    pub fn loss_and_grad(&self, log: &TrajLog1d, loss: DkfLoss) -> (f64, f64, f64);
    pub fn sgd_step(&mut self, log: &TrajLog1d, loss: DkfLoss, lr: f64) -> f64;
}

/// Soft resampling: mixture proposal lambda*w + (1-lambda)/M, reweighted to stay unbiased.
pub fn soft_resample<S: Clone, R: rand::Rng>(
    particles: &mut ParticleSet<S>, lambda: f64, rng: &mut R,
);
```

**Worked end-to-end example** (`cargo run --example clinic`): generate 50 000 $(x, z)$ pairs in the
Apartment via the Ch. 4 simulator's forward model (Thrun §9.3.2, modernized); fit (a) Ch. 10
`BeamModel::learn_intrinsics`, (b) `LearnedBeamModel` (candle, 2×64 MLP, ~20 s native), (c) (b) +
temperature. Print a benchmark table — held-out NLL, ECE, then MCL-on-identical-logs localization
RMSE and deprivation events for each — and emit reliability-diagram SVGs (plotters). Expected
outcome recorded for the test suite: (b) beats (a) on NLL but is *overconfident* (higher ECE);
(c) wins on all downstream metrics; a unit test reproduces the table's numbers under fixed seed.
This "learned-but-uncalibrated loses downstream" result is the chapter's empirical thesis.

**Runnable artifact:** native example above; WASM demos = the two apps in §4. The book page embeds
the Calibration Clinic first (hook), Trainer mid-chapter.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w25.1 | Calibration Clinic | wasm-sim | eframe, egui_plot, ch25_learning, sim, ch12 MCL | variance-scale slider, model picker, "send to MCL" | calibration ≠ accuracy; overconfidence poisons filters |
| w25.2 | Differentiable Filter Trainer | wasm-sim | eframe, egui_plot, DiffKf1d | play/pause training, loss picker, perturb-θ | the KF is a differentiable program; tuning = training |
| w25.3 | Tempering Lab | wasm-sim | eframe, ch12 MCL substrate | $\kappa$ slider, auto-repeat trials | tempering/independence violations collapse ESS |
| w25.4 | Resampling Gradient Microscope | animation | eframe (canvas painting) | hard/soft toggle, $\lambda$ slider, step scrub | where PF gradients die and how soft resampling revives them |
| w25.5 | Multimodal Action Sampler | animation | eframe; weights trained offline (candle), replayed | policy toggle, denoising-step scrub | action *distributions*; why means fail on multimodal tasks |
| — | fig-25-reliability | static-svg | plotters (from `clinic` example) | none (fallback for w25.1) | reliability diagram anatomy |

## 7. Exercises & Extensions

1. **(F)** Prove the log score is strictly proper (derivation F1 without the collapsible open),
   and exhibit a *non*-proper score a beam model could game (hint: reward peak density only).
2. **(F)** Derive $\partial \mathcal{L} / \partial \log q$ for the 1D KF under the evidence loss
   (F4 + F5). Check it against `DiffKf1d::loss_and_grad` by finite differences — the unit test
   skeleton is provided.
3. **(C, predict-then-verify)** In the Tempering Lab, predict how kidnap-recovery success changes
   at $\kappa = 3$ before touching the slider, and predict whether $\kappa = 0.5$ helps or hurts
   tracking accuracy. Verify; explain both using derivation F3.
4. **(C)** Use the Calibration Clinic to find the temperature minimizing ECE for the learned model;
   predict the MCL RMSE ranking of the three models before pressing "send to MCL".
5. **(P)** Implement `ObservationModel` + `calibrate` for the Ch. 10 `LikelihoodField`. Is it
   calibrated? Where does its PIT histogram bulge, and why (hint: max-range beams)?
6. **(P, stretch)** Wire `soft_resample` into the Ch. 8 `ParticleFilter` and train the beam-model
   temperature end-to-end through 1D-hallway MCL with candle (native). Compare against exercise 4's
   post-hoc temperature: same optimum?

## 8. Modernization Notes

- **What the 1999–2000 draft already had — credit where due:** learning is not grafted onto Thrun;
  the draft fits beam intrinsics by EM (Ch. 6 §6.3.2), trains *neural-network inverse sensor
  models from forward-model samples* (Ch. 9 §9.3 — essentially self-supervised sim-to-model
  learning, remarkably ahead of its time), learns maps with EM (Ch. 13), and learns POMDP value
  functions with nearest-neighbor function approximation (Ch. 16 §16.4). This chapter presents
  §9.3 as its direct ancestor and keeps its architecture (sample the forward model → fit the
  inverse) intact, swapping 2000-era backprop for candle.
- **What the baseline lacked:** calibration as a first-class, measurable property (reliability
  diagrams, ECE, PIT, temperature scaling); differentiable filters (2018+) and the
  soft-resampling repair; evidence/innovation training without ground truth stated as the
  principled system-identification route; diffusion policies (2023+) as probabilistic action
  models; any honest principled-vs-alchemy assessment.
- **Dropped/condensed from the baseline:** Ch. 13's EM mapping machinery (E-step smoothing +
  M-step map) is *not* re-derived — its data-association role was absorbed by factor graphs
  (Ch. 15) and FastSLAM (Ch. 17), and it is cited as history. MC-POMDP (16.4) appears only as an
  ancestor note in the diffusion-policy section.
- **Honesty ledger (the chapter's closing section, kept in the design):** *principled* — learned
  components emitting calibrated densities inside an unchanged Bayes filter; evidence-based noise
  training; post-hoc calibration on held-out data. *Still alchemy in 2026* — end-to-end filters
  whose "beliefs" are uncalibrated activations; learned resampling heuristics without variance
  analysis; sim-to-real transfer of learned sensor models (our own learned model is trained in the
  simulator that also evaluates it — the text must flag this circularity explicitly); diffusion
  policies' uncertainty (samplers, not densities: you can draw actions but not evaluate their
  probability, so they do not compose with Bayes rule the way our sensor models do).
