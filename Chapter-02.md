# Chapter 2 — Probability: The Language of Uncertainty

> Part I — Foundations: The Robot and Its Uncertainty · Estimated length: 8 web pages · Difficulty: Foundational

## 1. Purpose & Story Arc

Everything after this chapter is "apply Bayes rule cleverly to robots," so this chapter must make
the small probability toolkit *owned*, not reviewed: random variables, conditioning, Bayes rule
with background knowledge, expectation/covariance, entropy, and above all the Gaussian — in 1D,
in $n$D, in moments form *and* canonical form — plus the third representation the book leans on
constantly: samples. The "aha": **a Gaussian is not a formula, it is a shape you can drag, and
Bayes rule is not a formula either, it is pointwise multiplication** — and both claims are checked
to twelve decimal places by a unit test the reader can run. This chapter also quietly installs two
book-wide disciplines: the color code applied to *equations* (prior blue × likelihood green =
posterior purple), and the executable-worked-example convention that seeds the whole test suite.

Story line:

1. **Hook** — the door sensor from Ch. 1 said "door." How much should Rusty believe it? Two
   numbers ($p(z\mid \text{door})$, prior) in, one number out — but which formula, and why the
   normalizer?
2. **Discrete machinery** — random variables, joint/conditional, independence, total probability,
   Bayes with background knowledge; the hallway numbers from Ch. 1 recomputed *as* Bayes rule.
3. **Moments** — expectation, variance, covariance; the covariance matrix as the shape of a cloud
   (w2.1 play-first).
4. **The Gaussian** — 1D then $n$D density; iso-ellipses ↔ eigenstructure; linear transforms.
5. **Bayes among Gaussians** — product of Gaussians worked fully in 1D (w2.2); moments vs.
   canonical form; multiplication is addition in canonical form — the information-form duality the
   Kalman chapters will exploit.
6. **Samples as representation** — Monte Carlo, LLN, why 100 samples beat a wrong parametric form
   (w2.3); seeded reproducibility discipline.
7. **Entropy** — uncertainty as a scalar; the Gaussian's entropy; one paragraph of foreshadowing
   (information gain drives exploration in Ch. 24).
8. **Experiment** — implement `Gaussian<N>` and `Canonical<N>`, fuse two sensors, reproduce the
   worked numbers by test.

## 2. Prerequisites & Position

- **Builds on:** Ch. 1 (the hallway numbers, the belief idea, the argmax fallacy — here re-derived
  properly).
- **Feeds into:** Ch. 5 (Bayes filter = this chapter's rules applied recursively), Ch. 6 (Kalman
  filter = Derivations 3–5 industrialized; moments/canonical duality becomes KF/IF), Ch. 8
  (samples-as-representation becomes the particle filter), Ch. 9–10 (densities as motion/sensor
  models), Ch. 11 (Mahalanobis gating), Ch. 24 (entropy → information gain), Appendix B (the
  identities stated here are proved once there).
- **Baseline sources:** Thrun et al. Ch. 2 §2.2 (basic concepts in probability: Bayes rule, η,
  independence, expectation, entropy) — the notation baseline; Thrun et al. Ch. 3 §3.4.1
  (canonical/information parameterization — deliberately pulled forward two chapters); Choset et
  al., statistics appendix (style model for a self-contained probability reference). Pedagogy:
  bzarg (blob metaphor + color-coded equations), kalmanfilter.net (numeric micro-examples),
  visiondummy (error-ellipse geometry).

## 3. Foundation (F) — Mathematical Core

Chapter notation table (all Thrun-compatible; the manifold symbols wait for Ch. 3):

| Symbol | Meaning |
|---|---|
| $X$, $x$ | random variable, value; $p(x)$ pmf/pdf |
| $p(x, y)$, $p(x \mid y)$ | joint, conditional |
| $\eta$ | normalizer $\left(\int p(z\mid x)p(x)\,dx\right)^{-1}$ — "compute the shape, normalize later" |
| $\mathbb{E}[X]$, $\mathrm{Var}[X]$, $\Sigma$ | expectation, variance, covariance matrix |
| $\mathcal{N}(x;\mu,\Sigma)$ | Gaussian density in moments form |
| $\Omega = \Sigma^{-1}$, $\xi = \Sigma^{-1}\mu$ | canonical (information) form |
| $d_M^2(x) = (x-\mu)^\top \Sigma^{-1} (x-\mu)$ | squared Mahalanobis distance |
| $H(X)$ | entropy; for continuous $X$, differential entropy |

**Definitions:** discrete/continuous random variable; joint and conditional probability;
independence and *conditional* independence ($p(x,y\mid z) = p(x\mid z)p(y\mid z)$ — stated with
the warning that it neither implies nor is implied by independence; this assumption powers every
sensor model in Ch. 10); theorem of total probability; Bayes rule and Bayes with background
knowledge $p(x\mid z, y) = \eta\, p(z \mid x, y)\, p(x \mid y)$; expectation and its linearity;
covariance $\Sigma = \mathbb{E}[(X-\mu)(X-\mu)^\top]$, positive semi-definiteness; the Gaussian in
1D and $n$D:

$$\mathcal{N}(x;\mu,\Sigma) = \det(2\pi\Sigma)^{-1/2} \exp\!\big(-\tfrac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\big)$$

canonical form $p(x) \propto \exp(-\tfrac12 x^\top \Omega x + x^\top \xi)$; entropy
$H = -\sum_x p(x)\log_2 p(x)$ (differential analog with the standard caveat that it can be
negative).

**Derivations** (each: statement → 3–8 step sketch inline → full algebra in a collapsible):

1. **Bayes rule from the definition of conditioning.** *Statement:* $p(x\mid z) = \eta\,p(z\mid x)p(x)$.
   *Sketch (3 steps):* symmetry of the joint; divide by $p(z)$; recognize $p(z)$ as a constant in
   $x$ and name it $1/\eta$. *Collapsible:* the background-knowledge version and the hallway
   numbers from Ch. 1 recomputed in this notation.
2. **Product of two 1D Gaussians is an (unnormalized) Gaussian.** *Statement:*
   $\mathcal{N}(x;\mu_1,\sigma_1^2)\cdot\mathcal{N}(x;\mu_2,\sigma_2^2) \propto \mathcal{N}(x;\mu,\sigma^2)$ with
   $\sigma^2 = (\sigma_1^{-2}+\sigma_2^{-2})^{-1}$, $\mu = \sigma^2(\mu_1/\sigma_1^2 + \mu_2/\sigma_2^2)$.
   *Sketch (4 steps):* add exponents; collect the quadratic in $x$; complete the square (the
   book's first use of the technique that will derive the Kalman filter); read off precision-
   weighted mean. *Collapsible:* full algebra + the observation that the leftover constant is
   exactly what $\eta$ absorbs.
3. **Multiplication is addition in canonical form.** *Statement:* if $p_i \propto \exp(-\tfrac12 x^\top\Omega_i x + x^\top\xi_i)$
   then $p_1 p_2$ has $\Omega = \Omega_1 + \Omega_2$, $\xi = \xi_1 + \xi_2$. *Sketch (2 steps):*
   exponents add; quadratic/linear coefficients add. Moral stated for Ch. 6: measurement updates
   are cheap in information form, predictions cheap in moments form.
4. **Linear transformation of a Gaussian.** *Statement:* $Y = AX + b$ with $X\sim\mathcal{N}(\mu,\Sigma)$
   gives $Y \sim \mathcal{N}(A\mu + b,\, A\Sigma A^\top)$. *Sketch (3 steps):* linearity of
   expectation; covariance definition; substitute. *Collapsible:* density-level proof via change of
   variables for invertible $A$; note that this plus Derivation 2 *is* the Kalman filter waiting
   to happen.
5. **Iso-density contours are ellipses aligned with eigenvectors.** *Statement:* $\{x : d_M^2(x) = c\}$
   is an ellipsoid with axes along eigenvectors of $\Sigma$, semi-axis lengths $\sqrt{c\,\lambda_i}$;
   for 95% coverage in 2D, $c = 5.991$ ($\chi^2_{2}$). *Sketch (4 steps):* spectral decomposition
   $\Sigma = V\Lambda V^\top$; rotate coordinates; the quadratic decouples; identify axis lengths.
   *Collapsible:* the $\chi^2$ coverage computation — this is the formula `draw_cov_ellipse` in
   widget-kit implements, so figure and math are literally the same computation.
6. **Marginals and conditionals of a joint Gaussian** (statement only + sketch; full proof is
   Appendix B). *Statement:* for partitioned $(x_a, x_b)$, the marginal is
   $\mathcal{N}(\mu_a, \Sigma_{aa})$ and the conditional is
   $\mathcal{N}\!\big(\mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(x_b - \mu_b),\ \Sigma_{aa} - \Sigma_{ab}\Sigma_{bb}^{-1}\Sigma_{ba}\big)$
   — the Schur complement's first appearance, flagged as "you will meet this matrix again in
   Ch. 6 (Kalman gain) and Ch. 15 (marginalizing map variables)."
7. **Entropy of a Gaussian.** *Statement:* $H = \tfrac12 \ln\det(2\pi e\,\Sigma)$. *Sketch (3
   steps):* write $-\mathbb{E}[\ln p]$; the quadratic term's expectation is $n/2$ by the trace
   trick; collect. *Collapsible:* full computation; note that entropy depends only on $\Sigma$ —
   uncertainty, not location.

**Named algorithms** (signatures Rust-flavored; complexity for state dimension $n$):

- `gaussian_product_canonical(g1: Canonical<N>, g2: Canonical<N>) -> Canonical<N>` — $O(n^2)$
  (two matrix adds); converting back to moments costs one $O(n^3)$ solve.
- `sample_mvn(mu, chol_l, rng) -> x` — draw $z \sim \mathcal{N}(0, I)$, return $\mu + Lz$ where
  $\Sigma = LL^\top$; $O(n^2)$ per sample after one $O(n^3)$ Cholesky.
- `mahalanobis2(g, x) -> f64` — via triangular solve against the cached Cholesky factor, $O(n^2)$;
  never form $\Sigma^{-1}$ explicitly (numerical-hygiene rule stated here, obeyed book-wide).
- `sample_normal_distribution(b)` (Thrun Table 5.4) — the historical sum-of-12-uniforms trick;
  presented in a "how it was done in 2000" box and contrasted with `rand_distr`'s Ziggurat
  `StandardNormal`; we test both and keep the Ziggurat.

## 4. Conceptual (C) — Intuition & Visual Design

One metaphor rules the chapter (bzarg's, extended): *a Gaussian is a fuzzy blob you can drag and
squash; Bayes rule multiplies blobs.* All widgets autoplay, expose one primary control, and share
the color code.

- **Widget w2.1: Gaussian Playground** (flagship) — type: interactive sim. A 2D Gaussian rendered
  three ways at once, linked: live scatter cloud (500 seeded samples), iso-ellipses (1σ/2σ/95%),
  and eigenvector arrows. Reader manipulates: drag $\mu$; drag two ellipse handles (stretch
  σ-axes); one correlation slider $\rho \in (-1, 1)$. Observes: cloud shears as $\rho$ moves;
  ellipse tilts; live readout of $\Sigma$, eigenvalues, and $\det\Sigma$; at $\rho \to \pm 1$ the
  ellipse degenerates to a line and the readout flags "singular — no density." Misconception
  killed: *ellipse axis lengths are the marginal standard deviations* (visibly false once tilted —
  marginals shown as blue strips on the axes stay fixed while the axes rotate).
- **Widget w2.2: Blob Multiplier** (flagship) — type: interactive sim. Two 1D Gaussians — prior
  (blue) and likelihood (green) — with the pointwise product (purple) computed live; below, a
  synchronized canonical-form readout showing $\Omega_1 + \Omega_2$ and $\xi_1 + \xi_2$ as bar
  stacks that literally add. Reader manipulates: drag either blob's mean or width (one blob at a
  time — the other freezes). Observes: the posterior is always *narrower than both* inputs; its
  mean sits closer to the confident one; when widths are equal it lands exactly halfway; dragging
  the likelihood infinitely wide makes the posterior collapse onto the prior. Autoplay default:
  the Ch. 1 door fusion (prior $\mathcal{N}(5, 4)$, likelihood $\mathcal{N}(6.5, 1)$ → posterior
  $\mathcal{N}(6.2, 0.8)$, the chapter's worked numbers). Misconception killed: *"Bayes fusion
  averages the two estimates"* — it precision-weights them, and certainty always increases.
- **Widget w2.3: Sampling Convergence** — type: interactive sim. Histogram of $N$ seeded samples
  against the true pdf; one log-scale slider $N \in [10, 10^5]$; seed reroll. A second pane shows
  the sample mean/covariance error vs. $N$ on a log-log plot with the $1/\sqrt{N}$ reference line.
  Misconception killed: *"my 50 samples look nothing like the distribution, so sampling is
  broken"* — and, in the other direction, *"more samples fix everything fast"* ($1/\sqrt{N}$ is
  slow). Seeds the particle-count intuitions of Ch. 8/12.
- **Widget w2.4: Slice vs. Squash** — type: interactive sim. The w2.1 joint Gaussian with two
  linked 1D panes: the **marginal** of $x_a$ (integrate $x_b$ out — "squash the cloud flat") and
  the **conditional** $p(x_a \mid x_b = \beta)$ ("slice the cloud") with a draggable slice line
  $\beta$. Observes: the conditional is narrower than the marginal whenever $\rho \neq 0$, and its
  mean tracks the slice — knowing a correlated quantity is information. Misconception killed:
  conditioning ≡ marginalizing. This widget is the Kalman update's geometry, one chapter early;
  its caption says so.

Dashboard layout: w2.1 opens the covariance section (play first, formalism after); w2.2 sits
directly under Derivation 2 with matched colors between the equation terms and the blobs; w2.4
accompanies Derivation 6. Static fallbacks: default-state SVGs rendered in CI from the same code.

## 5. Practical (P) — Rust Implementation

- **Crates:** `nalgebra` 0.35 (`SVector`/`SMatrix`, `Cholesky` — const-generic dimensions catch
  shape bugs at compile time); `rand` 0.9 + `rand_distr` 0.6 (seeded `SmallRng`, Ziggurat normal —
  WASM-clean with no `getrandom` config because everything is seeded); `statrs` 0.19 (its
  `MultivariateNormal` is the cross-check oracle in tests, *not* the implementation — we build our
  own so the reader understands every line); `plotters` (build-time SVG figures).
- **Module plan:** `crates/pr-core/src/prob/{gaussian.rs, canonical.rs, sample.rs}` — the first
  real deposit into the accumulating library (`pr-core`) that every later chapter imports;
  `demos/ch02-gauss/` (bins `w2-1-playground`, `w2-2-blob-multiplier`, `w2-3-sampling`,
  `w2-4-slice-squash`).

Key types & signatures (compiles-in-spirit):

```rust
// crates/pr-core/src/prob/gaussian.rs
use nalgebra::{Cholesky, Const, SMatrix, SVector};
use rand::Rng;

/// Moments-form Gaussian. Invariant: `cov` symmetric positive-definite (checked in `new`).
#[derive(Clone, Debug, PartialEq)]
pub struct Gaussian<const N: usize> {
    pub mean: SVector<f64, N>,
    pub cov: SMatrix<f64, N, N>,
}

impl<const N: usize> Gaussian<N> {
    pub fn new(mean: SVector<f64, N>, cov: SMatrix<f64, N, N>) -> Result<Self, NotPosDef>;
    pub fn ln_pdf(&self, x: &SVector<f64, N>) -> f64;              // via cached Cholesky
    pub fn pdf(&self, x: &SVector<f64, N>) -> f64;
    pub fn mahalanobis2(&self, x: &SVector<f64, N>) -> f64;        // triangular solve, no inverse
    pub fn sample<R: Rng>(&self, rng: &mut R) -> SVector<f64, N>;  // mu + L z
    pub fn entropy(&self) -> f64;                                  // 0.5 ln det(2πe Σ)
    /// Pushforward through y = A x + b   (Derivation 4).
    pub fn transform<const M: usize>(
        &self, a: &SMatrix<f64, M, N>, b: &SVector<f64, M>,
    ) -> Gaussian<M>;
    pub fn to_canonical(&self) -> Canonical<N>;
}

// crates/pr-core/src/prob/canonical.rs
/// Canonical (information) form: p(x) ∝ exp(-½ xᵀΩx + xᵀξ).  (TOC symbols ξ, Ω.)
#[derive(Clone, Debug, PartialEq)]
pub struct Canonical<const N: usize> {
    pub xi: SVector<f64, N>,       // ξ = Σ⁻¹ μ
    pub omega: SMatrix<f64, N, N>, // Ω = Σ⁻¹
}

impl<const N: usize> Canonical<N> {
    /// Bayes product: Ω and ξ simply add (Derivation 3). O(n²).
    pub fn product(&self, other: &Self) -> Self;
    pub fn to_moments(&self) -> Gaussian<N>; // one O(n³) solve
}

// crates/pr-core/src/prob/sample.rs
pub const BOOK_SEED: u64 = 0x5EED_2026; // every demo and test starts here; widgets display it
pub fn rng(seed: u64) -> rand::rngs::SmallRng; // SmallRng::seed_from_u64 — the only RNG source in the book
```

**Seeded-reproducibility discipline** (stated as a boxed rule): all randomness flows from
`prob::rng(seed)`; every widget shows its seed and has a reroll button; every test fixes its seed;
therefore every figure in the book is reproducible bit-for-bit, native and WASM alike.

**Worked end-to-end example** (`cargo run -p pr-core --example fuse_two_sensors`): fuse prior
$\mathcal{N}(5.0, 4.0)$ (blue) with likelihood $\mathcal{N}(6.5, 1.0)$ (green) via
`to_canonical → product → to_moments`; print posterior $\mathcal{N}(6.2, 0.8)$ (purple). Then the
2D act: sample 10 000 points from $\Sigma = \begin{bmatrix}4.0 & 1.9\\ 1.9 & 1.0\end{bmatrix}$ at
seed `BOOK_SEED`, print the sample covariance (agrees to ~2%), emit `figures/ch02_cloud.svg`
(cloud + 95% ellipse + eigenvectors — the w2.1 fallback image).

**Test-suite convention seeded here** (the chapter says this explicitly, and every later chapter
follows it): `#[test] fn worked_example_ch02()` asserts $\mu = 6.2$, $\sigma^2 = 0.8$ to 1e-12 —
the printed numbers, the widget default, and the test are one artifact. Additional tests:
round-trip `to_canonical ∘ to_moments = id` (property test over random SPD matrices);
`Gaussian::sample` χ² coverage test (95% ellipse contains ≈ 95% of 10⁵ samples); `ln_pdf`
cross-checked against `statrs::distribution::MultivariateNormal` at 20 random points.

**Runnable artifact:** the example above, plus the four WASM widgets; the chapter closes by noting
that w2.2's purple curve is computed by the very `Canonical::product` the reader just read.

## 6. Simulations & Animations Manifest

| id | title | type | crate stack | interaction | teaches |
|---|---|---|---|---|---|
| w2.1 | Gaussian Playground | wasm-sim | eframe 0.35 + egui_plot 0.34 + pr-core + widget-kit | drag μ, stretch axes, ρ slider, seed reroll | covariance geometry; eigenstructure; correlation ≠ axis lengths |
| w2.2 | Blob Multiplier | wasm-sim | eframe + egui_plot + pr-core + widget-kit | drag mean/width of either blob | Bayes = pointwise product; precision weighting; canonical form adds |
| w2.3 | Sampling Convergence | wasm-sim | eframe + egui_plot + pr-core + widget-kit | log-N slider, seed reroll | samples as representation; 1/√N convergence |
| w2.4 | Slice vs. Squash | wasm-sim | eframe + egui_plot + pr-core + widget-kit | drag slice line β, ρ slider | marginal vs. conditional; correlation is information |
| — | ch02_cloud.svg (+ per-widget fallbacks) | static-svg | plotters, CI build | none | static fallback discipline |

## 7. Exercises & Extensions

1. **(F)** Derive the product of two $n$D Gaussians in canonical form (three lines) and then in
   moments form (a page). Conclude in one sentence why Ch. 6 will offer two filters.
2. **(F)** Show that $\mathrm{Var}[X] \succeq 0$ always, and that the Gaussian entropy formula
   recovers $\tfrac12\ln(2\pi e\sigma^2)$ in 1D. For which $\sigma$ is differential entropy
   negative, and why is that not a contradiction?
3. **(C — predict, then verify with w2.2)** Set the likelihood's σ to triple the prior's. Predict
   the posterior mean's position as a fraction of the distance between the two means; verify.
   Then predict what happens as likelihood σ → ∞ and confirm the posterior→prior collapse.
4. **(C — w2.1/w2.4)** With $\rho = 0.95$, predict whether the conditional slice at $x_b = \mu_b$
   is narrower or wider than at $x_b = \mu_b + 2\sigma_b$. Verify with w2.4 and explain using
   Derivation 6 (the conditional covariance does not depend on $\beta$ — did your intuition say
   otherwise?).
5. **(P)** Implement `Gaussian::condition::<K>(&self, idx, value) -> Gaussian<{N-K}>` using the
   Schur complement (Derivation 6); property-test against `statrs` and against brute-force grid
   integration in 2D.
6. **(P)** Implement Thrun's `sample_normal_distribution(b)` (12-uniform trick) and benchmark
   moment accuracy vs. `rand_distr::StandardNormal` at $10^6$ samples (Criterion bench included in
   the repo); write two sentences on why the book uses the Ziggurat.

## 8. Modernization Notes

- Thrun et al. §2.2 is a six-page refresher: Bayes rule, η, independence, expectation, and a
  passing mention of entropy — no covariance geometry, no canonical form (deferred there to
  Ch. 3's information filter), no treatment of sampling as a *representation*. We promote all
  three because the book's widgets make them visual and because Ch. 6/8/15 lean on them heavily;
  pulling the canonical form forward makes the KF/IF duality a one-line observation later instead
  of a surprise.
- Added relative to baseline: Mahalanobis distance introduced here (Thrun defers to Ch. 7's data
  association) so gating in Ch. 11 is pure reuse; the χ²-coverage ellipse formula the figures
  actually use; explicit conditional-independence caveats with forward pointers to the beam-model
  overconfidence discussion (Ch. 10); RNG engineering (seeding, Ziggurat, WASM entropy) that no
  2005-era text could have needed.
- Deliberately *not* modernized: notation. We keep Thrun's $\eta$, $\mu, \Sigma$, $\xi, \Omega$
  exactly so readers can cross-read the original.
- Dropped: measure-theoretic foundations (a "further reading" pointer suffices for this
  audience); Choset-appendix breadth on general statistics (hypothesis testing etc.) — only what
  the book uses survives.
